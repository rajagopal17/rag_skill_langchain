"""
skills/rag_query_engine.py  — Stage F1: RAG Query Engine (native Python + OpenAI)
────────────────────────────────────────────────────────────────────────
Retrieves relevant chunks from pgvector and generates answers with the
OpenAI Chat Completions API — no LangChain dependency at query time.

Pipeline
────────
  1. Embed the user's question  (text-embedding-3-small)
  2. Run cosine-similarity search against pgvector
  3. Build a prompt with the retrieved context
  4. Call gpt-4o-mini (or configured model) and return the answer

Configuration (env / .env)
──────────────────────────
  OPENAI_API_KEY
  OPENAI_CHAT_MODEL  = "gpt-4o-mini"
  OPENAI_EMBED_MODEL = "text-embedding-3-small"
  PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD
  PG_TABLE           = "rag_chunks"
  RAG_TOP_K          = 5     (chunks retrieved per query)
  RAG_MAX_CONTEXT    = 4000  (max chars of context fed to LLM)

Public API
──────────
  answer_question(
      question:        str,
      table:           str = "rag_chunks",
      top_k:           int = 5,
      filter_metadata: dict | None = None,
  ) -> RAGResult
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from skills.logger import StepLogger

log = StepLogger("rag_query")

# ── config ────────────────────────────────────────────────────────────────
CHAT_MODEL   = os.getenv("OPENAI_CHAT_MODEL",  "gpt-4o-mini")
EMBED_MODEL  = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
DEFAULT_TABLE = os.getenv("PG_TABLE",          "rag_chunks")
TOP_K        = int(os.getenv("RAG_TOP_K",      "5"))
MAX_CONTEXT  = int(os.getenv("RAG_MAX_CONTEXT","4000"))


# ── result type ───────────────────────────────────────────────────────────
@dataclass
class RetrievedChunk:
    content:    str
    score:      float
    source:     str
    file_name:  str
    chunk_index: int
    metadata:   Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResult:
    question:        str
    answer:          str
    model:           str
    chunks_used:     int
    retrieved:       List[RetrievedChunk]
    prompt_tokens:   int = 0
    completion_tokens: int = 0
    elapsed_s:       float = 0.0


# ── OpenAI client (lazy singleton) ───────────────────────────────────────
_client: Optional[OpenAI] = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set.")
        _client = OpenAI(api_key=api_key)
    return _client


# ── pgvector connection ───────────────────────────────────────────────────
def _get_pg_conn():
    conn = psycopg2.connect(
        host     = os.getenv("PG_HOST",     "localhost"),
        port     = int(os.getenv("PG_PORT", "5432")),
        dbname   = os.getenv("PG_DB",       "ragdb"),
        user     = os.getenv("PG_USER",     "postgres"),
        password = os.getenv("PG_PASSWORD", ""),
    )
    register_vector(conn)
    return conn


# ── step 1: embed question ────────────────────────────────────────────────
@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(min=1, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _embed_question(question: str) -> List[float]:
    client = _get_client()
    resp = client.embeddings.create(model=EMBED_MODEL, input=question)
    return resp.data[0].embedding


# ── step 2: retrieve chunks ───────────────────────────────────────────────
_BASE_SEARCH = """
SELECT
    uuid,
    content,
    source,
    file_name,
    chunk_index,
    doc_category,
    fiscal_period,
    entity,
    ingested_at,
    1 - (embedding <=> %s::vector) AS score
FROM {table}
{where_clause}
ORDER BY embedding <=> %s::vector
LIMIT %s;
"""


def _build_filter(filter_metadata: Optional[dict]) -> tuple[str, list]:
    """Build optional WHERE clause from metadata key-value pairs."""
    if not filter_metadata:
        return "", []
    clauses, params = [], []
    for k, v in filter_metadata.items():
        clauses.append(f"{k} = %s")
        params.append(v)
    return "WHERE " + " AND ".join(clauses), params


def _retrieve_chunks(
    query_vec: List[float],
    table: str,
    top_k: int,
    filter_metadata: Optional[dict],
) -> List[RetrievedChunk]:
    conn = _get_pg_conn()
    try:
        where_clause, filter_params = _build_filter(filter_metadata)
        sql = _BASE_SEARCH.format(table=f'"{table}"', where_clause=where_clause)
        params = filter_params + [query_vec, query_vec, top_k]

        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return [
            RetrievedChunk(
                content     = row["content"],
                score       = float(row["score"]),
                source      = row["source"] or "",
                file_name   = row["file_name"] or "",
                chunk_index = row["chunk_index"] or 0,
                metadata    = {
                    k: str(row[k]) if k == "uuid" else row[k]
                    for k in ("uuid","doc_category","fiscal_period","entity","ingested_at")
                    if row.get(k)
                },
            )
            for row in rows
        ]
    finally:
        conn.close()


# ── step 3: build prompt ──────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are a knowledgeable assistant. Answer the user's
question using ONLY the context provided below. If the context does not
contain enough information, say so clearly. Always cite the source file
name when referencing specific information.

Context:
────────
{context}
────────"""


def _build_prompt(question: str, chunks: List[RetrievedChunk]) -> str:
    parts, total = [], 0
    for c in chunks:
        snippet = f"[{c.file_name} | chunk {c.chunk_index} | score {c.score:.3f}]\n{c.content}"
        if total + len(snippet) > MAX_CONTEXT:
            break
        parts.append(snippet)
        total += len(snippet)
    context = "\n\n".join(parts)
    return _SYSTEM_PROMPT.format(context=context)


# ── step 4: generate answer ───────────────────────────────────────────────
@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(min=1, max=15),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _generate(system_prompt: str, question: str) -> tuple[str, int, int]:
    client = _get_client()
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": question},
        ],
        temperature=0.2,
    )
    answer = resp.choices[0].message.content
    usage  = resp.usage
    return answer, usage.prompt_tokens, usage.completion_tokens


# ── public API ────────────────────────────────────────────────────────────

def answer_question(
    question:        str,
    table:           str = DEFAULT_TABLE,
    top_k:           int = TOP_K,
    filter_metadata: Optional[dict] = None,
) -> RAGResult:
    """
    Answer *question* using RAG over the pgvector table.

    Parameters
    ----------
    question        : the user's natural language question
    table           : pgvector table to search
    top_k           : number of chunks to retrieve
    filter_metadata : optional dict for metadata filtering
                      e.g. {"doc_category": "sox_control", "entity": "APAC"}

    Returns
    -------
    RAGResult with answer, retrieved chunks, and token usage
    """
    t0 = time.perf_counter()
    log.start(f"Stage F1 — Query: '{question[:80]}…'" if len(question) > 80 else f"Stage F1 — Query: '{question}'")

    # Step 1: Embed question
    log.step("Embedding question ...")
    q_vec = _embed_question(question)
    log.step("Question embedded")

    # Step 2: Retrieve chunks
    log.step(f"Retrieving top-{top_k} chunks from '{table}' ...")
    if filter_metadata:
        log.step(f"Metadata filter: {filter_metadata}")
    chunks = _retrieve_chunks(q_vec, table, top_k, filter_metadata)
    log.step(
        f"Retrieved {len(chunks)} chunks  "
        f"[scores: {[round(c.score,3) for c in chunks]}]"
    )

    # Step 3: Build prompt
    log.step("Building prompt with context ...")
    system_prompt = _build_prompt(question, chunks)
    context_chars = len(system_prompt)
    log.step(f"Prompt ready — {context_chars:,} chars context")

    # Step 4: Generate answer
    log.step(f"Calling {CHAT_MODEL} for answer ...")
    answer, prompt_tok, compl_tok = _generate(system_prompt, question)
    elapsed = time.perf_counter() - t0

    log.done(
        f"F1 complete — {len(answer)} chars answer  "
        f"[tokens: {prompt_tok}→{compl_tok}, elapsed={elapsed:.1f}s]"
    )

    return RAGResult(
        question          = question,
        answer            = answer,
        model             = CHAT_MODEL,
        chunks_used       = len(chunks),
        retrieved         = chunks,
        prompt_tokens     = prompt_tok,
        completion_tokens = compl_tok,
        elapsed_s         = elapsed,
    )
