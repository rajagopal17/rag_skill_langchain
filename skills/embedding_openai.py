"""
skills/embedding_openai.py  — Stage C3: Embedding (OpenAI text-embedding-3-small)
──────────────────────────────────────────────────────────────────────────
Converts a list of LangChain Documents (chunks) into float32 embeddings
using the OpenAI text-embedding-3-small model.

Features
────────
  • Batched API calls (configurable, default 100 chunks per request)
  • Automatic retry with exponential back-off (via tenacity)
  • Token-usage logging per batch
  • Returns list[EmbeddedChunk] — a light dataclass pairing each
    Document with its embedding vector

Configuration
─────────────
  OPENAI_API_KEY  — required env variable (or set in .env)
  EMBED_MODEL     = "text-embedding-3-small"  (1 536-dim, cost-efficient)
  EMBED_BATCH     = 100  chunks per API call

Public API
──────────
  embed_chunks(
      chunks: list[Document],
      batch_size: int = 100,
  ) -> list[EmbeddedChunk]
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from skills.logger import StepLogger

log = StepLogger("embedding")

# ── config ────────────────────────────────────────────────────────────────
EMBED_MODEL   = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
EMBED_BATCH   = int(os.getenv("EMBED_BATCH_SIZE", "100"))
EMBED_DIMS    = 1536   # fixed for text-embedding-3-small


# ── result type ───────────────────────────────────────────────────────────
@dataclass
class EmbeddedChunk:
    """A chunk paired with its embedding vector."""
    document:  Document
    embedding: List[float]
    model:     str = EMBED_MODEL

    @property
    def metadata(self) -> dict:
        return self.document.metadata

    @property
    def text(self) -> str:
        return self.document.page_content


# ── retry-wrapped embed call ──────────────────────────────────────────────
@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _embed_batch(texts: List[str], embedder: OpenAIEmbeddings) -> List[List[float]]:
    return embedder.embed_documents(texts)


# ── public API ────────────────────────────────────────────────────────────
def embed_chunks(
    chunks: List[Document],
    batch_size: int = EMBED_BATCH,
) -> List[EmbeddedChunk]:
    """
    Embed every chunk in *chunks* and return EmbeddedChunk objects.

    Parameters
    ----------
    chunks     : output of b2_chunk.split_documents()
    batch_size : chunks per OpenAI API call (default 100)

    Returns
    -------
    list[EmbeddedChunk]
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Add it to your .env file."
        )

    log.start(
        f"Stage C3 — Embedding {len(chunks)} chunks  "
        f"[model={EMBED_MODEL}, batch={batch_size}]"
    )

    embedder = OpenAIEmbeddings(
        model=EMBED_MODEL,
        openai_api_key=api_key,
    )

    results: List[EmbeddedChunk] = []
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    t0 = time.perf_counter()

    for batch_num, start in enumerate(range(0, len(chunks), batch_size), 1):
        batch = chunks[start : start + batch_size]
        texts = [c.page_content for c in batch]

        log.step(
            f"Batch {batch_num}/{total_batches}  "
            f"({len(texts)} chunks, chars {sum(len(t) for t in texts):,})"
        )

        vectors = _embed_batch(texts, embedder)

        for chunk, vec in zip(batch, vectors):
            results.append(EmbeddedChunk(document=chunk, embedding=vec))

    elapsed = time.perf_counter() - t0
    log.done(
        f"C3 complete — {len(results)} embeddings  "
        f"[dim={EMBED_DIMS}, elapsed={elapsed:.1f}s]"
    )
    return results
