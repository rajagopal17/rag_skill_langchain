"""
main.py  — RAG Pipeline FastAPI Application
────────────────────────────────────────────
Exposes the full pipeline as a REST API.

Endpoints
─────────
  POST  /ingest
        Upload one or more files and run the full pipeline:
        A2 → B2 → C3 → D2 → E1
        Supports mode: insert | upsert | replace

  POST  /update
        Update (upsert) additional documents into an existing table.
        Same pipeline but mode is always "upsert".

  POST  /query
        Ask a question.  Returns the answer + retrieved chunks.

  GET   /health
        Service health check + DB connectivity test.

  GET   /docs
        Auto-generated Swagger UI (built into FastAPI).

Run
───
  uvicorn main:app --reload --port 8000

Environment (.env)
──────────────────
  OPENAI_API_KEY
  PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD
  PG_TABLE            (default: rag_chunks)
  OPENAI_CHAT_MODEL   (default: gpt-4o-mini)
  OPENAI_EMBED_MODEL  (default: text-embedding-3-small)
  RAG_TOP_K           (default: 5)
  DOC_CATEGORY        (optional, applied to all uploads)
  FISCAL_PERIOD       (optional)
  ENTITY              (optional)
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Literal, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from skills.data_ingestion_langchain import load_documents
from skills.chunking_langchain       import split_documents
from skills.embedding_openai         import embed_chunks
from skills.metadata_langchain       import enrich_metadata, export_metadata_json
from skills.vector_store_pgvector    import upload_chunks
from skills.rag_query_engine         import answer_question, RAGResult
from skills.logger     import get_logger

load_dotenv()

log = get_logger("api")
app = FastAPI(
    title       = "RAG Pipeline API",
    description = "Ingest documents, build pgvector embeddings, query with OpenAI.",
    version     = "1.0.0",
)

DEFAULT_TABLE = os.getenv("PG_TABLE", "rag_chunks")


# ══════════════════════════════════════════════════════════════════════════
# Request / Response models
# ══════════════════════════════════════════════════════════════════════════

class IngestResponse(BaseModel):
    status:          str
    table:           str
    mode:            str
    files_received:  int
    documents_loaded: int
    chunks_created:  int
    rows_written:    int
    metadata_export: Optional[str] = None
    pipeline_run_id: Optional[str] = None


class QueryRequest(BaseModel):
    question:        str = Field(..., min_length=3, description="Natural language question")
    table:           str = Field(DEFAULT_TABLE, description="pgvector table to search")
    top_k:           int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    filter_doc_category:  Optional[str] = Field(None, description="Filter by doc_category")
    filter_fiscal_period: Optional[str] = Field(None, description="Filter by fiscal_period")
    filter_entity:        Optional[str] = Field(None, description="Filter by entity")


class ChunkResult(BaseModel):
    uuid:        Optional[str]
    source:      str
    file_name:   str
    chunk_index: int
    score:       float
    preview:     str


class QueryResponse(BaseModel):
    question:          str
    answer:            str
    model:             str
    chunks_used:       int
    elapsed_s:         float
    prompt_tokens:     int
    completion_tokens: int
    retrieved_chunks:  List[ChunkResult]


class HealthResponse(BaseModel):
    status:    str
    db:        str
    openai_key: bool


# ══════════════════════════════════════════════════════════════════════════
# Helper: run the full ingestion pipeline
# ══════════════════════════════════════════════════════════════════════════

def _run_ingestion(
    upload_dir:    Path,
    table:         str,
    mode:          str,
    doc_category:  Optional[str],
    fiscal_period: Optional[str],
    entity:        Optional[str],
    export_meta:   bool,
) -> dict:
    """Run A2→B2→C3→D2→E1 and return summary dict."""
    log.info(f"[pipeline] start  table={table}  mode={mode}  dir={upload_dir}")

    # A2 — ingest
    docs = load_documents(upload_dir)
    if not docs:
        raise HTTPException(status_code=422, detail="No text could be extracted from the uploaded files.")

    # B2 — chunk
    chunks = split_documents(docs)

    # C3 — embed
    embedded = embed_chunks(chunks)

    # D2 — metadata
    export_path = Path("logs") / f"metadata_{table}.json" if export_meta else None
    enrich_metadata(
        [ec.document for ec in embedded],
        doc_category  = doc_category,
        fiscal_period = fiscal_period,
        entity        = entity,
        export_path   = export_path,
    )

    # Grab run_id from first chunk metadata
    run_id = embedded[0].document.metadata.get("pipeline_run_id") if embedded else None

    # E1 — upload
    rows = upload_chunks(embedded, table=table, mode=mode)

    log.info(
        f"[pipeline] done  docs={len(docs)}  chunks={len(chunks)}  "
        f"rows={rows}  run_id={run_id}"
    )
    return {
        "documents_loaded": len(docs),
        "chunks_created":   len(chunks),
        "rows_written":     rows,
        "metadata_export":  str(export_path) if export_path else None,
        "pipeline_run_id":  run_id,
    }


# ══════════════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════════════

@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest(
    files:         List[UploadFile] = File(..., description="One or more documents to ingest"),
    table:         str  = Form(DEFAULT_TABLE, description="Target pgvector table"),
    mode:          Literal["insert","upsert","replace"] = Form("insert",
                       description="insert=append, upsert=update existing, replace=delete+reinsert"),
    doc_category:  Optional[str] = Form(None, description="Domain label, e.g. sox_control"),
    fiscal_period: Optional[str] = Form(None, description="e.g. Q3-2024"),
    entity:        Optional[str] = Form(None, description="Business entity, e.g. APAC"),
    export_metadata: bool        = Form(True,  description="Export metadata JSON to logs/"),
):
    """
    **Full ingestion pipeline**: upload files → chunk → embed → store.

    - `mode=insert`  — always append new rows
    - `mode=upsert`  — update existing rows matched on (source, chunk_index)
    - `mode=replace` — delete all rows for each uploaded source, then insert fresh
    """
    log.info(f"POST /ingest  files={[f.filename for f in files]}  mode={mode}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for uf in files:
            dest = tmp_dir / uf.filename
            dest.write_bytes(await uf.read())
            log.info(f"  saved upload: {uf.filename} ({dest.stat().st_size:,} bytes)")

        summary = _run_ingestion(
            upload_dir    = tmp_dir,
            table         = table,
            mode          = mode,
            doc_category  = doc_category,
            fiscal_period = fiscal_period,
            entity        = entity,
            export_meta   = export_metadata,
        )

    return IngestResponse(
        status           = "ok",
        table            = table,
        mode             = mode,
        files_received   = len(files),
        **summary,
    )


@app.post("/update", response_model=IngestResponse, tags=["Ingestion"])
async def update_documents(
    files:         List[UploadFile] = File(..., description="Additional or updated documents"),
    table:         str  = Form(DEFAULT_TABLE),
    doc_category:  Optional[str] = Form(None),
    fiscal_period: Optional[str] = Form(None),
    entity:        Optional[str] = Form(None),
    replace_source: bool = Form(False,
        description="True = delete+replace rows for each file; False = upsert"),
):
    """
    **Add or update documents** in an existing table.

    Use this endpoint to:
    - Add new documents to an existing knowledge base.
    - Re-ingest updated versions of existing documents.

    Set `replace_source=true` to fully replace all chunks from a given source file;
    `false` (default) will upsert matching (source, chunk_index) rows.
    """
    mode = "replace" if replace_source else "upsert"
    log.info(f"POST /update  files={[f.filename for f in files]}  mode={mode}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for uf in files:
            dest = tmp_dir / uf.filename
            dest.write_bytes(await uf.read())

        summary = _run_ingestion(
            upload_dir    = tmp_dir,
            table         = table,
            mode          = mode,
            doc_category  = doc_category,
            fiscal_period = fiscal_period,
            entity        = entity,
            export_meta   = True,
        )

    return IngestResponse(
        status         = "ok",
        table          = table,
        mode           = mode,
        files_received = len(files),
        **summary,
    )


@app.get("/query", response_model=QueryResponse, tags=["Query"])
async def query_get(
    question: str,
    table:    str = DEFAULT_TABLE,
    top_k:    int = 5,
):
    """
    **Ask a question via GET** — convenient for browser/curl testing.

    Example: `/query?question=What are the fixed asset categories?`
    """
    req = QueryRequest(question=question, table=table, top_k=top_k)
    return await query(req)


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(req: QueryRequest):
    """
    **Ask a question** and get an answer grounded in your documents.

    Optionally filter by `doc_category`, `fiscal_period`, or `entity`
    to narrow retrieval to a specific subset of your knowledge base.
    """
    log.info(f"POST /query  question='{req.question[:60]}…'  table={req.table}")

    filter_meta = {
        k: v for k, v in {
            "doc_category":  req.filter_doc_category,
            "fiscal_period": req.filter_fiscal_period,
            "entity":        req.filter_entity,
        }.items() if v
    } or None

    try:
        result: RAGResult = answer_question(
            question        = req.question,
            table           = req.table,
            top_k           = req.top_k,
            filter_metadata = filter_meta,
        )
    except Exception as exc:
        log.error(f"Query failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    return QueryResponse(
        question          = result.question,
        answer            = result.answer,
        model             = result.model,
        chunks_used       = result.chunks_used,
        elapsed_s         = round(result.elapsed_s, 2),
        prompt_tokens     = result.prompt_tokens,
        completion_tokens = result.completion_tokens,
        retrieved_chunks  = [
            ChunkResult(
                uuid        = c.metadata.get("uuid"),
                source      = c.source,
                file_name   = c.file_name,
                chunk_index = c.chunk_index,
                score       = round(c.score, 4),
                preview     = c.content[:200],
            )
            for c in result.retrieved
        ],
    )


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Quick health check — tests DB connectivity and OpenAI key presence."""
    import psycopg2
    db_status = "ok"
    try:
        conn = psycopg2.connect(
            host     = os.getenv("PG_HOST",     "localhost"),
            port     = int(os.getenv("PG_PORT", "5432")),
            dbname   = os.getenv("PG_DB",       "ragdb"),
            user     = os.getenv("PG_USER",     "postgres"),
            password = os.getenv("PG_PASSWORD", ""),
            connect_timeout = 3,
        )
        conn.close()
    except Exception as exc:
        db_status = f"error: {exc}"

    return HealthResponse(
        status     = "ok",
        db         = db_status,
        openai_key = bool(os.getenv("OPENAI_API_KEY")),
    )


# ── entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host    = "0.0.0.0",
        port    = int(os.getenv("PORT", "8000")),
        reload  = os.getenv("ENV", "dev") == "dev",
        log_level = "info",
    )
