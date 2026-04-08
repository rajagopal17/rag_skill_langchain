"""
skills/vector_store_pgvector.py  — Stage E1: pgvector Upload (native psycopg2)
─────────────────────────────────────────────────────────────────────
Stores EmbeddedChunk objects in a PostgreSQL table that has the
pgvector extension enabled.

Table schema (auto-created on first run)
────────────────────────────────────────
  CREATE TABLE <table> (
      id              SERIAL PRIMARY KEY,
      source          TEXT,
      file_name       TEXT,
      file_type       TEXT,
      chunk_index     INT,
      doc_index       INT,
      char_count      INT,
      word_count      INT,
      char_offset     INT,
      doc_category    TEXT,
      fiscal_period   TEXT,
      entity          TEXT,
      ingested_at     TIMESTAMPTZ,
      pipeline_run_id TEXT,
      content         TEXT,
      embedding       VECTOR(1536)
  );

Update / append mode
────────────────────
  mode="insert"   — always inserts new rows  (default for first load)
  mode="upsert"   — INSERT … ON CONFLICT (source, chunk_index) DO UPDATE
                    Use this when re-ingesting updated documents.
  mode="replace"  — DELETE rows with matching source, then INSERT fresh rows
                    Use when you want a clean replacement for specific files.

Connection
──────────
  PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD  (env / .env)
  PG_TABLE   — target table name (default "rag_chunks")

Public API
──────────
  init_table(conn, table: str)
  upload_chunks(
      embedded: list[EmbeddedChunk],
      table:    str = "rag_chunks",
      mode:     Literal["insert","upsert","replace"] = "insert",
  ) -> int   # rows written
"""

from __future__ import annotations

import os
from typing import List, Literal

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

from skills.embedding_openai import EmbeddedChunk
from skills.logger import StepLogger

log = StepLogger("vector_store")

# ── defaults ──────────────────────────────────────────────────────────────
DEFAULT_TABLE  = os.getenv("PG_TABLE", "rag_chunks")
VECTOR_DIM     = 1536
BATCH_SIZE     = int(os.getenv("PG_BATCH_SIZE", "200"))


# ── connection ────────────────────────────────────────────────────────────

def _get_connection():
    conn = psycopg2.connect(
        host     = os.getenv("PG_HOST",     "localhost"),
        port     = int(os.getenv("PG_PORT", "5432")),
        dbname   = os.getenv("PG_DB",       "ragdb"),
        user     = os.getenv("PG_USER",     "postgres"),
        password = os.getenv("PG_PASSWORD", ""),
    )
    register_vector(conn)
    return conn


# ── DDL ───────────────────────────────────────────────────────────────────
_CREATE_EXTENSION = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
"""

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS {table} (
    id              SERIAL PRIMARY KEY,
    uuid            UUID         NOT NULL DEFAULT uuid_generate_v4(),
    source          TEXT,
    file_name       TEXT,
    file_type       TEXT,
    chunk_index     INT,
    doc_index       INT,
    char_count      INT,
    word_count      INT,
    char_offset     INT,
    doc_category    TEXT,
    fiscal_period   TEXT,
    entity          TEXT,
    ingested_at     TIMESTAMPTZ,
    pipeline_run_id TEXT,
    content         TEXT         NOT NULL,
    embedding       VECTOR({dim}) NOT NULL
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS {idx}_embedding_idx
    ON {table} USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
"""

_UNIQUE_CONSTRAINT = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = '{safe}_source_chunk_uniq'
    ) THEN
        ALTER TABLE {table}
        ADD CONSTRAINT {safe}_source_chunk_uniq
        UNIQUE (source, chunk_index);
    END IF;
END$$;
"""


def _safe_name(table: str) -> str:
    """Return a safe SQL identifier name (for indexes/constraints) by replacing hyphens."""
    return table.replace("-", "_")


def init_table(conn, table: str = DEFAULT_TABLE) -> None:
    """Create the pgvector extension + table + index if they don't exist."""
    log.step(f"Initialising table '{table}' ...")
    safe = _safe_name(table)
    quoted = f'"{table}"'
    with conn.cursor() as cur:
        cur.execute(_CREATE_EXTENSION)
        cur.execute(_CREATE_TABLE.format(table=quoted, dim=VECTOR_DIM))
        cur.execute(_CREATE_INDEX.format(table=quoted, idx=safe))
        cur.execute(_UNIQUE_CONSTRAINT.format(table=quoted, safe=safe))
    conn.commit()
    log.step(f"Table '{table}' ready (extension + index ensured)")


# ── SQL templates ─────────────────────────────────────────────────────────
_COLS = """
    uuid, source, file_name, file_type, chunk_index, doc_index,
    char_count, word_count, char_offset,
    doc_category, fiscal_period, entity,
    ingested_at, pipeline_run_id,
    content, embedding
"""

_INSERT_SQL = f"""
INSERT INTO {{table}} ({_COLS})
VALUES (uuid_generate_v4(),%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
"""

_UPSERT_SQL = f"""
INSERT INTO {{table}} ({_COLS})
VALUES (uuid_generate_v4(),%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
ON CONFLICT (source, chunk_index) DO UPDATE SET
    uuid            = uuid_generate_v4(),
    file_name       = EXCLUDED.file_name,
    char_count      = EXCLUDED.char_count,
    word_count      = EXCLUDED.word_count,
    char_offset     = EXCLUDED.char_offset,
    ingested_at     = EXCLUDED.ingested_at,
    pipeline_run_id = EXCLUDED.pipeline_run_id,
    content         = EXCLUDED.content,
    embedding       = EXCLUDED.embedding,
    doc_category    = EXCLUDED.doc_category,
    fiscal_period   = EXCLUDED.fiscal_period,
    entity          = EXCLUDED.entity;
"""

_DELETE_BY_SOURCE = "DELETE FROM {table} WHERE source = %s;"


# ── row builder ───────────────────────────────────────────────────────────

def _to_row(ec: EmbeddedChunk) -> tuple:
    m = ec.metadata
    return (
        m.get("source"),
        m.get("file_name"),
        m.get("file_type"),
        m.get("chunk_index"),
        m.get("doc_index"),
        m.get("char_count"),
        m.get("word_count"),
        m.get("char_offset"),
        m.get("doc_category"),
        m.get("fiscal_period"),
        m.get("entity"),
        m.get("ingested_at"),
        m.get("pipeline_run_id"),
        ec.text,
        ec.embedding,
    )


# ── public API ────────────────────────────────────────────────────────────

def upload_chunks(
    embedded: List[EmbeddedChunk],
    table:    str = DEFAULT_TABLE,
    mode:     Literal["insert", "upsert", "replace"] = "insert",
) -> int:
    """
    Write *embedded* chunks to PostgreSQL.

    Parameters
    ----------
    embedded : output of c3_embed.embed_chunks()
    table    : target table name
    mode     : "insert" | "upsert" | "replace"

    Returns
    -------
    Number of rows written.
    """
    log.start(
        f"Stage E1 — Uploading {len(embedded)} rows to '{table}'  [mode={mode}]"
    )

    conn = _get_connection()
    try:
        init_table(conn, table)

        quoted = f'"{table}"'
        # ── replace: delete existing rows for each unique source ──────
        if mode == "replace":
            sources = {ec.metadata.get("source") for ec in embedded}
            with conn.cursor() as cur:
                for src in sources:
                    cur.execute(_DELETE_BY_SOURCE.format(table=quoted), (src,))
                    log.step(f"Deleted existing rows for source: {src}")
            conn.commit()
            sql = _INSERT_SQL.format(table=quoted)
        elif mode == "upsert":
            sql = _UPSERT_SQL.format(table=quoted)
        else:
            sql = _INSERT_SQL.format(table=quoted)

        # ── batch insert ──────────────────────────────────────────────
        rows_written = 0
        total_batches = (len(embedded) + BATCH_SIZE - 1) // BATCH_SIZE

        with conn.cursor() as cur:
            for b_num, start in enumerate(range(0, len(embedded), BATCH_SIZE), 1):
                batch = embedded[start : start + BATCH_SIZE]
                rows  = [_to_row(ec) for ec in batch]
                psycopg2.extras.execute_batch(cur, sql, rows, page_size=BATCH_SIZE)
                rows_written += len(rows)
                log.step(
                    f"Batch {b_num}/{total_batches} — "
                    f"{rows_written}/{len(embedded)} rows committed"
                )
                conn.commit()

    finally:
        conn.close()

    log.done(
        f"E1 complete — {rows_written} rows written to '{table}'  [mode={mode}]"
    )
    return rows_written
