"""
skills/metadata_langchain.py  — Stage D2: Metadata (LangChain Document.metadata)
──────────────────────────────────────────────────────────────────────────
Enriches the metadata already present on each LangChain Document/chunk
and exports a JSON side-file for inspection and auditing.

Metadata fields added / normalised
────────────────────────────────────
  source          — absolute file path (already set by A2)
  file_name       — basename (already set by A2)
  file_type       — extension without dot (already set by A2)
  chunk_index     — 0-based position within source document (set by B2)
  total_chunks    — total chunks in source document (set by B2)
  doc_index       — index of source doc in the ingestion batch (set by B2)
  char_count      — character length of this chunk's text
  word_count      — approximate word count of this chunk's text
  ingested_at     — ISO-8601 UTC timestamp of this run
  pipeline_run_id — short UUID identifying the current ingestion run

Optional domain fields (set when env vars are present or passed in)
────────────────────────────────────────────────────────────────────
  doc_category    — e.g. "sox_control", "audit_report", "gl_reconciliation"
  fiscal_period   — e.g. "Q3-2024"
  entity          — e.g. "APAC", "HoldCo"

Public API
──────────
  enrich_metadata(
      chunks:      list[Document],
      doc_category: str | None = None,
      fiscal_period: str | None = None,
      entity:       str | None = None,
      export_path:  str | Path | None = None,
  ) -> list[Document]

  export_metadata_json(
      chunks:     list[Document],
      path:       str | Path,
  ) -> Path
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

from skills.logger import StepLogger

log = StepLogger("metadata")


# ── internal helpers ──────────────────────────────────────────────────────

def _iso_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _word_count(text: str) -> int:
    return len(text.split())


# ── public API ────────────────────────────────────────────────────────────

def enrich_metadata(
    chunks: List[Document],
    doc_category:  Optional[str] = None,
    fiscal_period: Optional[str] = None,
    entity:        Optional[str] = None,
    export_path:   Optional[str | Path] = None,
) -> List[Document]:
    """
    Enrich metadata on every chunk and (optionally) export a JSON file.

    Parameters
    ----------
    chunks        : output of b2_chunk.split_documents()
    doc_category  : domain label (e.g. "sox_control")
    fiscal_period : fiscal period string (e.g. "Q3-2024")
    entity        : business entity (e.g. "APAC")
    export_path   : if given, write metadata.json to this path

    Returns
    -------
    The same list of Documents with enriched .metadata dicts (in-place).
    """
    log.start(f"Stage D2 — Enriching metadata for {len(chunks)} chunks")

    run_id      = str(uuid.uuid4())[:8]
    ingested_at = _iso_now()

    # Resolve optional domain fields from env if not passed explicitly
    doc_category  = doc_category  or os.getenv("DOC_CATEGORY")
    fiscal_period = fiscal_period or os.getenv("FISCAL_PERIOD")
    entity        = entity        or os.getenv("ENTITY")

    for i, chunk in enumerate(chunks):
        m = chunk.metadata

        # ── computed fields ───────────────────────────────────────────
        m["char_count"]      = len(chunk.page_content)
        m["word_count"]      = _word_count(chunk.page_content)
        m["ingested_at"]     = ingested_at
        m["pipeline_run_id"] = run_id

        # Rename LangChain's 'start_index' → 'char_offset' for clarity
        if "start_index" in m:
            m["char_offset"] = m.pop("start_index")

        # ── optional domain fields ─────────────────────────────────────
        if doc_category:
            m["doc_category"]  = doc_category
        if fiscal_period:
            m["fiscal_period"] = fiscal_period
        if entity:
            m["entity"]        = entity

        if i % 50 == 0 or i == len(chunks) - 1:
            log.step(
                f"  chunk {i+1}/{len(chunks)}  "
                f"file={m.get('file_name','?')}  "
                f"chars={m['char_count']}"
            )

    log.done(
        f"D2 metadata enriched — run_id={run_id}, ingested_at={ingested_at}"
    )

    if export_path:
        export_metadata_json(chunks, export_path)

    return chunks


def export_metadata_json(
    chunks: List[Document],
    path:   str | Path,
) -> Path:
    """
    Write chunk metadata (without page_content) to a JSON file.

    Useful for auditing, debugging, and SOX traceability.

    Returns the resolved Path of the written file.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for i, chunk in enumerate(chunks):
        rec = {"chunk_seq": i, **chunk.metadata}
        # Add a short text preview for readability
        rec["text_preview"] = chunk.page_content[:120].replace("\n", " ")
        records.append(rec)

    out.write_text(
        json.dumps(records, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    log.done(f"Metadata JSON exported → {out}  ({len(records)} records)")
    return out
