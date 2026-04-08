"""
skills/data_ingestion_langchain.py  — Stage A2: Document Ingestion (LangChain loaders)
────────────────────────────────────────────────────────────────────────
Reads PDF, DOCX, XLSX, CSV, PPTX, TXT from a file or directory and
returns a list of LangChain Document objects (raw text + basic metadata).

Supported formats
─────────────────
  .pdf   → PyPDFLoader  (page-level documents)
  .docx  → Docx2txtLoader
  .xlsx  → UnstructuredExcelLoader
  .csv   → CSVLoader
  .pptx  → UnstructuredPowerPointLoader
  .txt   → TextLoader
  *      → UnstructuredFileLoader  (fallback)

Public API
──────────
  load_documents(source: str | Path) -> list[Document]
      source can be a single file path or a directory.
      When a directory is given every supported file inside is loaded.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import List

from langchain_core.documents import Document

# LangChain community loaders
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredFileLoader,
    UnstructuredPowerPointLoader,
)

from skills.logger import StepLogger

log = StepLogger("data_ingestion")

# ── supported extensions ──────────────────────────────────────────────────
_LOADER_MAP: dict[str, type] = {
    ".pdf":  PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".xls":  UnstructuredExcelLoader,
    ".csv":  CSVLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".txt":  TextLoader,
    ".md":   TextLoader,
}

SUPPORTED_EXTENSIONS = set(_LOADER_MAP.keys())


# ── internal helpers ──────────────────────────────────────────────────────

def _load_file(path: Path) -> List[Document]:
    """Load a single file and return LangChain Documents."""
    ext = path.suffix.lower()
    loader_cls = _LOADER_MAP.get(ext, UnstructuredFileLoader)

    try:
        if loader_cls is CSVLoader:
            loader = CSVLoader(str(path), encoding="utf-8")
        else:
            loader = loader_cls(str(path))

        docs = loader.load()

        # Normalise metadata: add source + file_name on every doc
        for doc in docs:
            doc.metadata.setdefault("source", str(path))
            doc.metadata.setdefault("file_name", path.name)
            doc.metadata.setdefault("file_type", ext.lstrip("."))

        log.step(f"{path.name}  →  {len(docs)} document(s) loaded")
        return docs

    except Exception as exc:
        log.error(f"Failed to load {path.name}: {exc}")
        return []


def _collect_files(source: Path) -> List[Path]:
    """Return all loadable files under source (file or directory)."""
    if source.is_file():
        return [source]

    files = [
        p for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    files.sort()
    return files


# ── public API ────────────────────────────────────────────────────────────

def load_documents(source: str | Path) -> List[Document]:
    """
    Load all documents from *source* (file or directory).

    Returns
    -------
    list[Document]  — each item has .page_content and .metadata
    """
    source = Path(source)
    log.start(f"Stage A2 — Loading documents from: {source}")

    if not source.exists():
        log.error(f"Source path does not exist: {source}")
        raise FileNotFoundError(f"Source not found: {source}")

    files = _collect_files(source)
    if not files:
        log.warn("No supported files found — nothing to load.")
        return []

    log.step(f"Found {len(files)} file(s): {[f.name for f in files]}")

    all_docs: List[Document] = []
    for f in files:
        all_docs.extend(_load_file(f))

    total_chars = sum(len(d.page_content) for d in all_docs)
    log.done(
        f"A2 complete — {len(all_docs)} documents loaded, "
        f"~{total_chars:,} characters total"
    )
    return all_docs
