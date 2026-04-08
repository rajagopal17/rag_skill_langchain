"""
skills/chunking_langchain.py  — Stage B2: Chunking (LangChain RecursiveCharacterTextSplitter)
─────────────────────────────────────────────────────────────────────────────────────
Splits a list of LangChain Documents into smaller overlapping chunks
ready for embedding.

Configuration (defaults match the pipeline spec)
─────────────────────────────────────────────────
  CHUNK_SIZE    = 1000  characters
  CHUNK_OVERLAP = 200   characters

Strategy — RecursiveCharacterTextSplitter
──────────────────────────────────────────
  Tries to split on paragraph breaks → newlines → sentences → words
  before falling back to hard character splits.  This preserves
  semantic boundaries better than a simple fixed-size slice.

Public API
──────────
  split_documents(
      docs: list[Document],
      chunk_size: int = 1000,
      chunk_overlap: int = 200,
  ) -> list[Document]
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from skills.logger import StepLogger

log = StepLogger("chunking")

# ── defaults (match pipeline spec) ───────────────────────────────────────
DEFAULT_CHUNK_SIZE    = 1000
DEFAULT_CHUNK_OVERLAP = 200


# ── public API ────────────────────────────────────────────────────────────

def split_documents(
    docs: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split *docs* into overlapping text chunks.

    Parameters
    ----------
    docs          : output of a2_ingest.load_documents()
    chunk_size    : max characters per chunk  (default 1 000)
    chunk_overlap : overlap between consecutive chunks (default 200)

    Returns
    -------
    list[Document] — each item carries the parent document's metadata
                     plus chunk_index (0-based position within source doc)
    """
    log.start(
        f"Stage B2 — Chunking {len(docs)} document(s)  "
        f"[size={chunk_size}, overlap={chunk_overlap}]"
    )

    if not docs:
        log.warn("No documents to split — returning empty list.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        is_separator_regex=False,
        add_start_index=True,   # adds 'start_index' key to metadata
    )

    chunks: List[Document] = []
    global_chunk_idx = 0

    for doc_idx, doc in enumerate(docs):
        source = doc.metadata.get("file_name", f"doc_{doc_idx}")
        raw_chunks = splitter.split_documents([doc])

        # Attach chunk-level metadata — chunk_index is globally unique per source
        for i, chunk in enumerate(raw_chunks):
            chunk.metadata["chunk_index"] = global_chunk_idx
            chunk.metadata["chunk_index_in_doc"] = i
            chunk.metadata["total_chunks_in_doc"] = len(raw_chunks)
            chunk.metadata["doc_index"] = doc_idx
            global_chunk_idx += 1

        chunks.extend(raw_chunks)
        log.step(
            f"{source}  →  {len(raw_chunks)} chunk(s)  "
            f"(avg {int(sum(len(c.page_content) for c in raw_chunks)/max(len(raw_chunks),1))} chars)"
        )

    # Summary stats
    sizes = [len(c.page_content) for c in chunks]
    avg   = int(sum(sizes) / len(sizes)) if sizes else 0
    log.done(
        f"B2 complete — {len(chunks)} chunks total  "
        f"[min={min(sizes, default=0)}, avg={avg}, max={max(sizes, default=0)} chars]"
    )
    return chunks
