# RAG Pipeline — LangChain + pgvector + OpenAI

A modular Retrieval-Augmented Generation (RAG) pipeline that ingests documents,
embeds them with OpenAI, stores vectors in PostgreSQL (pgvector), and answers
natural language questions via a FastAPI REST API.

---

## Architecture

```
Document (PDF/DOCX/XLSX/CSV/PPTX/TXT)
    │
    ▼
[A] Data Ingestion        skills/data_ingestion_langchain.py
    │  LangChain loaders → list[Document]
    ▼
[B] Chunking              skills/chunking_langchain.py
    │  RecursiveCharacterTextSplitter (1000 chars / 200 overlap)
    ▼
[C] Embedding             skills/embedding_openai.py
    │  OpenAI text-embedding-3-small → 1536-dim vectors
    ▼
[D] Metadata Enrichment   skills/metadata_langchain.py
    │  Timestamps, word counts, pipeline_run_id
    ▼
[E] Vector Store          skills/vector_store_pgvector.py
    │  psycopg2 + pgvector → PostgreSQL table (with UUID)
    ▼
[F] Query Engine          skills/rag_query_engine.py
       Cosine similarity retrieval + OpenAI gpt-4o-mini answer
```

---

## Project Structure

```
rag_skill_langchain/
├── main.py                           ← FastAPI app (ingest / query / health)
├── requirements.txt                  ← Python dependencies
├── .env                              ← Environment variables (not committed)
└── skills/
    ├── __init__.py
    ├── logger.py                     ← Shared StepLogger (console + file)
    ├── data_ingestion_langchain.py   ← Stage A: load documents
    ├── chunking_langchain.py         ← Stage B: split into chunks
    ├── embedding_openai.py           ← Stage C: OpenAI embeddings
    ├── metadata_langchain.py         ← Stage D: enrich metadata
    ├── vector_store_pgvector.py      ← Stage E: upload to pgvector
    └── rag_query_engine.py           ← Stage F: retrieve + answer
```

---

## Setup

### 1. Prerequisites

- Python 3.11+
- PostgreSQL with `pgvector` and `uuid-ossp` extensions
- OpenAI API key

### 2. Install dependencies

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 3. Configure environment

Edit `.env`:

```env
OPENAI_API_KEY=sk-...

PG_HOST=localhost
PG_PORT=5434
PG_USER=postgres
PG_PASSWORD=your_password
PG_DB=your_database
PG_TABLE=your-table-name

FILE_PATH=D:\your_document.pdf
```

### 4. Enable PostgreSQL extensions

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
```

---

## Ingestion

Run the full pipeline to load, chunk, embed, and store a document:

```bash
PYTHONIOENCODING=utf-8 venv/Scripts/python -c "
from dotenv import load_dotenv
load_dotenv()
import os, logging
logging.disable(logging.CRITICAL)

from skills.data_ingestion_langchain import load_documents
from skills.chunking_langchain       import split_documents
from skills.embedding_openai         import embed_chunks
from skills.metadata_langchain       import enrich_metadata
from skills.vector_store_pgvector    import upload_chunks

file_path = os.getenv('FILE_PATH','').strip('\"').strip()
table     = os.getenv('PG_TABLE', 'rag_chunks')

docs     = load_documents(file_path)
chunks   = split_documents(docs)
embedded = embed_chunks(chunks)
enrich_metadata([ec.document for ec in embedded])
rows = upload_chunks(embedded, table=table, mode='insert')
print(f'Uploaded {rows} rows — Done!')
"
```

**Upload modes:**

| Mode | Behaviour |
|------|-----------|
| `insert` | Always append new rows |
| `upsert` | Update existing `(source, chunk_index)` or insert |
| `replace` | Delete all rows for source, then insert fresh |

---

## FastAPI Server

### Start

```bash
PYTHONIOENCODING=utf-8 venv/Scripts/python main.py
```

Server runs at `http://localhost:8000`

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/query?question=...` | Query via browser/curl |
| `POST` | `/query` | Query with full JSON options |
| `POST` | `/ingest` | Upload files + run full pipeline |
| `POST` | `/update` | Add/update documents |
| `GET` | `/health` | DB + OpenAI connectivity check |
| `GET` | `/docs` | Swagger UI |

---

## Querying

### Browser / GET

```
http://localhost:8000/query?question=What is asset depreciation?&top_k=5
```

### curl POST

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is asset depreciation?",
    "table": "sap-joule-knowledge-base",
    "top_k": 5
  }'
```

### Python script

```python
from dotenv import load_dotenv
load_dotenv()
import logging
logging.disable(logging.CRITICAL)

from skills.rag_query_engine import answer_question

result = answer_question(
    question='What is asset depreciation?',
    table='sap-joule-knowledge-base',
    top_k=5,
)

print('Answer:', result.answer)
print(f'Model: {result.model} | Tokens: {result.prompt_tokens}+{result.completion_tokens} | Time: {result.elapsed_s:.1f}s')

for c in result.retrieved:
    print(f'  [{c.score:.3f}] uuid={c.metadata.get("uuid")}  chunk={c.chunk_index}')
    print(f'         {c.content[:120]}...')
```

### Direct SQL (psycopg2)

```python
import psycopg2, psycopg2.extras

conn = psycopg2.connect(
    host='localhost', port=5434,
    dbname='business_partner',
    user='postgres', password='your_password'
)
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

# Keyword search
cur.execute('''
    SELECT uuid, chunk_index, file_name, content
    FROM "sap-joule-knowledge-base"
    WHERE content ILIKE %s
    ORDER BY chunk_index
    LIMIT 10;
''', ('%depreciation%',))

for row in cur.fetchall():
    print(row['uuid'], row['chunk_index'], row['content'][:100])

cur.close()
conn.close()
```

---

## Table Schema

```sql
CREATE TABLE "sap-joule-knowledge-base" (
    id              SERIAL PRIMARY KEY,
    uuid            UUID         NOT NULL DEFAULT uuid_generate_v4(),
    source          TEXT,
    file_name       TEXT,
    file_type       TEXT,
    chunk_index     INT,          -- globally unique per source
    doc_index       INT,
    char_count      INT,
    word_count      INT,
    char_offset     INT,
    doc_category    TEXT,
    fiscal_period   TEXT,
    entity          TEXT,
    ingested_at     TIMESTAMPTZ,
    pipeline_run_id TEXT,
    content         TEXT NOT NULL,
    embedding       VECTOR(1536) NOT NULL
);
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | **Required.** OpenAI API key |
| `OPENAI_EMBED_MODEL` | `text-embedding-3-small` | Embedding model |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Chat model for answers |
| `PG_HOST` | `localhost` | PostgreSQL host |
| `PG_PORT` | `5432` | PostgreSQL port |
| `PG_DB` | `ragdb` | Database name |
| `PG_USER` | `postgres` | Database user |
| `PG_PASSWORD` | — | **Required.** Database password |
| `PG_TABLE` | `rag_chunks` | Target vector table |
| `PG_BATCH_SIZE` | `200` | Rows per SQL batch insert |
| `RAG_TOP_K` | `5` | Chunks retrieved per query |
| `RAG_MAX_CONTEXT` | `4000` | Max context chars sent to LLM |
| `FILE_PATH` | — | Path to document for ingestion |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `PORT` | `8000` | FastAPI server port |
