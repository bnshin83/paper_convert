# On-demand LLM reading over your Markdown corpus (SQLite FTS5)

This note documents the on-demand workflow we set up for your converted paper Markdown files.

The key idea is:

- **Index once** (chunk the Markdown, store in a local search index)
- **Retrieve on demand** (for each question, fetch only the top relevant chunks)
- **Send only those chunks to the LLM** (instead of asking the LLM to read thousands of papers)

This gives you a scalable “LLM reads papers” system without paying to ingest everything end-to-end.

---

## Why this workflow

LLMs are best used for:

- synthesizing and critiquing
- comparing methods
- extracting structured insights from a *small set* of relevant excerpts

They are not efficient at “reading everything” across thousands of documents.

By building a retrieval index, you turn your corpus into something like:

- a **search engine** for relevant evidence
- plus an **LLM reasoning layer** on top

---

## What was added to the repo

A new folder:

- `paper/rag/`

Two scripts:

- `paper/rag/md_fts_index.py`
  - builds/updates a local SQLite FTS5 index (`.sqlite3`) over Markdown text
  - chunks documents with simple heading-aware heuristics
  - incremental: only re-indexes files whose `mtime/size` changed

- `paper/rag/md_fts_query.py`
  - queries the index and returns top matching chunks
  - can output an **LLM-ready context block** with citations `[paper_id:chunk_id]`

The index DB default is:

- `paper/rag/md_fts.sqlite3`

---

## Important constraint in this environment

Your converted Markdown folder `markdown_output/` is **gitignored**, and IDE assistants may not be allowed to open it directly.

That’s OK: these scripts run locally and can read `markdown_output/` just fine.

---

## Data assumptions (how paper_id is inferred)

The indexer assumes the common layout produced by your pipeline:

```
.../markdown_output/<PAPER_ID>__<title>/<something>.md
```

The script infers:

- `paper_id = parent_folder_name.split("__", 1)[0]`

If your folder naming differs, we can adjust `infer_paper_id()`.

---

## Step 1 — Build the index

Example: index NeurIPS 2025 converted Markdown.

```bash
python rag/md_fts_index.py \
  --md-root /scratch/gautschi/shin283/paper/neurips_2025/pdf2md/markdown_output \
  --db /scratch/gautschi/shin283/paper/rag/md_fts.sqlite3 \
  --strip-code-blocks
```

Notes:

- `--md-root` is **repeatable**, so you can index multiple venues/years:

```bash
python rag/md_fts_index.py \
  --md-root /scratch/gautschi/shin283/paper/neurips_2025/pdf2md/markdown_output \
  --md-root /scratch/gautschi/shin283/paper/iclr_2025/pdf2md/markdown_output \
  --md-root /scratch/gautschi/shin283/paper/icml_2025/pdf2md/markdown_output
```

- `--strip-code-blocks` is optional but often improves search by removing huge fenced blocks.

---

## Chunking configuration

Defaults:

- `--max-chunk-chars 4000`
- `--overlap-chars 400`
- `--min-chunk-chars 200`

You can tune them, e.g. smaller chunks for more precise retrieval:

```bash
python rag/md_fts_index.py \
  --md-root .../markdown_output \
  --max-chunk-chars 2500 \
  --overlap-chars 250
```

---

## Step 2 — Query the index

Plain output (good for inspection):

```bash
python rag/md_fts_query.py \
  --db rag/md_fts.sqlite3 \
  --query "diffusion transformer training stability" \
  --top-k 8
```

LLM-ready context output:

```bash
python rag/md_fts_query.py \
  --db rag/md_fts.sqlite3 \
  --query "What are common limitations in diffusion transformer papers?" \
  --top-k 10 \
  --llm-context > context.md
```

The produced `context.md` includes:

- your user question
- excerpts from multiple papers
- a citation handle: `[paper_id:chunk_id]`

---

## Step 3 — Use the retrieved context with an LLM

Recommended pattern:

1. Run the query and generate `context.md`
2. Send to your LLM along with a focused instruction like:

Example instruction:

- “Using only the provided excerpts, synthesize the top 5 emerging themes, and for each theme cite 2–4 sources as `[paper_id:chunk_id]`. Then list open problems and limitations.”

This creates a reliable chain:

- retrieval constrains evidence
- LLM does synthesis

---

## Recommended “systems” you can build on top

Once retrieval works, you can add higher-level scripts without changing the index format:

- **Topic mapper**
  - query broad seeds ("diffusion", "agents", "alignment")
  - retrieve top chunks
  - LLM proposes a taxonomy

- **Per-paper structured extraction**
  - query by paper_id or by title terms
  - retrieve top chunks (intro/method/experiments)
  - extract: contributions, datasets, metrics, limitations

- **Rolling cache**
  - store LLM outputs per paper or per query to avoid recomputation

---

## Troubleshooting

### 1) “FTS5 is not supported”
If Python/SQLite on your system is built without FTS5, SQLite will error when creating the virtual table.

Check quickly:

```bash
python -c "import sqlite3; con=sqlite3.connect(':memory:'); con.execute('CREATE VIRTUAL TABLE t USING fts5(x)'); print('fts5 ok')"
```

If this fails, we can switch to:

- plain SQLite + `LIKE` (slow)
- Whoosh (pure python)
- or embeddings-based search (FAISS, etc.)

### 2) Index is too big / slow
- index only `abstract/introduction` sections (we can add a filter)
- increase `--min-chunk-chars`
- strip code blocks

### 3) Search quality is weak
- try more specific query terms
- increase `--top-k`
- adjust chunk size smaller

---

## Future Improvements (v2 considerations)

### 1) Embedding-based semantic search

FTS5 uses BM25 (keyword matching). For semantic queries like "papers about efficient training" that don't use exact keywords, FTS5 may miss relevant papers. Consider adding:

- FAISS + sentence-transformers for dense retrieval
- Hybrid scoring: combine BM25 + cosine similarity

### 2) Metadata filtering

The current schema only indexes text. Adding metadata (venue, primary_area, acceptance status) would enable queries like:

- "diffusion papers in vision track"
- "rejected papers about transformers"

This requires extending the SQLite schema to include a `papers` table with metadata columns.

### 3) Section-aware chunking

The heading-aware chunking is basic. Papers have predictable structure:

- Abstract → Introduction → Methods → Experiments → Conclusion

Section-aware chunking (tagging chunks with their section type) could yield better retrieval, e.g., "only search Methods sections".

### 4) Hybrid scoring with metadata

Combining BM25 with:

- recency (prefer newer papers)
- citation count or review ratings
- acceptance status

Would help surface more relevant papers for research analysis.

### 5) End-to-end automation

The workflow currently stops at "send context.md to LLM". A complete pipeline script could automate:

```
query → retrieve → synthesize → output
```

This would enable batch processing of multiple research questions.

---

## Files touched/created

Created:

- `paper/rag/md_fts_index.py`
- `paper/rag/md_fts_query.py`
- `paper/RAG_FTS_WORKFLOW.md` (this file)
