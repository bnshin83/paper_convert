 #!/usr/bin/env python3
"""Query a local SQLite FTS5 index over a Markdown corpus.

Typical usage:
  python rag/md_fts_query.py --db rag/md_fts.sqlite3 --query "diffusion transformer" --top-k 8

You can also emit an LLM-ready context block:
  python rag/md_fts_query.py --db rag/md_fts.sqlite3 --query "offline RL evaluation" --top-k 10 --llm-context > ctx.txt
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class Hit:
    score: float
    paper_id: str
    path: str
    chunk_id: str
    heading: str
    text: str


def connect_db(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    return con


def fts_query_string(q: str) -> str:
    """Convert free text into a conservative FTS5 query.

    We mostly want AND semantics, but keep it simple:
    - tokenize words
    - join with AND
    """
    toks = re.findall(r"[A-Za-z0-9_\-]+", q)
    toks = [t for t in toks if t]
    if not toks:
        return ""
    return " AND ".join(toks)


def search(con: sqlite3.Connection, query: str, *, top_k: int) -> List[Hit]:
    q = fts_query_string(query)
    if not q:
        return []

    # bm25() is supported by SQLite FTS5.
    rows = con.execute(
        """
        SELECT
          bm25(chunks, 0.0, 0.0, 0.0, 5.0, 1.0) AS score,
          paper_id,
          path,
          chunk_id,
          coalesce(heading, '') AS heading,
          text
        FROM chunks
        WHERE chunks MATCH ?
        ORDER BY score
        LIMIT ?
        """,
        (q, int(top_k)),
    ).fetchall()

    hits: List[Hit] = []
    for r in rows:
        hits.append(
            Hit(
                score=float(r["score"]),
                paper_id=str(r["paper_id"]),
                path=str(r["path"]),
                chunk_id=str(r["chunk_id"]),
                heading=str(r["heading"]),
                text=str(r["text"]),
            )
        )
    return hits


def format_hits_plain(hits: List[Hit]) -> str:
    lines: List[str] = []
    for i, h in enumerate(hits, start=1):
        lines.append(f"[{i}] score={h.score:.3f} paper_id={h.paper_id} chunk={h.chunk_id}")
        lines.append(f"path: {h.path}")
        if h.heading:
            lines.append(f"heading: {h.heading}")
        lines.append(h.text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_hits_llm_context(hits: List[Hit], *, user_query: str) -> str:
    lines: List[str] = []
    lines.append("# Retrieved context")
    lines.append("")
    lines.append("Use the following excerpts to answer the user question. Cite sources as [paper_id:chunk_id].")
    lines.append("")
    lines.append(f"## User question")
    lines.append("")
    lines.append(user_query.strip())
    lines.append("")

    for h in hits:
        lines.append(f"## Source [{h.paper_id}:{h.chunk_id}]")
        lines.append(f"Path: `{h.path}`")
        if h.heading:
            lines.append(f"Heading: {h.heading}")
        lines.append("")
        lines.append(h.text)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Query an SQLite FTS5 index over Markdown chunks.")
    ap.add_argument("--db", default="rag/md_fts.sqlite3", help="Path to index db.")
    ap.add_argument("--query", required=True, help="Search query.")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--llm-context", action="store_true", help="Output LLM-ready markdown context.")
    args = ap.parse_args(argv)

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Missing DB: {db_path}. Build it first with rag/md_fts_index.py", file=sys.stderr)
        return 2

    con = connect_db(db_path)
    try:
        hits = search(con, args.query, top_k=args.top_k)
    finally:
        con.close()

    if args.llm_context:
        sys.stdout.write(format_hits_llm_context(hits, user_query=args.query))
    else:
        sys.stdout.write(format_hits_plain(hits))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
