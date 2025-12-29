#!/usr/bin/env python3
"""Build a local SQLite FTS5 index over a Markdown corpus.

This is designed for on-demand LLM reading:
- index once (or incrementally)
- retrieve top matching chunks for a question
- feed only those chunks into an LLM

No third-party deps: uses only Python stdlib + SQLite (FTS5 must be enabled).
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple


HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")
CODE_FENCE_RE = re.compile(r"^\s*```")


@dataclass(frozen=True)
class FileStat:
    path: str
    mtime: float
    size: int


def iter_markdown_files(roots: list[Path]) -> Iterator[Path]:
    for root in roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            # Avoid common huge dirs
            dirnames[:] = [d for d in dirnames if d not in {"images", ".git", "__pycache__"}]
            for fn in filenames:
                if fn.lower().endswith(".md"):
                    yield Path(dirpath) / fn


def infer_paper_id(md_path: Path) -> str:
    # Expected pattern: markdown_output/<ID>__<title>/<file>.md
    parent = md_path.parent.name
    pid = parent.split("__", 1)[0].strip()
    return pid


def normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s


def iter_md_chunks(
    md_path: Path,
    *,
    max_chunk_chars: int,
    overlap_chars: int,
    min_chunk_chars: int,
    strip_code_blocks: bool,
) -> Iterable[Tuple[str, str]]:
    """Yield (heading, chunk_text)."""

    heading = ""
    in_code = False

    buf: list[str] = []
    buf_len = 0

    def flush(force: bool = False) -> Optional[Tuple[str, str]]:
        nonlocal buf, buf_len
        if not buf:
            return None
        text = "".join(buf)
        text = normalize_text(text).strip()
        if (not force) and len(text) < min_chunk_chars:
            return None
        # Keep chunks reasonably sized; trim extreme whitespace
        text = re.sub(r"[ \t]+", " ", text)
        return (heading, text)

    with md_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if strip_code_blocks and CODE_FENCE_RE.match(line):
                in_code = not in_code
                continue
            if strip_code_blocks and in_code:
                continue

            m = HEADING_RE.match(line)
            if m:
                # Update current heading. If buffer is already substantial, flush as a boundary.
                new_heading = m.group(2).strip()
                if buf_len >= max_chunk_chars // 2:
                    out = flush(force=False)
                    if out is not None:
                        yield out
                        # overlap from the previous chunk
                        prev = out[1]
                        ov = prev[-overlap_chars:] if overlap_chars > 0 else ""
                        buf = [ov] if ov else []
                        buf_len = len(ov)
                heading = new_heading
                # Keep the heading line itself as text (helps search)
                line = line.strip() + "\n"

            buf.append(line)
            buf_len += len(line)

            if buf_len >= max_chunk_chars:
                out = flush(force=True)
                if out is not None:
                    yield out
                    prev = out[1]
                    ov = prev[-overlap_chars:] if overlap_chars > 0 else ""
                    buf = [ov] if ov else []
                    buf_len = len(ov)

    out = flush(force=True)
    if out is not None:
        yield out


def connect_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA temp_store=MEMORY;")
    return con


def init_db(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS files (
          path TEXT PRIMARY KEY,
          mtime REAL NOT NULL,
          size INTEGER NOT NULL
        );
        """
    )
    con.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
          path UNINDEXED,
          paper_id UNINDEXED,
          chunk_id UNINDEXED,
          heading,
          text,
          tokenize='unicode61 remove_diacritics 2'
        );
        """
    )


def get_file_stat(p: Path) -> FileStat:
    st = p.stat()
    return FileStat(path=str(p), mtime=float(st.st_mtime), size=int(st.st_size))


def needs_reindex(con: sqlite3.Connection, st: FileStat) -> bool:
    row = con.execute("SELECT mtime, size FROM files WHERE path=?", (st.path,)).fetchone()
    if row is None:
        return True
    old_mtime, old_size = float(row[0]), int(row[1])
    return (old_mtime != st.mtime) or (old_size != st.size)


def mark_indexed(con: sqlite3.Connection, st: FileStat) -> None:
    con.execute(
        "INSERT INTO files(path, mtime, size) VALUES(?,?,?) "
        "ON CONFLICT(path) DO UPDATE SET mtime=excluded.mtime, size=excluded.size",
        (st.path, st.mtime, st.size),
    )


def delete_chunks_for_path(con: sqlite3.Connection, path: str) -> None:
    con.execute("DELETE FROM chunks WHERE path=?", (path,))


def index_one_file(
    con: sqlite3.Connection,
    md_path: Path,
    *,
    max_chunk_chars: int,
    overlap_chars: int,
    min_chunk_chars: int,
    strip_code_blocks: bool,
) -> int:
    st = get_file_stat(md_path)
    if not needs_reindex(con, st):
        return 0

    delete_chunks_for_path(con, st.path)

    paper_id = infer_paper_id(md_path)
    inserted = 0
    for i, (heading, text) in enumerate(
        iter_md_chunks(
            md_path,
            max_chunk_chars=max_chunk_chars,
            overlap_chars=overlap_chars,
            min_chunk_chars=min_chunk_chars,
            strip_code_blocks=strip_code_blocks,
        )
    ):
        con.execute(
            "INSERT INTO chunks(path, paper_id, chunk_id, heading, text) VALUES(?,?,?,?,?)",
            (st.path, paper_id, str(i), heading, text),
        )
        inserted += 1

    mark_indexed(con, st)
    return inserted


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build/update an SQLite FTS5 index over Markdown files.")
    ap.add_argument(
        "--md-root",
        action="append",
        required=True,
        help="Root directory containing markdown files (repeatable).",
    )
    ap.add_argument(
        "--db",
        default="rag/md_fts.sqlite3",
        help="SQLite DB path (default: rag/md_fts.sqlite3).",
    )
    ap.add_argument("--max-files", type=int, default=0, help="If >0, stop after indexing this many files.")
    ap.add_argument("--max-chunk-chars", type=int, default=4000)
    ap.add_argument("--overlap-chars", type=int, default=400)
    ap.add_argument("--min-chunk-chars", type=int, default=200)
    ap.add_argument("--strip-code-blocks", action="store_true")
    ap.add_argument("--commit-every", type=int, default=50)
    args = ap.parse_args(argv)

    roots = [Path(r) for r in args.md_root]
    db_path = Path(args.db)

    con = connect_db(db_path)
    try:
        init_db(con)
        con.commit()

        t0 = time.time()
        files_seen = 0
        files_indexed = 0
        chunks_inserted = 0

        for md_path in iter_markdown_files(roots):
            files_seen += 1
            try:
                n = index_one_file(
                    con,
                    md_path,
                    max_chunk_chars=args.max_chunk_chars,
                    overlap_chars=args.overlap_chars,
                    min_chunk_chars=args.min_chunk_chars,
                    strip_code_blocks=bool(args.strip_code_blocks),
                )
                if n:
                    files_indexed += 1
                    chunks_inserted += n

                if args.commit_every and (files_seen % args.commit_every == 0):
                    con.commit()
                    dt = time.time() - t0
                    print(
                        f"[progress] files_seen={files_seen} files_indexed={files_indexed} chunks_inserted={chunks_inserted} elapsed_s={dt:.1f}",
                        file=sys.stderr,
                    )

                if args.max_files and files_seen >= args.max_files:
                    break

            except Exception as e:
                print(f"[warn] failed to index {md_path}: {e}", file=sys.stderr)
                continue

        con.commit()
        dt = time.time() - t0
        print(
            f"[done] db={db_path} files_seen={files_seen} files_indexed={files_indexed} chunks_inserted={chunks_inserted} elapsed_s={dt:.1f}",
            file=sys.stderr,
        )
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
