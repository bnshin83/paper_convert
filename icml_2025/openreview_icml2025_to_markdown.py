#!/usr/bin/env python3
"""
Fetch ICML 2025 papers from OpenReview (public API; no API key) and export to Markdown.

Default source:
  invitation = ICML.cc/2025/Conference/-/Submission

By default, filters to "accepted" papers by requiring:
  content.venue.value starts with "ICML 2025"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import requests


OPENREVIEW_API2 = "https://api2.openreview.net"
OPENREVIEW_WEB = "https://openreview.net"


def _get_value(maybe_field: Any) -> Any:
    """
    OpenReview API2 note content is typically shaped like:
      {"title": {"value": "..."}, ...}
    This extracts the inner 'value' when present.
    """
    if isinstance(maybe_field, dict) and "value" in maybe_field:
        return maybe_field["value"]
    return maybe_field


def _clean_content_dict(content: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert OpenReview API2 content dict {field: {"value": ...}} to plain JSON-serializable values.
    """
    out: Dict[str, Any] = {}
    for k, v in (content or {}).items():
        vv = _get_value(v)
        # Normalize lists of dict-wrapped values
        if isinstance(vv, list):
            vv = [_get_value(x) for x in vv]
        out[k] = vv
    return out


def _safe_filename(name: str, max_len: int = 160) -> str:
    name = name.strip()
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"[^\w\-. ()\[\]]+", "_", name)
    name = name.strip(" ._")
    if not name:
        return "paper"
    if len(name) > max_len:
        return name[:max_len].rstrip(" ._")
    return name


def _normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _leading_int(s: str) -> Optional[int]:
    """
    Parse OpenReview-style ratings like "6: Weak Accept" or "4: Borderline".
    Returns the leading integer if present.
    """
    if not s:
        return None
    m = re.match(r"^\s*(\d+)\s*[:\-]?", str(s))
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


@dataclass(frozen=True)
class Paper:
    id: str
    title: str
    authors: List[str]
    abstract: str
    venue: str
    primary_area: str
    keywords: List[str]
    pdf_path: str
    bibtex: str

    @property
    def forum_url(self) -> str:
        return f"{OPENREVIEW_WEB}/forum?id={self.id}"

    @property
    def pdf_url(self) -> str:
        if not self.pdf_path:
            return ""
        if self.pdf_path.startswith("http://") or self.pdf_path.startswith("https://"):
            return self.pdf_path
        # API returns paths like "/pdf/<sha>.pdf"
        return f"{OPENREVIEW_WEB}{self.pdf_path}"


@dataclass(frozen=True)
class RenderOptions:
    include_venue: bool = True
    include_primary_area: bool = True
    include_keywords: bool = True


def fetch_notes(
    invitation: str,
    limit: int = 1000,
    sleep_s: float = 0.25,
    session: Optional[requests.Session] = None,
) -> Iterable[Dict[str, Any]]:
    """
    Generator over all notes for an invitation using offset pagination.
    Continues until a page returns zero notes.
    """
    if session is None:
        session = requests.Session()

    offset = 0
    while True:
        params = {
            "invitation": invitation,
            "limit": limit,
            "offset": offset,
        }
        resp = session.get(f"{OPENREVIEW_API2}/notes", params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        notes = data.get("notes", [])
        if not notes:
            break

        for n in notes:
            yield n

        offset += len(notes)
        if sleep_s > 0:
            time.sleep(sleep_s)


def download_pdf(
    session: requests.Session,
    paper: Paper,
    pdf_dir: str,
    *,
    overwrite: bool = False,
    max_retries: int = 3,
) -> str:
    """
    Download paper PDF (if available) to pdf_dir and return local path.
    Includes retry logic with exponential backoff for rate limiting.
    """
    os.makedirs(pdf_dir, exist_ok=True)
    if not paper.pdf_url:
        return ""

    base = f"{paper.id}__{_safe_filename(paper.title or paper.id, max_len=120)}.pdf"
    out_path = os.path.join(pdf_dir, base)

    if (not overwrite) and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    for attempt in range(max_retries):
        try:
            resp = session.get(paper.pdf_url, stream=True, timeout=120)
            if resp.status_code == 429:
                # Rate limited - wait and retry
                wait_time = 2 ** (attempt + 2)  # 4, 8, 16 seconds
                time.sleep(wait_time)
                continue
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            return out_path
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise

    raise requests.exceptions.RequestException(f"Failed after {max_retries} retries")


def note_to_paper(note: Dict[str, Any]) -> Paper:
    content = note.get("content", {}) or {}

    title = str(_get_value(content.get("title", "")) or "").strip()
    authors = _get_value(content.get("authors", [])) or []
    if not isinstance(authors, list):
        authors = [str(authors)]
    authors = [str(a).strip() for a in authors if str(a).strip()]

    abstract = str(_get_value(content.get("abstract", "")) or "").strip()
    venue = str(_get_value(content.get("venue", "")) or "").strip()
    primary_area = str(_get_value(content.get("primary_area", "")) or "").strip()

    keywords = _get_value(content.get("keywords", [])) or []
    if not isinstance(keywords, list):
        keywords = [str(keywords)]
    keywords = [str(k).strip() for k in keywords if str(k).strip()]

    pdf_path = str(_get_value(content.get("pdf", "")) or "").strip()
    bibtex = str(_get_value(content.get("_bibtex", "")) or "").strip()

    return Paper(
        id=str(note.get("id", "")).strip(),
        title=title,
        authors=authors,
        abstract=abstract,
        venue=venue,
        primary_area=primary_area,
        keywords=keywords,
        pdf_path=pdf_path,
        bibtex=bibtex,
    )


def is_accepted_icml2025(p: Paper) -> bool:
    # Accepted papers have venue like "ICML 2025 poster/spotlightposter/oral".
    return p.venue.startswith("ICML 2025")


def render_paper_md(p: Paper, opt: RenderOptions) -> str:
    lines: List[str] = []
    lines.append(f"## {p.title or '(untitled)'}")
    lines.append("")
    if p.authors:
        lines.append(f"- **Authors**: {', '.join(p.authors)}")
    if opt.include_venue and p.venue:
        lines.append(f"- **Venue**: {p.venue}")
    if opt.include_primary_area and p.primary_area:
        lines.append(f"- **Primary area**: {p.primary_area}")
    lines.append(f"- **OpenReview**: `{p.forum_url}`")
    if p.pdf_url:
        lines.append(f"- **PDF**: `{p.pdf_url}`")
    if opt.include_keywords and p.keywords:
        lines.append(f"- **Keywords**: {', '.join(p.keywords)}")
    lines.append("")
    if p.abstract:
        lines.append("### Abstract")
        lines.append("")
        lines.append(p.abstract)
        lines.append("")
    if p.bibtex:
        lines.append("### BibTeX")
        lines.append("")
        lines.append("```")
        lines.append(p.bibtex)
        lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_index_md(papers: List[Paper]) -> str:
    lines: List[str] = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines.append("# ICML 2025 papers (OpenReview export)")
    lines.append("")
    lines.append(f"- **Generated (UTC)**: {ts}")
    lines.append(f"- **Count**: {len(papers)}")
    lines.append("")
    lines.append("## Index")
    lines.append("")
    for p in papers:
        title = p.title or p.id
        lines.append(f"- [{title}](#{slugify_heading(title)})")
    lines.append("")
    return "\n".join(lines)


def slugify_heading(title: str) -> str:
    t = title.strip().lower()
    t = re.sub(r"[^\w\s-]", "", t)
    t = re.sub(r"[\s_-]+", "-", t).strip("-")
    return t or "paper"


def export_submissions_jsonl(
    path: str,
    papers: List[Paper],
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in papers:
            rec = {
                "id": p.id,
                "title": p.title,
                "authors": p.authors,
                "venue": p.venue,
                "primary_area": p.primary_area,
                "keywords": p.keywords,
                "abstract": p.abstract,
                "forum_url": p.forum_url,
                "pdf_url": p.pdf_url,
                "bibtex": p.bibtex,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def export_submissions_csv(
    path: str,
    papers: List[Paper],
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "title",
                "authors",
                "venue",
                "primary_area",
                "keywords",
                "abstract",
                "forum_url",
                "pdf_url",
            ],
        )
        w.writeheader()
        for p in papers:
            w.writerow(
                {
                    "id": p.id,
                    "title": p.title,
                    "authors": "; ".join(p.authors),
                    "venue": p.venue,
                    "primary_area": p.primary_area,
                    "keywords": "; ".join(p.keywords),
                    "abstract": p.abstract,
                    "forum_url": p.forum_url,
                    "pdf_url": p.pdf_url,
                }
            )


def parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Export ICML 2025 papers from OpenReview to Markdown (no API key)."
    )
    ap.add_argument(
        "--invitation",
        default="ICML.cc/2025/Conference/-/Submission",
        help="OpenReview invitation to list notes from.",
    )
    ap.add_argument(
        "--out",
        default="icml2025.md",
        help="Output Markdown file path.",
    )
    ap.add_argument(
        "--export-dir",
        default="",
        help="If set, write machine-readable exports into this directory (JSONL/CSV).",
    )
    ap.add_argument(
        "--download-pdfs",
        action="store_true",
        help="Download PDFs locally for papers that have a PDF link.",
    )
    ap.add_argument(
        "--pdf-dir",
        default="pdfs/",
        help="Where to store downloaded PDFs.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing downloaded PDFs.",
    )
    ap.add_argument(
        "--accepted-only",
        action="store_true",
        default=True,
        help="Keep only accepted papers (venue starts with 'ICML 2025'). Default: true.",
    )
    ap.add_argument(
        "--all",
        dest="accepted_only",
        action="store_false",
        help="Do not filter by acceptance; export all submissions.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Page size for OpenReview API pagination.",
    )
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.25,
        help="Sleep seconds between pages to be polite to the API.",
    )
    ap.add_argument(
        "--max-papers",
        type=int,
        default=0,
        help="If >0, stop after this many papers (useful for quick tests).",
    )
    return ap.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    sess = requests.Session()
    sess.headers.update({"User-Agent": "icml2025-markdown-export/1.0"})

    papers: List[Paper] = []
    seen: set[str] = set()

    print(f"Fetching papers from {args.invitation}...", file=sys.stderr)

    for note in fetch_notes(
        invitation=args.invitation, limit=args.limit, sleep_s=args.sleep, session=sess
    ):
        p = note_to_paper(note)
        if not p.id or p.id in seen:
            continue
        seen.add(p.id)
        if args.accepted_only and not is_accepted_icml2025(p):
            continue
        papers.append(p)
        if args.max_papers and len(papers) >= args.max_papers:
            break

        if len(papers) % 250 == 0:
            print(f"Collected {len(papers)} papers...", file=sys.stderr)

    print(f"Total accepted papers: {len(papers)}", file=sys.stderr)

    # Sort by venue then title
    papers.sort(key=lambda x: (x.venue, x.title.lower()))

    opt = RenderOptions(
        include_venue=True,
        include_primary_area=True,
        include_keywords=True,
    )

    if args.export_dir:
        os.makedirs(args.export_dir, exist_ok=True)
        export_submissions_jsonl(
            os.path.join(args.export_dir, "submissions.jsonl"),
            papers,
        )
        export_submissions_csv(
            os.path.join(args.export_dir, "submissions.csv"),
            papers,
        )
        print(f"Wrote exports to {args.export_dir}", file=sys.stderr)

    if args.download_pdfs:
        print(f"Downloading {len(papers)} PDFs to {args.pdf_dir}...", file=sys.stderr)
        success = 0
        skipped = 0
        errors = 0
        for i, p in enumerate(papers, start=1):
            # Check if already exists before trying
            base = f"{p.id}__{_safe_filename(p.title or p.id, max_len=120)}.pdf"
            out_path = os.path.join(args.pdf_dir, base)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                skipped += 1
                success += 1
                if i % 500 == 0:
                    print(f"Progress: {i}/{len(papers)} (success: {success}, skipped: {skipped}, errors: {errors})...", file=sys.stderr)
                continue

            try:
                path = download_pdf(sess, p, args.pdf_dir, overwrite=args.overwrite)
                if path:
                    success += 1
                # Rate limit: sleep between downloads to avoid 429
                time.sleep(0.5)
            except Exception as e:
                errors += 1
                print(f"[warn] PDF download failed for {p.id}: {e}", file=sys.stderr)
            if i % 100 == 0:
                print(f"Progress: {i}/{len(papers)} (success: {success}, skipped: {skipped}, errors: {errors})...", file=sys.stderr)
        print(f"PDF download complete: {success} success, {skipped} skipped, {errors} errors", file=sys.stderr)

    # Write markdown summary
    out_path = args.out
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(render_index_md(papers))
        for p in papers:
            f.write("\n")
            f.write(render_paper_md(p, opt))

    print(f"Wrote {len(papers)} papers to {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
