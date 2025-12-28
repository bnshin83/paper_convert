#!/usr/bin/env python3
"""
Fetch NeurIPS 2025 papers from OpenReview (public API; no API key) and export to Markdown.

Default source:
  invitation = NeurIPS.cc/2025/Conference/-/Submission

By default, filters to "accepted" papers by requiring:
  content.venue.value starts with "NeurIPS 2025"
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


def fetch_forum_notes(
    forum_id: str,
    limit: int = 1000,
    sleep_s: float = 0.0,
    session: Optional[requests.Session] = None,
) -> Iterable[Dict[str, Any]]:
    """
    Generator over all notes in a forum thread (submission + reviews/comments/decisions/etc.)
    using offset pagination.
    """
    if session is None:
        session = requests.Session()

    offset = 0
    while True:
        params = {
            "forum": forum_id,
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


def fetch_forum_notes_head(
    forum_id: str,
    limit: int = 50,
    session: Optional[requests.Session] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch the first page of notes for a forum. In practice, decisions are often present here.
    """
    if session is None:
        session = requests.Session()
    params = {"forum": forum_id, "limit": limit, "offset": 0}
    resp = session.get(f"{OPENREVIEW_API2}/notes", params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    notes = data.get("notes", [])
    return notes if isinstance(notes, list) else []


def decision_and_review_summary(
    forum_id: str,
    session: requests.Session,
    *,
    head_limit: int = 80,
) -> Dict[str, Any]:
    """
    Extract lightweight label-like signals from a paper forum:
    - Decision (Accept/Reject/Poster/Spotlight/etc.) if present
    - Review rating/confidence (numeric prefix) aggregates if present

    This intentionally looks only at a head page for speed.
    """
    notes = fetch_forum_notes_head(forum_id, limit=head_limit, session=session)

    decision: str = ""
    decision_comment: str = ""

    rating_vals: List[int] = []
    confidence_vals: List[int] = []
    rating_raw: List[str] = []
    confidence_raw: List[str] = []

    for n in notes:
        invitations = n.get("invitations", []) or []
        content = _clean_content_dict(n.get("content", {}) or {})

        # Decision note
        if (not decision) and (
            any(str(inv).endswith("/-/Decision") for inv in invitations)
            or ("decision" in content and content.get("decision"))
        ):
            decision = str(content.get("decision", "") or "").strip()
            decision_comment = str(content.get("comment", "") or "").strip()

        # Official review note
        if any(str(inv).endswith("/-/Official_Review") for inv in invitations):
            r = content.get("rating")
            c = content.get("confidence")
            if isinstance(r, str) and r.strip():
                rating_raw.append(r.strip())
                li = _leading_int(r)
                if li is not None:
                    rating_vals.append(li)
            if isinstance(c, str) and c.strip():
                confidence_raw.append(c.strip())
                li = _leading_int(c)
                if li is not None:
                    confidence_vals.append(li)

    def _avg(xs: List[int]) -> Optional[float]:
        if not xs:
            return None
        return sum(xs) / len(xs)

    return {
        "decision": decision,
        "decision_comment": decision_comment,
        "review_rating_avg": _avg(rating_vals),
        "review_confidence_avg": _avg(confidence_vals),
        "review_rating_raw": rating_raw,
        "review_confidence_raw": confidence_raw,
        "review_count": len(rating_raw) if rating_raw else 0,
    }

def download_pdf(
    session: requests.Session,
    paper: Paper,
    pdf_dir: str,
    *,
    overwrite: bool = False,
) -> str:
    """
    Download paper PDF (if available) to pdf_dir and return local path.
    """
    os.makedirs(pdf_dir, exist_ok=True)
    if not paper.pdf_url:
        return ""

    base = f"{paper.id}__{_safe_filename(paper.title or paper.id, max_len=120)}.pdf"
    out_path = os.path.join(pdf_dir, base)

    if (not overwrite) and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    resp = session.get(paper.pdf_url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    return out_path


def pdf_to_text(pdf_path: str) -> str:
    # Imported lazily so the script still works for metadata-only runs.
    from pdfminer.high_level import extract_text  # type: ignore

    text = extract_text(pdf_path) or ""
    return _normalize_text(text)


def render_paper_fulltext_md(p: Paper, opt: RenderOptions, fulltext: str) -> str:
    # Reuse the standard header, then append extracted text.
    head = render_paper_md(p, opt).rstrip()
    lines = [head, "", "### Full text (PDF-extracted, best-effort)", "", fulltext, ""]
    return "\n".join(lines)


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


def is_accepted_neurips2025(p: Paper) -> bool:
    # Empirically, accepted papers have venue like "NeurIPS 2025 poster/spotlight/oral".
    return p.venue.startswith("NeurIPS 2025")


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
        # Keep it readable as plain text; avoid blockquotes for easier copy/paste.
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
    lines.append("# NeurIPS 2025 papers (OpenReview export)")
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
    # GitHub-style-ish anchor slug; good enough for navigation in most markdown renderers.
    t = title.strip().lower()
    t = re.sub(r"[^\w\s-]", "", t)
    t = re.sub(r"[\s_-]+", "-", t).strip("-")
    return t or "paper"


def write_split_files(out_dir: str, papers: List[Paper], opt: RenderOptions) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for p in papers:
        fname = _safe_filename(p.title or p.id) + ".md"
        path = os.path.join(out_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(render_paper_md(p, opt))


def _primary_area_counts(papers: List[Paper]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for p in papers:
        key = p.primary_area.strip() if p.primary_area else "(missing)"
        counts[key] = counts.get(key, 0) + 1
    return counts


def export_submissions_jsonl(
    path: str,
    papers: List[Paper],
    *,
    summaries_by_forum: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in papers:
            summ = (summaries_by_forum or {}).get(p.id, {})
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
                # Label-like signals (useful for accept vs reject analysis)
                "decision": summ.get("decision", ""),
                "decision_comment": summ.get("decision_comment", ""),
                "review_rating_avg": summ.get("review_rating_avg"),
                "review_confidence_avg": summ.get("review_confidence_avg"),
                "review_rating_raw": summ.get("review_rating_raw", []),
                "review_confidence_raw": summ.get("review_confidence_raw", []),
                "review_count": summ.get("review_count", 0),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def export_submissions_csv(
    path: str,
    papers: List[Paper],
    *,
    summaries_by_forum: Optional[Dict[str, Dict[str, Any]]] = None,
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
                "decision",
                "review_rating_avg",
                "review_confidence_avg",
                "review_count",
            ],
        )
        w.writeheader()
        for p in papers:
            summ = (summaries_by_forum or {}).get(p.id, {})
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
                    "decision": summ.get("decision", ""),
                    "review_rating_avg": summ.get("review_rating_avg"),
                    "review_confidence_avg": summ.get("review_confidence_avg"),
                    "review_count": summ.get("review_count", 0),
                }
            )


def export_forum_notes_jsonl(path: str, session: requests.Session, papers: List[Paper]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, p in enumerate(papers, start=1):
            for n in fetch_forum_notes(p.id, session=session):
                content = _clean_content_dict(n.get("content", {}) or {})
                rec = {
                    "id": n.get("id"),
                    "forum": n.get("forum"),
                    "replyto": n.get("replyto"),
                    "invitations": n.get("invitations", []),
                    "signatures": n.get("signatures", []),
                    "readers": n.get("readers", []),
                    "writers": n.get("writers", []),
                    "cdate": n.get("cdate"),
                    "mdate": n.get("mdate"),
                    "tcdate": n.get("tcdate"),
                    "tmdate": n.get("tmdate"),
                    "number": n.get("number"),
                    "content": content,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 200 == 0:
                print(f"Exported forum notes for {i}/{len(papers)} papers...", file=sys.stderr)

def parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Export NeurIPS 2025 papers from OpenReview to Markdown (no API key)."
    )
    ap.add_argument(
        "--invitation",
        default="NeurIPS.cc/2025/Conference/-/Submission",
        help="OpenReview invitation to list notes from.",
    )
    ap.add_argument(
        "--out",
        default="neurips2025.md",
        help="Output Markdown file path (ignored if --split-out-dir is set).",
    )
    ap.add_argument(
        "--split-out-dir",
        default="",
        help="If set, write one Markdown file per paper into this directory.",
    )
    ap.add_argument(
        "--export-dir",
        default="",
        help="If set, write machine-readable exports into this directory (JSONL/CSV).",
    )
    ap.add_argument(
        "--include-decision-summary",
        action="store_true",
        help="When using --export-dir, add decision/review summary fields to submissions exports (requires extra forum queries).",
    )
    ap.add_argument(
        "--include-forum-notes",
        action="store_true",
        help="When using --export-dir, also export forum notes (reviews/decisions/comments/etc.) to JSONL.",
    )
    ap.add_argument(
        "--download-pdfs",
        action="store_true",
        help="Download PDFs locally for papers that have a PDF link.",
    )
    ap.add_argument(
        "--pdf-dir",
        default="pdfs/",
        help="Where to store downloaded PDFs (used with --download-pdfs / --convert-pdfs-to-md).",
    )
    ap.add_argument(
        "--convert-pdfs-to-md",
        action="store_true",
        help="Download PDFs (if needed) and export one Markdown file per paper with PDF-extracted full text.",
    )
    ap.add_argument(
        "--fulltext-out-dir",
        default="fulltext_md/",
        help="Where to write full-text Markdown files (used with --convert-pdfs-to-md).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing downloaded PDFs / generated fulltext markdown files.",
    )
    ap.add_argument(
        "--no-subsets",
        action="store_true",
        help="Hide subset-like fields (primary area + keywords) and sort flat by title.",
    )
    ap.add_argument(
        "--hide-primary-area",
        action="store_true",
        help="Do not print the 'Primary area' line in Markdown.",
    )
    ap.add_argument(
        "--hide-keywords",
        action="store_true",
        help="Do not print the 'Keywords' line in Markdown.",
    )
    ap.add_argument(
        "--sort-by",
        choices=["venue_then_title", "title"],
        default="venue_then_title",
        help="Sort order for output. Default matches conference-style grouping.",
    )
    ap.add_argument(
        "--accepted-only",
        action="store_true",
        default=True,
        help="Keep only accepted papers (venue starts with 'NeurIPS 2025'). Default: true.",
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
    ap.add_argument(
        "--list-primary-areas",
        action="store_true",
        help="Print distinct primary areas (and counts) from the fetched set, then exit.",
    )
    return ap.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    sess = requests.Session()
    sess.headers.update({"User-Agent": "neurips2025-markdown-export/1.0"})

    papers: List[Paper] = []
    seen: set[str] = set()

    for note in fetch_notes(
        invitation=args.invitation, limit=args.limit, sleep_s=args.sleep, session=sess
    ):
        p = note_to_paper(note)
        if not p.id or p.id in seen:
            continue
        seen.add(p.id)
        if args.accepted_only and not is_accepted_neurips2025(p):
            continue
        papers.append(p)
        if args.max_papers and len(papers) >= args.max_papers:
            break

        if len(papers) % 250 == 0:
            print(f"Collected {len(papers)} papers...", file=sys.stderr)

    if args.no_subsets:
        args.hide_primary_area = True
        args.hide_keywords = True
        args.sort_by = "title"

    if args.sort_by == "title":
        papers.sort(key=lambda x: (x.title.lower(), x.id))
    else:
        papers.sort(key=lambda x: (x.venue, x.title.lower()))

    if args.list_primary_areas:
        counts = _primary_area_counts(papers)
        for area, cnt in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].lower())):
            print(f"{cnt}\t{area}")
        return 0

    opt = RenderOptions(
        include_venue=True,
        include_primary_area=not args.hide_primary_area,
        include_keywords=not args.hide_keywords,
    )

    if args.export_dir:
        os.makedirs(args.export_dir, exist_ok=True)
        summaries_by_forum: Optional[Dict[str, Dict[str, Any]]] = None
        if args.include_decision_summary:
            summaries_by_forum = {}
            for i, p in enumerate(papers, start=1):
                try:
                    summaries_by_forum[p.id] = decision_and_review_summary(p.id, sess)
                except Exception as e:
                    print(f"[warn] failed to fetch decision/review summary for {p.id}: {e}", file=sys.stderr)
                if i % 250 == 0:
                    print(f"Fetched decision/review summaries for {i}/{len(papers)} papers...", file=sys.stderr)

        export_submissions_jsonl(
            os.path.join(args.export_dir, "submissions.jsonl"),
            papers,
            summaries_by_forum=summaries_by_forum,
        )
        export_submissions_csv(
            os.path.join(args.export_dir, "submissions.csv"),
            papers,
            summaries_by_forum=summaries_by_forum,
        )
        if args.include_forum_notes:
            export_forum_notes_jsonl(
                os.path.join(args.export_dir, "forum_notes.jsonl"), sess, papers
            )
        print(f"Wrote exports to {args.export_dir}", file=sys.stderr)
        return 0

    if args.convert_pdfs_to_md:
        os.makedirs(args.fulltext_out_dir, exist_ok=True)
        for i, p in enumerate(papers, start=1):
            pdf_path = download_pdf(sess, p, args.pdf_dir, overwrite=args.overwrite)
            if not pdf_path:
                continue
            md_name = f"{p.id}__{_safe_filename(p.title or p.id, max_len=120)}.md"
            md_path = os.path.join(args.fulltext_out_dir, md_name)
            if (not args.overwrite) and os.path.exists(md_path) and os.path.getsize(md_path) > 0:
                continue

            try:
                fulltext = pdf_to_text(pdf_path)
            except Exception as e:
                print(f"[warn] PDF text extraction failed for {p.id}: {e}", file=sys.stderr)
                continue

            with open(md_path, "w", encoding="utf-8") as f:
                f.write(render_paper_fulltext_md(p, opt, fulltext))

            if i % 50 == 0:
                print(f"Converted {i}/{len(papers)} PDFs to Markdown...", file=sys.stderr)

        print(f"Wrote fulltext Markdown to {args.fulltext_out_dir}", file=sys.stderr)
        return 0

    if args.download_pdfs:
        for i, p in enumerate(papers, start=1):
            try:
                download_pdf(sess, p, args.pdf_dir, overwrite=args.overwrite)
            except Exception as e:
                print(f"[warn] PDF download failed for {p.id}: {e}", file=sys.stderr)
            if i % 200 == 0:
                print(f"Downloaded {i}/{len(papers)} PDFs...", file=sys.stderr)

    if args.split_out_dir:
        write_split_files(args.split_out_dir, papers, opt)
        print(f"Wrote {len(papers)} Markdown files to {args.split_out_dir}", file=sys.stderr)
        return 0

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


