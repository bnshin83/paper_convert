#!/usr/bin/env python3
"""
Fetch decision + lightweight review summary for OpenReview forums with retry/backoff.

Why:
  openreview_neurips2025_to_markdown.py --include-decision-summary can hit HTTP 429 and leave
  many records missing decision/rating/confidence. This script fills them in incrementally.

Input:
  exports_all_submissions/submissions.jsonl  (needs 'id' per record)

Output:
  analysis/summaries.jsonl  (one record per id, can be merged later)
  analysis/summaries_failed.txt

Safe usage:
  - Run with --sleep 1.0 or higher to avoid 429.
  - Use --resume to skip already-fetched ids.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

OPENREVIEW_API2 = "https://api2.openreview.net"


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _get_value(maybe_field: Any) -> Any:
    if isinstance(maybe_field, dict) and "value" in maybe_field:
        return maybe_field["value"]
    return maybe_field


def _clean_content_dict(content: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (content or {}).items():
        vv = _get_value(v)
        if isinstance(vv, list):
            vv = [_get_value(x) for x in vv]
        out[k] = vv
    return out


def _leading_int(s: Any) -> Optional[int]:
    if not s:
        return None
    s = str(s)
    i = 0
    while i < len(s) and s[i].isspace():
        i += 1
    j = i
    while j < len(s) and s[j].isdigit():
        j += 1
    if j == i:
        return None
    try:
        return int(s[i:j])
    except Exception:
        return None


def fetch_forum_notes_head(
    forum_id: str, session: requests.Session, *, head_limit: int
) -> List[Dict[str, Any]]:
    params = {"forum": forum_id, "limit": head_limit, "offset": 0}
    resp = session.get(f"{OPENREVIEW_API2}/notes", params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    notes = data.get("notes", [])
    return notes if isinstance(notes, list) else []


def decision_and_review_summary_from_head_notes(notes: List[Dict[str, Any]]) -> Dict[str, Any]:
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

    out: Dict[str, Any] = {
        "decision": decision,
        "decision_comment": decision_comment,
        "review_count": len(rating_raw),
        "review_rating_raw": rating_raw,
        "review_confidence_raw": confidence_raw,
        "review_rating_avg": (sum(rating_vals) / len(rating_vals)) if rating_vals else None,
        "review_confidence_avg": (sum(confidence_vals) / len(confidence_vals)) if confidence_vals else None,
    }
    return out


def fetch_summary_with_backoff(
    forum_id: str,
    session: requests.Session,
    *,
    head_limit: int,
    max_retries: int,
    base_sleep: float,
) -> Dict[str, Any]:
    for attempt in range(max_retries + 1):
        try:
            notes = fetch_forum_notes_head(forum_id, session, head_limit=head_limit)
            return decision_and_review_summary_from_head_notes(notes)
        except requests.HTTPError as e:
            resp = getattr(e, "response", None)
            status = resp.status_code if resp is not None else None
            if status == 429 and attempt < max_retries:
                # Prefer server-provided retry-after if present
                retry_after = None
                if resp is not None:
                    ra = resp.headers.get("Retry-After")
                    if ra:
                        try:
                            retry_after = float(ra)
                        except Exception:
                            retry_after = None
                sleep_s = retry_after if retry_after is not None else (base_sleep * (2**attempt))
                time.sleep(sleep_s)
                continue
            raise


def load_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                pid = str(rec.get("id", "")).strip()
                if pid:
                    done.add(pid)
            except Exception:
                continue
    return done


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submissions-jsonl", default="exports_all_submissions/submissions.jsonl")
    ap.add_argument("--out-jsonl", default="analysis/summaries.jsonl")
    ap.add_argument("--out-failed", default="analysis/summaries_failed.txt")
    ap.add_argument("--head-limit", type=int, default=80)
    ap.add_argument("--sleep", type=float, default=1.0, help="Base sleep between successful requests")
    ap.add_argument("--max-retries", type=int, default=8, help="Retries on HTTP 429 with exponential backoff")
    ap.add_argument("--max-papers", type=int, default=0, help="If >0, stop after this many ids")
    ap.add_argument("--resume", action="store_true", help="Skip ids already present in --out-jsonl")
    args = ap.parse_args(argv)

    submissions = Path(args.submissions_jsonl)
    out_jsonl = Path(args.out_jsonl)
    out_failed = Path(args.out_failed)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    done = load_done_ids(out_jsonl) if args.resume else set()

    sess = requests.Session()
    sess.headers.update({"User-Agent": "neurips2025-summary-fetch/1.0"})

    failed: List[str] = []
    wrote = 0

    with out_jsonl.open("a", encoding="utf-8") as f_out:
        for rec in iter_jsonl(submissions):
            pid = str(rec.get("id", "")).strip()
            if not pid:
                continue
            if pid in done:
                continue

            try:
                summ = fetch_summary_with_backoff(
                    pid,
                    sess,
                    head_limit=args.head_limit,
                    max_retries=args.max_retries,
                    base_sleep=args.sleep,
                )
                f_out.write(json.dumps({"id": pid, **summ}, ensure_ascii=False) + "\n")
                wrote += 1
            except Exception:
                failed.append(pid)

            # polite pacing
            if args.sleep > 0:
                time.sleep(args.sleep)

            if args.max_papers and wrote >= args.max_papers:
                break

    if failed:
        out_failed.write_text("\n".join(failed) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


