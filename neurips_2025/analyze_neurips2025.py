#!/usr/bin/env python3
"""
NeurIPS 2025 corpus analysis (trends + acceptance/prediction baselines).

Primary input:
  exports_all_submissions/submissions.jsonl

Optional augmentation:
  pdf2md/markdown_output/<openreview_id>__*/<paper>.md

Outputs (created under ./analysis/):
  - paper_features.csv
  - primary_area_stats.csv
  - keyword_stats_top.csv
  - rating_bins.csv
  - report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)")
MD_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
MD_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+")
MD_MATH_BLOCK_RE = re.compile(r"^\s*\$\$\s*$")
CODE_LINK_RE = re.compile(r"(github\.com/|gitlab\.com/|bitbucket\.org/)", re.IGNORECASE)


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, str) and not x.strip():
            return None
        return float(x)
    except Exception:
        return None


def _safe_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        if isinstance(x, str) and not x.strip():
            return None
        return int(x)
    except Exception:
        return None


def mean_or_none(xs: List[float]) -> Optional[float]:
    xs2 = [x for x in xs if x is not None and not math.isnan(x)]  # type: ignore[arg-type]
    return statistics.mean(xs2) if xs2 else None


def median_or_none(xs: List[float]) -> Optional[float]:
    xs2 = [x for x in xs if x is not None and not math.isnan(x)]  # type: ignore[arg-type]
    return statistics.median(xs2) if xs2 else None


def slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_-]+", "-", s).strip("-")
    return s or "paper"


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


@dataclass(frozen=True)
class MdFeatures:
    md_path: str
    md_bytes: int
    line_count: int
    heading_count: int
    figure_link_count: int
    table_row_count: int
    math_block_count: int
    has_broken_table_token: int


def extract_md_features(md_path: Path) -> MdFeatures:
    line_count = 0
    heading_count = 0
    figure_link_count = 0
    table_row_count = 0
    math_block_count = 0
    has_broken_table_token = 0

    in_math = False
    with md_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_count += 1
            if MD_HEADING_RE.match(line):
                heading_count += 1
            if MD_IMAGE_RE.search(line):
                figure_link_count += 1
            if MD_TABLE_ROW_RE.match(line):
                table_row_count += 1
            if "rable " in line.lower():  # catches common OCR/table corruption like "rable 1"
                has_broken_table_token = 1

            if MD_MATH_BLOCK_RE.match(line):
                # Toggle on/off; count each fence as half-block and round down later.
                in_math = not in_math
                math_block_count += 1

    # Count pairs of $$ fences as blocks
    math_block_count = math_block_count // 2

    return MdFeatures(
        md_path=str(md_path),
        md_bytes=md_path.stat().st_size,
        line_count=line_count,
        heading_count=heading_count,
        figure_link_count=figure_link_count,
        table_row_count=table_row_count,
        math_block_count=math_block_count,
        has_broken_table_token=has_broken_table_token,
    )


def load_md_feature_index(md_root: Path) -> Dict[str, MdFeatures]:
    """
    Index markdown features by OpenReview forum id (paper id).
    Expected directory pattern:
      pdf2md/markdown_output/<ID>__*/<something>.md
    """
    out: Dict[str, MdFeatures] = {}
    if not md_root.exists():
        return out
    for d in md_root.iterdir():
        if not d.is_dir():
            continue
        # Extract id from prefix before "__"
        paper_id = d.name.split("__", 1)[0].strip()
        if not paper_id:
            continue
        md_files = list(d.glob("*.md"))
        if not md_files:
            continue
        # Use the largest md as the "main" one
        md_files.sort(key=lambda p: p.stat().st_size, reverse=True)
        out[paper_id] = extract_md_features(md_files[0])
    return out


def is_accepted(rec: Dict[str, Any]) -> Optional[bool]:
    """
    Robust-ish label:
      - If 'decision' mentions accept/reject, use it.
      - Else if 'venue' starts with 'NeurIPS 2025', treat as accepted.
      - Else unknown (None).
    """
    decision = (rec.get("decision") or "").strip().lower()
    venue = (rec.get("venue") or "").strip()
    if decision:
        if "accept" in decision:
            return True
        if "reject" in decision:
            return False
    if venue.startswith("NeurIPS 2025"):
        return True
    # Many OpenReview exports use this for papers without final venue assignment
    # (often rejected/undecided/withdrawn depending on conference policy).
    if venue.lower().startswith("submitted to neurips 2025"):
        return False
    if venue == "":
        return None
    return None


def decision_bucket(rec: Dict[str, Any]) -> str:
    d = (rec.get("decision") or "").strip().lower()
    v = (rec.get("venue") or "").strip().lower()
    s = d or v
    if "oral" in s:
        return "oral"
    if "spotlight" in s:
        return "spotlight"
    if "poster" in s:
        return "poster"
    if "submitted to neurips 2025" in s:
        return "submitted"
    if "reject" in s:
        return "reject"
    if s:
        return "other"
    return "unknown"


def has_code_link(rec: Dict[str, Any]) -> int:
    text = " ".join(
        [
            str(rec.get("title") or ""),
            str(rec.get("abstract") or ""),
            str(rec.get("decision_comment") or ""),
        ]
    )
    return 1 if CODE_LINK_RE.search(text) else 0


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _tokenize_keywords(kws: List[str]) -> List[str]:
    out: List[str] = []
    for kw in kws:
        kw = str(kw or "").strip().lower()
        if not kw:
            continue
        # keep full keyword as one token + split into words
        out.append(f"kw:{kw}")
        for w in re.split(r"[\s/,\-_;:+()]+", kw):
            w = w.strip()
            if w:
                out.append(f"w:{w}")
    return out


def train_test_split(ids: List[int], test_frac: float = 0.2) -> Tuple[List[int], List[int]]:
    # deterministic split (no RNG dependency): use modular hash
    train, test = [], []
    for i in ids:
        if (i * 2654435761) % 100 < int(test_frac * 100):
            test.append(i)
        else:
            train.append(i)
    return train, test


def multinomial_nb_train(
    X: List[List[str]],
    y: List[str],
    *,
    alpha: float = 1.0,
) -> Dict[str, Any]:
    """
    Simple Multinomial Naive Bayes over token counts.
    Returns a model dict usable by multinomial_nb_predict.
    """
    label_counts: Dict[str, int] = {}
    tok_counts: Dict[str, Dict[str, int]] = {}
    total_tok: Dict[str, int] = {}
    vocab: set[str] = set()

    for toks, label in zip(X, y):
        label_counts[label] = label_counts.get(label, 0) + 1
        if label not in tok_counts:
            tok_counts[label] = {}
            total_tok[label] = 0
        for t in toks:
            vocab.add(t)
            tok_counts[label][t] = tok_counts[label].get(t, 0) + 1
            total_tok[label] += 1

    return {
        "alpha": alpha,
        "label_counts": label_counts,
        "tok_counts": tok_counts,
        "total_tok": total_tok,
        "vocab_size": len(vocab),
        "labels": sorted(label_counts.keys()),
        "n": sum(label_counts.values()),
    }


def multinomial_nb_predict(model: Dict[str, Any], toks: List[str]) -> str:
    alpha = float(model["alpha"])
    label_counts: Dict[str, int] = model["label_counts"]
    tok_counts: Dict[str, Dict[str, int]] = model["tok_counts"]
    total_tok: Dict[str, int] = model["total_tok"]
    vocab_size = int(model["vocab_size"])
    n = int(model["n"])

    best_label = None
    best_score = None
    for label, lc in label_counts.items():
        # log prior
        score = math.log((lc + 1e-9) / (n + 1e-9))
        denom = total_tok[label] + alpha * vocab_size
        counts = tok_counts[label]
        for t in toks:
            score += math.log((counts.get(t, 0) + alpha) / denom)
        if best_score is None or score > best_score:
            best_score = score
            best_label = label
    return str(best_label)


def confusion_matrix(y_true: List[str], y_pred: List[str], labels: List[str]) -> List[List[int]]:
    idx = {l: i for i, l in enumerate(labels)}
    m = [[0 for _ in labels] for _ in labels]
    for yt, yp in zip(y_true, y_pred):
        if yt not in idx or yp not in idx:
            continue
        m[idx[yt]][idx[yp]] += 1
    return m


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--submissions-jsonl",
        default="exports_all_submissions/submissions.jsonl",
        help="Path to submissions.jsonl from openreview_neurips2025_to_markdown.py",
    )
    ap.add_argument(
        "--md-root",
        default="pdf2md/markdown_output",
        help="Optional: directory of per-paper markdown outputs (for MD quality features).",
    )
    ap.add_argument(
        "--out-dir",
        default="analysis",
        help="Output directory for analysis artifacts.",
    )
    ap.add_argument(
        "--summaries-jsonl",
        default="",
        help="Optional: JSONL of per-paper summaries (id + decision/rating/confidence) to override missing fields.",
    )
    ap.add_argument(
        "--top-keywords",
        type=int,
        default=60,
        help="How many top keywords to report in keyword stats.",
    )
    args = ap.parse_args(argv)

    submissions_path = Path(args.submissions_jsonl)
    if not submissions_path.exists():
        raise FileNotFoundError(f"Missing {submissions_path}. Run NEXT_STEPS.md step (export-dir) first.")

    md_index = load_md_feature_index(Path(args.md_root))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = list(iter_jsonl(submissions_path))
    n_total = len(records)

    # Optional: merge better summaries (e.g., from fetch_openreview_summaries.py)
    if args.summaries_jsonl:
        summ_path = Path(args.summaries_jsonl)
        if not summ_path.exists():
            raise FileNotFoundError(f"--summaries-jsonl not found: {summ_path}")
        by_id: Dict[str, Dict[str, Any]] = {}
        for srec in iter_jsonl(summ_path):
            pid = str(srec.get("id", "")).strip()
            if pid:
                by_id[pid] = srec
        for rec in records:
            pid = str(rec.get("id", "")).strip()
            s = by_id.get(pid)
            if not s:
                continue
            # override only summary-like fields
            for k in [
                "decision",
                "decision_comment",
                "review_rating_avg",
                "review_confidence_avg",
                "review_rating_raw",
                "review_confidence_raw",
                "review_count",
            ]:
                if k in s and s[k] not in (None, "", [], 0):
                    rec[k] = s[k]

    # Build per-paper feature rows
    paper_rows: List[Dict[str, Any]] = []
    labeled_accept: List[int] = []
    labeled_rating: List[float] = []

    for rec in records:
        pid = rec.get("id", "")
        acc = is_accepted(rec)
        acc_i = None if acc is None else int(bool(acc))

        r_avg = _safe_float(rec.get("review_rating_avg"))
        c_avg = _safe_float(rec.get("review_confidence_avg"))
        r_cnt = _safe_int(rec.get("review_count"))

        md_feat = md_index.get(str(pid), None)

        row: Dict[str, Any] = {
            "id": pid,
            "title": rec.get("title", ""),
            "primary_area": rec.get("primary_area", ""),
            "venue": rec.get("venue", ""),
            "decision": rec.get("decision", ""),
            "decision_bucket": decision_bucket(rec),
            "accepted_label": acc_i,
            "review_rating_avg": r_avg,
            "review_confidence_avg": c_avg,
            "review_count": r_cnt if r_cnt is not None else 0,
            "author_count": len(rec.get("authors") or []),
            "keyword_count": len(rec.get("keywords") or []),
            "title_chars": len((rec.get("title") or "").strip()),
            "abstract_chars": len((rec.get("abstract") or "").strip()),
            "has_code_link": has_code_link(rec),
            # MD quality features (if present)
            "md_present": 1 if md_feat else 0,
            "md_line_count": md_feat.line_count if md_feat else "",
            "md_figures": md_feat.figure_link_count if md_feat else "",
            "md_tables": md_feat.table_row_count if md_feat else "",
            "md_math_blocks": md_feat.math_block_count if md_feat else "",
            "md_has_broken_table_token": md_feat.has_broken_table_token if md_feat else "",
            "md_path": md_feat.md_path if md_feat else "",
        }
        paper_rows.append(row)

        if acc_i is not None and r_avg is not None:
            labeled_accept.append(acc_i)
            labeled_rating.append(r_avg)

    write_csv(
        out_dir / "paper_features.csv",
        paper_rows,
        fieldnames=list(paper_rows[0].keys()) if paper_rows else [],
    )

    # Primary area stats
    by_area: Dict[str, List[Dict[str, Any]]] = {}
    for r in paper_rows:
        by_area.setdefault(r.get("primary_area", "") or "(missing)", []).append(r)

    area_rows: List[Dict[str, Any]] = []
    for area, rows in sorted(by_area.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        acc_labels = [x["accepted_label"] for x in rows if x["accepted_label"] != "" and x["accepted_label"] is not None]
        acc_rate = statistics.mean(acc_labels) if acc_labels else ""
        rating_vals = [x["review_rating_avg"] for x in rows if isinstance(x.get("review_rating_avg"), (int, float))]
        area_rows.append(
            {
                "primary_area": area,
                "paper_count": len(rows),
                "labeled_count": len(acc_labels),
                "accept_rate_labeled": acc_rate,
                "review_rating_avg_mean": mean_or_none(rating_vals) or "",
                "review_rating_avg_median": median_or_none(rating_vals) or "",
            }
        )

    write_csv(
        out_dir / "primary_area_stats.csv",
        area_rows,
        fieldnames=[
            "primary_area",
            "paper_count",
            "labeled_count",
            "accept_rate_labeled",
            "review_rating_avg_mean",
            "review_rating_avg_median",
        ],
    )

    # Keyword stats (top N by frequency), with acceptance rate when labeled
    kw_counts: Dict[str, int] = {}
    kw_accept: Dict[str, List[int]] = {}
    for r in records:
        kws = r.get("keywords") or []
        acc = is_accepted(r)
        for kw in kws:
            kw = str(kw).strip()
            if not kw:
                continue
            kw_counts[kw] = kw_counts.get(kw, 0) + 1
            if acc is not None:
                kw_accept.setdefault(kw, []).append(int(acc))

    top_k = sorted(kw_counts.items(), key=lambda kv: (-kv[1], kv[0]))[: args.top_keywords]
    kw_rows: List[Dict[str, Any]] = []
    for kw, cnt in top_k:
        labels = kw_accept.get(kw, [])
        kw_rows.append(
            {
                "keyword": kw,
                "paper_count": cnt,
                "labeled_count": len(labels),
                "accept_rate_labeled": (statistics.mean(labels) if labels else ""),
            }
        )
    write_csv(
        out_dir / "keyword_stats_top.csv",
        kw_rows,
        fieldnames=["keyword", "paper_count", "labeled_count", "accept_rate_labeled"],
    )

    # Rating bin analysis (simple “acceptance criteria” proxy)
    # Only for papers where we have BOTH accept label + rating.
    pairs = [(a, r) for a, r in zip(labeled_accept, labeled_rating)]
    bins = [(1, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 9), (9, 11)]
    bin_rows: List[Dict[str, Any]] = []
    for lo, hi in bins:
        in_bin = [a for (a, r) in pairs if r >= lo and r < hi]
        if not in_bin:
            continue
        bin_rows.append(
            {
                "rating_bin": f"[{lo},{hi})",
                "count": len(in_bin),
                "accept_rate": statistics.mean(in_bin),
            }
        )
    write_csv(out_dir / "rating_bins.csv", bin_rows, fieldnames=["rating_bin", "count", "accept_rate"])

    # Prediction baseline: venue bucket (oral/spotlight/poster/submitted) from metadata keywords
    # This avoids relying on non-public review fields.
    pred_labels = {"oral", "spotlight", "poster", "submitted"}
    pred_X: List[List[str]] = []
    pred_y: List[str] = []
    pred_rows: List[Dict[str, Any]] = []
    for rec in records:
        yb = decision_bucket(rec)
        if yb not in pred_labels:
            continue
        toks = []
        toks.extend(_tokenize_keywords(rec.get("keywords") or []))
        pa = str(rec.get("primary_area") or "").strip().lower()
        if pa:
            toks.append(f"pa:{pa}")
        toks.append(f"code:{has_code_link(rec)}")
        # coarse length features
        alen = len((rec.get("abstract") or "").strip())
        if alen:
            toks.append(f"absbin:{min(alen // 500, 10)}")
        pred_X.append(toks)
        pred_y.append(yb)
        pred_rows.append({"id": rec.get("id", ""), "label": yb})

    if pred_y:
        ids = list(range(len(pred_y)))
        train_ids, test_ids = train_test_split(ids, test_frac=0.2)
        X_tr = [pred_X[i] for i in train_ids]
        y_tr = [pred_y[i] for i in train_ids]
        X_te = [pred_X[i] for i in test_ids]
        y_te = [pred_y[i] for i in test_ids]
        model = multinomial_nb_train(X_tr, y_tr, alpha=1.0)
        y_hat = [multinomial_nb_predict(model, x) for x in X_te]
        labels_sorted = sorted(pred_labels)
        cm = confusion_matrix(y_te, y_hat, labels_sorted)
        acc = sum(1 for a, b in zip(y_te, y_hat) if a == b) / max(len(y_te), 1)

        # save a small prediction report
        pred_md = []
        pred_md.append("# Baseline prediction: venue bucket")
        pred_md.append("")
        pred_md.append("This predicts `oral/spotlight/poster/submitted` using a simple Multinomial Naive Bayes model over:")
        pred_md.append("- keywords (full keyword + split words)")
        pred_md.append("- primary_area")
        pred_md.append("- abstract length bin")
        pred_md.append("- presence of a code-host link (github/gitlab/bitbucket) in title/abstract/decision_comment")
        pred_md.append("")
        pred_md.append(f"- Train size: {len(train_ids)}")
        pred_md.append(f"- Test size: {len(test_ids)}")
        pred_md.append(f"- Accuracy: {acc:.3f}")
        pred_md.append("")
        pred_md.append("## Confusion matrix (rows=true, cols=pred)")
        pred_md.append("")
        pred_md.append("| true \\ pred | " + " | ".join(labels_sorted) + " |")
        pred_md.append("|---|" + "|".join(["---"] * len(labels_sorted)) + "|")
        for li, row in zip(labels_sorted, cm):
            pred_md.append("| " + li + " | " + " | ".join(str(x) for x in row) + " |")
        pred_md.append("")
        (out_dir / "prediction_venue_bucket.md").write_text("\n".join(pred_md) + "\n", encoding="utf-8")

    # Write a human report (markdown)
    n_labeled = sum(1 for r in paper_rows if r["accepted_label"] is not None and r["accepted_label"] != "")
    n_accepted_labeled = sum(1 for r in paper_rows if r["accepted_label"] == 1)
    n_reject_labeled = sum(1 for r in paper_rows if r["accepted_label"] == 0)
    n_md = sum(1 for r in paper_rows if r["md_present"] == 1)

    report = []
    report.append("# NeurIPS 2025 corpus analysis")
    report.append("")
    report.append(f"- **Total submissions (records)**: {n_total}")
    report.append(f"- **Labeled accept/reject (from decision/venue)**: {n_labeled}")
    report.append(f"  - accepted labeled: {n_accepted_labeled}")
    report.append(f"  - rejected labeled: {n_reject_labeled}")
    report.append(f"- **Papers with local PDF→MD available (pdf2md/markdown_output)**: {n_md}")
    report.append("")
    report.append("## Outputs")
    report.append("")
    report.append("- `paper_features.csv`: one row per submission with metadata + optional MD-quality features")
    report.append("- `primary_area_stats.csv`: counts + labeled acceptance rate per primary area")
    report.append("- `keyword_stats_top.csv`: top keywords with labeled acceptance rate (when available)")
    report.append("- `rating_bins.csv`: acceptance rate vs average review rating bins (when available)")
    report.append("")
    report.append("## Notes / limitations")
    report.append("")
    report.append("- Some decision/review summaries may be missing due to OpenReview rate limiting (HTTP 429).")
    report.append("- If you need full coverage of decision/review summaries, rerun the export with a larger `--sleep` (e.g. 1.0–2.0) or rerun later.")
    report.append("- Review ratings/confidence may be missing if official reviews are not public in the forum head notes; in that case, prediction uses only metadata (keywords/area/etc.).")
    report.append("- The MD-quality features are heuristics; they’re useful for trends, but they’re not a substitute for a human quality assessment.")
    report.append("")

    (out_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


