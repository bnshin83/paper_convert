## Next steps (NeurIPS 2025 corpus for trend/prediction analysis)

### 0) Activate your environment

```bash
cd /Users/bnsmac/neurips_2025
source .venv/bin/activate
```

### 1) Build the analysis dataset (all submissions: accepted + rejected)

For “why accepted vs why rejected”, export **all submissions** and attach decision/review summaries.

This writes:
- `submissions.jsonl` (best for analysis)
- `submissions.csv`
- optional `forum_notes.jsonl` (full reviews/decisions/comments/rebuttals **if public**)

```bash
python openreview_neurips2025_to_markdown.py \
  --all \
  --export-dir exports_all_submissions/ \
  --include-decision-summary \
  --sleep 0.25 \
  --limit 1000
```

Outputs:
- `exports_all_submissions/submissions.jsonl`
- `exports_all_submissions/submissions.csv`

Notes:
- Reviewer identity is typically **anonymous**; you’ll still get note `signatures` and review content fields (when present).
- If you see timeouts, increase `--sleep` (e.g. `0.5` or `1.0`).

If you also want the full forum thread notes (larger export):

```bash
python openreview_neurips2025_to_markdown.py \
  --all \
  --export-dir exports_all_submissions/ \
  --include-forum-notes \
  --sleep 0.25 \
  --limit 1000
```

### 2) Optional: Full text (PDF → Markdown)

This downloads PDFs and writes **one Markdown file per paper** with PDF-extracted full text (best-effort / noisy):

```bash
python openreview_neurips2025_to_markdown.py \
  --convert-pdfs-to-md \
  --pdf-dir pdfs/ \
  --fulltext-out-dir fulltext_md/ \
  --sleep 0.25 \
  --limit 1000
```

Outputs:
- `pdfs/` (downloaded PDFs)
- `fulltext_md/` (per-paper Markdown files with extracted text)

### 3) Optional: Metadata Markdown (human-readable)

```bash
python openreview_neurips2025_to_markdown.py --out neurips2025.md
```


