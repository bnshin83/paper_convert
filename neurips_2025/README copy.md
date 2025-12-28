## NeurIPS 2025 → Markdown (OpenReview API, no API key)

This repo contains a small script that downloads **paper metadata** for NeurIPS 2025 from OpenReview and exports it as **Markdown** (title, authors, abstract, keywords, OpenReview link, PDF link, BibTeX).

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Generate one big Markdown file

```bash
python openreview_neurips2025_to_markdown.py --out neurips2025.md
```

### “Delete all subsets” (flat export: hide primary areas + keywords)

```bash
python openreview_neurips2025_to_markdown.py --no-subsets --out neurips2025_flat.md
```

### Download PDFs (optional)

```bash
python openreview_neurips2025_to_markdown.py --download-pdfs --pdf-dir pdfs/
```

### Convert PDFs → Markdown (optional, best-effort)

PDF-to-Markdown conversion is inherently noisy (multi-column layouts, math, figures/tables). This option uses `pdfminer.six` to extract text and writes **one `.md` per paper**.

```bash
python openreview_neurips2025_to_markdown.py --convert-pdfs-to-md --pdf-dir pdfs/ --fulltext-out-dir fulltext_md/
```

### Generate one Markdown file per paper

```bash
python openreview_neurips2025_to_markdown.py --split-out-dir neurips2025_md/
```

### List all primary areas (subsets) and counts

```bash
python openreview_neurips2025_to_markdown.py --list-primary-areas
```

### Export machine-readable datasets (recommended for trend/prediction work)

This writes:
- `submissions.jsonl` (one JSON record per paper)
- `submissions.csv`

```bash
python openreview_neurips2025_to_markdown.py --export-dir exports/
```

If NeurIPS has made reviews/comments/decisions public for the forum threads, you can also export them:

```bash
python openreview_neurips2025_to_markdown.py --export-dir exports/ --include-forum-notes
```

If you want a simple **accept/reject label** directly in `submissions.jsonl/.csv`, add:

```bash
python openreview_neurips2025_to_markdown.py --export-dir exports/ --include-decision-summary
```

Note: “reviewer identity” is typically **anonymous** on OpenReview. You’ll still get `signatures` for each review note (often anonymous IDs), plus rating/confidence if present in the review content.

### Accepted-only vs all submissions

Default is **accepted-only** (keeps papers where `venue` starts with `"NeurIPS 2025"`).

Export everything:

```bash
python openreview_neurips2025_to_markdown.py --all --out neurips2025_all.md
```

### Quick smoke test (first 20 papers)

```bash
python openreview_neurips2025_to_markdown.py --max-papers 20 --out sample.md
```


