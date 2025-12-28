# ICLR 2025 Paper Processing Pipeline

## Overview
- **Total accepted papers**: ~3,710 (out of 11,565 submissions, ~32.08% acceptance rate)
- **Venue categories**: oral, spotlight, poster
- **Source**: OpenReview API (`ICLR.cc/2025/Conference/-/Submission`)
- **Conference dates**: April 24-28, 2025, Singapore

## Directory Structure
```
/scratch/gautschi/shin283/paper/iclr_2025/
├── openreview_iclr2025_to_markdown.py  # Fetch metadata & download PDFs
├── pdfs/                                # Downloaded PDFs (~3,710 files)
├── exports/                             # Metadata exports
│   ├── submissions.jsonl
│   └── submissions.csv
├── iclr2025.md                          # Markdown summary of all papers
├── pdf2md/
│   ├── slurm_batch_convert.sh          # Main SLURM job for conversion
│   ├── postprocess_marker.py           # Clean up Marker output
│   ├── batch_convert.py                # Python batch converter
│   ├── markdown_output/                # Converted markdown files
│   └── logs/
└── logs/
```

## Pipeline Steps

### Step 1: Download PDFs from OpenReview
```bash
cd /scratch/gautschi/shin283/paper/iclr_2025
python3 openreview_iclr2025_to_markdown.py \
    --download-pdfs \
    --pdf-dir ./pdfs \
    --export-dir ./exports \
    --out ./iclr2025.md
```

**Notes:**
- Script filters accepted papers (excludes "Withdrawn" and "Rejected")
- Includes retry logic with exponential backoff for 429 errors
- 0.5s delay between downloads to avoid rate limiting
- Resume support: skips already-downloaded PDFs

### Step 2: Convert PDFs to Markdown (GPU required)
```bash
cd /scratch/gautschi/shin283/paper/iclr_2025/pdf2md
sbatch slurm_batch_convert.sh
```

**SLURM Configuration:**
- Partition: `ai` (H100 GPUs)
- Account: `jhaddock`
- Resources: 1 GPU, 14 CPUs, 64GB RAM
- Time limit: 2 days
- Uses Marker for conversion

### Step 3: Monitor Progress
```bash
# Check download progress
tail -f /scratch/gautschi/shin283/paper/iclr_2025/download_full.log

# Check conversion progress
tail -f /scratch/gautschi/shin283/paper/iclr_2025/pdf2md/logs/*_pdf2md.out

# Count completed conversions
ls /scratch/gautschi/shin283/paper/iclr_2025/pdf2md/markdown_output/ | wc -l
```

## Key Differences from Other Conferences

### Venue Filter
ICLR uses venues like:
- `ICLR 2025 poster`
- `ICLR 2025 spotlight`
- `ICLR 2025 oral`
- `ICLR 2025 Conference Withdrawn Submission` (excluded)

The script filters to only include papers where venue starts with "ICLR 2025" and excludes "Withdrawn" or "Rejected" in the venue string.

## Related Pipelines
- NeurIPS 2025: `/scratch/gautschi/shin283/paper/neurips_2025/` (~4,972 papers)
- ICML 2025: `/scratch/gautschi/shin283/paper/icml_2025/` (~3,260 papers)

## Sources
- [ICLR 2025 Statistics](https://papercopilot.com/statistics/iclr-statistics/iclr-2025-statistics/)
- [ICLR 2025 Paper List](https://papercopilot.com/paper-list/iclr-paper-list/iclr-2025-paper-list/)
