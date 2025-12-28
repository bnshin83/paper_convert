# ML Conference Paper Processing Pipeline

Download papers from OpenReview and convert PDFs to Markdown using Marker.

## Supported Conferences

| Conference | Papers | Status |
|------------|--------|--------|
| NeurIPS 2025 | ~4,972 | Processing |
| ICML 2025 | ~3,260 | Downloading |
| ICLR 2025 | ~3,703 | Downloading |

## Directory Structure

```
paper/
├── neurips_2025/
│   ├── openreview_neurips2025_to_markdown.py
│   ├── pdfs/                    # Downloaded PDFs
│   ├── pdf2md/
│   │   ├── slurm_batch_convert.sh
│   │   ├── postprocess_marker.py
│   │   └── markdown_output/     # Converted files
│   └── README.md
├── icml_2025/
│   └── (same structure)
├── iclr_2025/
│   └── (same structure)
└── README.md
```

## Quick Start

### 1. Download PDFs from OpenReview

```bash
cd <conference>_2025
python3 openreview_<conf>2025_to_markdown.py \
    --download-pdfs \
    --pdf-dir ./pdfs \
    --export-dir ./exports \
    --out ./<conf>2025.md
```

### 2. Convert PDFs to Markdown (GPU cluster)

```bash
cd <conference>_2025/pdf2md
sbatch slurm_batch_convert.sh
```

### 3. Monitor Progress

```bash
# Download progress
tail -f <conf>_2025/download_full.log

# Conversion progress
tail -f <conf>_2025/pdf2md/logs/*_pdf2md.out
```

## Cluster Configuration

### Gautschi Cluster
- Partition: `ai` (H100 GPUs)
- Account: `jhaddock`
- Virtual env: `/scratch/gautschi/shin283/paper/neurips_2025/pdf2md/.venv`

### Gilbreth Cluster
- Adapt SLURM scripts for Gilbreth's partition/account settings
- Ensure Marker and dependencies are installed

## Key Scripts

### OpenReview Fetcher
- `openreview_<conf>2025_to_markdown.py`
- Fetches paper metadata from OpenReview API
- Downloads PDFs with rate limiting (0.5s delay)
- Retry logic for 429 errors
- Resume support (skips existing files)

### PDF Converter
- `pdf2md/slurm_batch_convert.sh` - SLURM job script
- Uses Marker for PDF to Markdown conversion
- Post-processing cleans up output (consolidates images, strips spans)
- Skip-if-exists for resume support

## Rate Limiting Notes

OpenReview rate limits API requests. The scripts include:
- 0.5s delay between downloads
- Exponential backoff retry (4s, 8s, 16s)
- Resume support to continue after interruption

## Output Format

Each paper produces:
```
markdown_output/<paper_id>__<title>/
├── <paper_id>__<title>.md       # Main content
├── <paper_id>__<title>_meta.json
└── images/                       # Extracted figures
```

## Parallel Processing

For large datasets, split into multiple SLURM jobs:
```bash
# Part 1: PDFs 1-1500
START_IDX=0 END_IDX=1500 sbatch slurm_part1.sh

# Part 2: PDFs 1501-3000 (exclude part1's node)
sbatch --exclude=h000 slurm_part2.sh
```
