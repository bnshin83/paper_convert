# ICML 2025 Paper Processing Pipeline

## Overview
- **Total accepted papers**: 3,260 (out of 12,107 submissions, ~26.9% acceptance rate)
- **Venue categories**: oral, spotlightposter, poster
- **Source**: OpenReview API (`ICML.cc/2025/Conference/-/Submission`)

## Directory Structure
```
/scratch/gautschi/shin283/paper/icml_2025/
├── openreview_icml2025_to_markdown.py  # Fetch metadata & download PDFs
├── pdfs/                                # Downloaded PDFs (~3,260 files)
├── exports/                             # Metadata exports
│   ├── submissions.jsonl
│   └── submissions.csv
├── icml2025.md                          # Markdown summary of all papers
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
cd /scratch/gautschi/shin283/paper/icml_2025
python3 openreview_icml2025_to_markdown.py \
    --download-pdfs \
    --pdf-dir ./pdfs \
    --export-dir ./exports \
    --out ./icml2025.md
```

**Notes:**
- OpenReview rate limits to ~2 requests/second
- Script includes retry logic with exponential backoff for 429 errors
- 0.5s delay between downloads to avoid rate limiting
- Resume support: skips already-downloaded PDFs
- Expected time: ~1-1.5 hours for 3,260 PDFs

### Step 2: Convert PDFs to Markdown (GPU required)
```bash
cd /scratch/gautschi/shin283/paper/icml_2025/pdf2md
sbatch slurm_batch_convert.sh
```

**SLURM Configuration:**
- Partition: `ai` (H100 GPUs)
- Account: `jhaddock`
- Resources: 1 GPU, 14 CPUs, 64GB RAM
- Time limit: 2 days
- Uses Marker for conversion (~50 seconds/PDF on H100)

**Notes:**
- Reuses NeurIPS 2025 virtual environment: `/scratch/gautschi/shin283/paper/neurips_2025/pdf2md/.venv`
- Skip-if-exists logic prevents duplicate work
- Post-processing cleans up Marker output (consolidates images, strips spans)

### Step 3: Monitor Progress
```bash
# Check download progress
tail -f /scratch/gautschi/shin283/paper/icml_2025/download_full.log

# Check conversion progress
tail -f /scratch/gautschi/shin283/paper/icml_2025/pdf2md/logs/*_pdf2md.out

# Count completed conversions
ls /scratch/gautschi/shin283/paper/icml_2025/pdf2md/markdown_output/ | wc -l
```

## Lessons Learned (from NeurIPS 2025 pipeline)

### Rate Limiting
- OpenReview returns 429 errors if you download too fast
- Solution: Add 0.5s delay between downloads + retry with exponential backoff

### SLURM GPU Scheduling
- Multiple jobs on same node can cause "CUDA device busy" errors
- Solution: Use `--exclude=<node>` to avoid nodes with running jobs
- Check node usage: `squeue -u $USER`

### Parallel Processing
- For large datasets, split into multiple SLURM jobs by PDF range
- Example: Part 1 (1-1000), Part 2 (1001-2000), etc.
- Each job should use `--exclude` to target different GPU nodes

### Virtual Environment
- Marker requires specific dependencies; reuse existing venv when possible
- Cache directories should be on scratch to avoid home quota issues:
  ```bash
  export HF_HOME="/scratch/gautschi/shin283/cache/huggingface"
  export TORCH_HOME="/scratch/gautschi/shin283/cache/torch"
  ```

### Conversion Speed
- Marker on H100: ~50 seconds/PDF average
- 3,260 PDFs ≈ 45 hours single-threaded
- Consider splitting into 2-3 parallel jobs for faster completion

## Splitting Conversion into Parallel Jobs

To speed up conversion, create multiple SLURM scripts with different PDF ranges:

```bash
# In slurm_batch_convert_part1.sh (PDFs 1-1500)
START_IDX = 0
END_IDX = 1500

# In slurm_batch_convert_part2.sh (PDFs 1501-3000)
START_IDX = 1500
END_IDX = 3000

# In slurm_batch_convert_part3.sh (PDFs 3001-3260)
START_IDX = 3000
END_IDX = 3260
```

Submit with node exclusions:
```bash
sbatch slurm_batch_convert_part1.sh  # runs on h000
sbatch --exclude=h000 slurm_batch_convert_part2.sh  # runs on h001
sbatch --exclude=h000,h001 slurm_batch_convert_part3.sh  # runs on h002
```

## Output Format

Each converted paper produces:
```
markdown_output/<paper_id>__<title>/
├── <paper_id>__<title>.md       # Main markdown file
├── <paper_id>__<title>_meta.json # Metadata
└── images/                       # Extracted figures
    ├── figure_1.png
    └── ...
```

## Related: NeurIPS 2025 Pipeline
Located at: `/scratch/gautschi/shin283/paper/neurips_2025/`
- Same structure and tools
- 4,972 accepted papers
- Conversion jobs: 5634584 (original), 5652198 (part 3), 5652202 (part 4)
