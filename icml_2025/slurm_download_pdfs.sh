#!/bin/bash
#SBATCH --job-name=icml25_pdf_dl
#SBATCH --account=jhaddock
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/gautschi/shin283/paper/icml_2025/logs/%j_download.out
#SBATCH --error=/scratch/gautschi/shin283/paper/icml_2025/logs/%j_download.err

# ============================================
# Download ICML 2025 PDFs from OpenReview
# Expected: ~3,260 papers
# ============================================

echo "========================================="
echo "ICML 2025 PDF Download"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "========================================="

WORK_DIR="/scratch/gautschi/shin283/paper/icml_2025"
cd ${WORK_DIR}

module load python

# Use system python with requests
python3 openreview_icml2025_to_markdown.py \
    --download-pdfs \
    --pdf-dir "${WORK_DIR}/pdfs" \
    --export-dir "${WORK_DIR}/exports" \
    --out "${WORK_DIR}/icml2025.md"

echo ""
echo "========================================="
echo "Download completed"
echo "End time: $(date)"
echo "========================================="

PDF_COUNT=$(find ${WORK_DIR}/pdfs -name "*.pdf" | wc -l)
echo "Downloaded PDFs: ${PDF_COUNT}"
