#!/bin/bash
#SBATCH --job-name=pdf2md_full
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/gautschi/shin283/paper/neurips_2025/pdf2md/logs/%j_pdf2md.out
#SBATCH --error=/scratch/gautschi/shin283/paper/neurips_2025/pdf2md/logs/%j_pdf2md.err

# ============================================
# Batch PDF to Markdown Conversion on H100
# Full NeurIPS 2025 dataset (~5000 PDFs)
# ============================================

echo "========================================="
echo "PDF to Markdown Batch Conversion"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "========================================="

# Setup
WORK_DIR="/scratch/gautschi/shin283/paper/neurips_2025/pdf2md"
CACHE_DIR="/scratch/gautschi/shin283/cache"
cd ${WORK_DIR}

module load cuda
module load python

# Set cache directories to scratch (avoid filling home quota)
export PIP_CACHE_DIR="${CACHE_DIR}/pip"
export HF_HOME="${CACHE_DIR}/huggingface"
export HUGGINGFACE_HUB_CACHE="${CACHE_DIR}/huggingface/hub"
export TORCH_HOME="${CACHE_DIR}/torch"

# Activate virtual environment
source ${WORK_DIR}/.venv/bin/activate

# Configuration
INPUT_DIR="/scratch/gautschi/shin283/paper/neurips_2025/pdfs"
OUTPUT_DIR="${WORK_DIR}/markdown_output"

# Create output and log directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${WORK_DIR}/logs

# GPU info
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Count PDFs
PDF_COUNT=$(find ${INPUT_DIR} -name "*.pdf" | wc -l)
echo "Found ${PDF_COUNT} PDFs to process"
echo ""

# Check GPU availability
echo "Checking CUDA..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Verify GPU works
python << 'PYCHECK'
import torch
import sys
if not torch.cuda.is_available():
    print("ERROR: CUDA not available")
    sys.exit(1)
try:
    x = torch.zeros(1, device='cuda')
    del x
    torch.cuda.empty_cache()
    print("GPU allocation test: PASSED")
except Exception as e:
    print(f"ERROR: GPU allocation failed: {e}")
    sys.exit(1)
PYCHECK

if [ $? -ne 0 ]; then
    echo "GPU check failed. Exiting."
    exit 1
fi

echo ""
echo "Starting Marker batch conversion..."
echo "This will take several hours for ${PDF_COUNT} PDFs"
echo ""

# Run Marker in batch mode with post-processing
python << 'PYEOF'
import os
import sys
import time
from pathlib import Path

# Paths
input_dir = Path("/scratch/gautschi/shin283/paper/neurips_2025/pdfs")
output_dir = Path("/scratch/gautschi/shin283/paper/neurips_2025/pdf2md/markdown_output")
output_dir.mkdir(exist_ok=True)

print("Loading Marker...")
sys.stdout.flush()

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import save_output

# Import post-processor
try:
    from postprocess_marker import MarkerPostprocessConfig, postprocess_marker_output_dir
    HAS_POSTPROCESS = True
    print("Post-processing enabled (images folder + clean markdown)")
except ImportError:
    HAS_POSTPROCESS = False
    print("Post-processing not available")

print("Creating model dict (this may take a minute)...")
sys.stdout.flush()
models = create_model_dict()

print("Creating converter...")
sys.stdout.flush()
converter = PdfConverter(artifact_dict=models)

# Get all PDFs
pdfs = sorted(input_dir.glob("*.pdf"))
total = len(pdfs)
print(f"Found {total} PDFs to convert")
sys.stdout.flush()

# Track progress
success = 0
errors = 0
error_list = []
start_time = time.time()

for i, pdf_path in enumerate(pdfs):
    # Check if already converted
    out_subdir = output_dir / pdf_path.stem
    md_file = out_subdir / f"{pdf_path.stem}.md"

    if md_file.exists():
        print(f"[{i+1}/{total}] Skipping (exists): {pdf_path.name}")
        success += 1
        continue

    print(f"[{i+1}/{total}] Converting: {pdf_path.name}")
    sys.stdout.flush()

    try:
        rendered = converter(str(pdf_path))
        out_subdir.mkdir(parents=True, exist_ok=True)
        save_output(rendered, str(out_subdir), pdf_path.stem)

        # Post-process if available
        if HAS_POSTPROCESS:
            postprocess_marker_output_dir(
                out_subdir,
                pdf_path.stem,
                config=MarkerPostprocessConfig(images_subdir="images", strip_spans=True)
            )

        success += 1

        # Progress update every 100 PDFs
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"  Progress: {i+1}/{total} ({100*(i+1)/total:.1f}%) | Rate: {rate:.2f} PDF/s | ETA: {eta/3600:.1f}h")
            sys.stdout.flush()

    except Exception as e:
        errors += 1
        error_list.append({"pdf": str(pdf_path), "error": str(e)})
        print(f"  ERROR: {e}")
        sys.stdout.flush()

# Final summary
elapsed = time.time() - start_time
print("")
print("=" * 50)
print(f"Conversion completed in {elapsed/3600:.2f} hours")
print(f"Success: {success}/{total}")
print(f"Errors: {errors}/{total}")
print(f"Rate: {total/elapsed:.2f} PDFs/second")
print("=" * 50)

# Save error log if any
if error_list:
    import json
    error_file = output_dir / "errors.json"
    with open(error_file, "w") as f:
        json.dump(error_list, f, indent=2)
    print(f"Error log saved to: {error_file}")

PYEOF

echo ""
echo "========================================="
echo "Conversion completed"
echo "End time: $(date)"
echo "Output: ${OUTPUT_DIR}"
echo "========================================="

# Summary
SUCCESS_COUNT=$(find ${OUTPUT_DIR} -name "*.md" | wc -l)
echo "Successfully converted: ${SUCCESS_COUNT} / ${PDF_COUNT} PDFs"
