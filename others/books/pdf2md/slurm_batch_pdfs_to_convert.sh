#!/bin/bash
#SBATCH --job-name=pdfs_to_convert
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/gautschi/shin283/paper/others/books/pdf2md/logs/%j_pdfs_to_convert.out
#SBATCH --error=/scratch/gautschi/shin283/paper/others/books/pdf2md/logs/%j_pdfs_to_convert.err
#SBATCH --exclude=h000,h003

# ============================================
# PDF to Markdown Conversion - pdfs_to_convert
# Converting 5 PDFs from pdfs_to_convert folder
# ============================================

echo "========================================="
echo "PDF to Markdown - pdfs_to_convert"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "========================================="

# Setup
WORK_DIR="/scratch/gautschi/shin283/paper/others/books/pdf2md"
CACHE_DIR="/scratch/gautschi/shin283/cache"
cd ${WORK_DIR}

module load cuda
module load python

export PIP_CACHE_DIR="${CACHE_DIR}/pip"
export HF_HOME="${CACHE_DIR}/huggingface"
export HUGGINGFACE_HUB_CACHE="${CACHE_DIR}/huggingface/hub"
export TORCH_HOME="${CACHE_DIR}/torch"

source /scratch/gautschi/shin283/paper/neurips_2025/pdf2md/.venv/bin/activate

INPUT_DIR="/scratch/gautschi/shin283/paper/others/books/pdfs_to_convert"
OUTPUT_DIR="${WORK_DIR}/markdown_output"
mkdir -p ${OUTPUT_DIR}
mkdir -p ${WORK_DIR}/logs

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

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
echo "Starting Marker batch conversion - pdfs_to_convert..."
echo ""

python << PYEOF
import os
import sys
import time
from pathlib import Path

input_dir = Path("/scratch/gautschi/shin283/paper/others/books/pdfs_to_convert")
output_dir = Path("/scratch/gautschi/shin283/paper/others/books/pdf2md/markdown_output")
output_dir.mkdir(exist_ok=True)

print("Loading Marker...")
sys.stdout.flush()

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import save_output

try:
    sys.path.insert(0, "/scratch/gautschi/shin283/paper/neurips_2025/pdf2md")
    from postprocess_marker import MarkerPostprocessConfig, postprocess_marker_output_dir
    HAS_POSTPROCESS = True
    print("Post-processing enabled")
except ImportError:
    HAS_POSTPROCESS = False
    print("Post-processing not available")

print("Creating model dict...")
sys.stdout.flush()
models = create_model_dict()

print("Creating converter...")
sys.stdout.flush()
converter = PdfConverter(artifact_dict=models)

all_pdfs = sorted(input_dir.rglob("*.pdf"))
total = len(all_pdfs)
print(f"Found {total} PDFs to convert")
sys.stdout.flush()

success = 0
skipped = 0
errors = 0
error_list = []
start_time = time.time()

for i, pdf_path in enumerate(all_pdfs):
    rel_path = pdf_path.relative_to(input_dir)
    if rel_path.parent != Path("."):
        out_name = "pdfs_to_convert_" + str(rel_path.parent).replace("/", "_").replace(" ", "_") + "_" + pdf_path.stem
    else:
        out_name = "pdfs_to_convert_" + pdf_path.stem
    
    out_subdir = output_dir / out_name
    md_file = out_subdir / f"{pdf_path.stem}.md"

    if md_file.exists():
        print(f"[{i+1}/{total}] Skipping (exists): {rel_path}")
        skipped += 1
        success += 1
        continue

    print(f"[{i+1}/{total}] Converting: {rel_path}")
    sys.stdout.flush()

    try:
        rendered = converter(str(pdf_path))
        out_subdir.mkdir(parents=True, exist_ok=True)
        save_output(rendered, str(out_subdir), pdf_path.stem)

        if HAS_POSTPROCESS:
            postprocess_marker_output_dir(
                out_subdir,
                pdf_path.stem,
                config=MarkerPostprocessConfig(images_subdir="images", strip_spans=True)
            )

        success += 1

    except Exception as e:
        errors += 1
        error_list.append({"pdf": str(pdf_path), "error": str(e)})
        print(f"  ERROR: {e}")
        sys.stdout.flush()

elapsed = time.time() - start_time
print("")
print("=" * 50)
print(f"Conversion completed in {elapsed/60:.2f} minutes")
print(f"Success: {success}/{total}, Skipped: {skipped}, Errors: {errors}")
print("=" * 50)

if error_list:
    import json
    error_file = output_dir / "errors_pdfs_to_convert.json"
    with open(error_file, "w") as f:
        json.dump(error_list, f, indent=2)
    print(f"Error log: {error_file}")

PYEOF

echo ""
echo "pdfs_to_convert conversion completed at $(date)"
