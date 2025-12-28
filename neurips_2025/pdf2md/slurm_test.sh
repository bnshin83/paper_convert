#!/bin/bash
#SBATCH --job-name=pdf2md_test
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gautschi/shin283/paper/neurips_2025/pdf2md/logs/%j_test.out
#SBATCH --error=/scratch/gautschi/shin283/paper/neurips_2025/pdf2md/logs/%j_test.err

echo "========================================="
echo "PDF to Markdown TEST (3 PDFs)"
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

# Set cache directories to scratch
export PIP_CACHE_DIR="${CACHE_DIR}/pip"
export HF_HOME="${CACHE_DIR}/huggingface"
export HUGGINGFACE_HUB_CACHE="${CACHE_DIR}/huggingface/hub"
export TORCH_HOME="${CACHE_DIR}/torch"

# Activate virtual environment
source ${WORK_DIR}/.venv/bin/activate

# Test with sample PDFs
INPUT_DIR="${WORK_DIR}/sample_pdfs"
OUTPUT_DIR="${WORK_DIR}/sample_output"
ENGINE="mineru"
# If you want to allow CPU fallback, submit with: sbatch --export=ALL,PDF2MD_DEVICE=auto slurm_test.sh
PDF2MD_DEVICE="${PDF2MD_DEVICE:-cuda}"  # auto|cpu|cuda
export PDF2MD_DEVICE

mkdir -p ${OUTPUT_DIR}

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

PDF_COUNT=$(find ${INPUT_DIR} -name "*.pdf" | wc -l)
echo "Found ${PDF_COUNT} PDFs to process"
echo ""

echo "Using Marker Python API directly (no multiprocessing)"

# Debug GPU access
echo "Checking CUDA..."
nvidia-smi
echo ""

# Debug PyTorch CUDA
python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"
echo ""

# Check SLURM GPU environment
echo "SLURM GPU vars:"
echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"

# Use SLURM-assigned GPU
python << 'PYEOF'
import os
import sys
from pathlib import Path

requested_device = os.environ.get("PDF2MD_DEVICE", "auto").lower()
if requested_device not in {"auto", "cpu", "cuda"}:
    raise ValueError(f"Invalid PDF2MD_DEVICE={requested_device}. Use auto|cpu|cuda")

print(f"PDF2MD_DEVICE: {requested_device}")
print(f"CUDA_VISIBLE_DEVICES from env: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version (PyTorch): {torch.version.cuda}")
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")

device = "cpu"
if requested_device == "cpu":
    device = "cpu"
elif requested_device in {"auto", "cuda"} and torch.cuda.is_available():
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    try:
        x = torch.zeros(1, device='cuda')
        print(f"GPU allocation success!")
        del x
        torch.cuda.empty_cache()
        device = "cuda"
    except Exception as e:
        print(f"GPU failed: {e}")
        if requested_device == "cuda":
            print("")
            print("FATAL: GPU requested but CUDA context/allocation failed on this node.")
            print("Suggested next steps:")
            print(f"  - Resubmit on a different node (current: {os.uname().nodename})")
            print(f"    Example: sbatch --exclude={os.uname().nodename.split('.')[0]} slurm_test.sh")
            print("  - Or allow CPU fallback: sbatch --export=ALL,PDF2MD_DEVICE=auto slurm_test.sh")
            print("")
            raise RuntimeError("PDF2MD_DEVICE=cuda requested, but CUDA allocation failed") from e
        print("Will use CPU mode")
        device = "cpu"

print(f"Marker/Surya device: {device}")

print("Loading Marker models...")
sys.stdout.flush()

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import save_output
from postprocess_marker import MarkerPostprocessConfig, postprocess_marker_output_dir

input_dir = Path("/scratch/gautschi/shin283/paper/neurips_2025/pdf2md/sample_pdfs")
output_dir = Path("/scratch/gautschi/shin283/paper/neurips_2025/pdf2md/sample_output")
output_dir.mkdir(exist_ok=True)

print("Creating model dict...")
sys.stdout.flush()
models = create_model_dict(device=device)

print("Creating converter...")
sys.stdout.flush()
converter = PdfConverter(artifact_dict=models)

pdfs = list(input_dir.glob("*.pdf"))
print(f"Found {len(pdfs)} PDFs")

for i, pdf_path in enumerate(pdfs):
    print(f"Converting {i+1}/{len(pdfs)}: {pdf_path.name}")
    sys.stdout.flush()
    try:
        rendered = converter(str(pdf_path))
        out_subdir = output_dir / pdf_path.stem
        out_subdir.mkdir(exist_ok=True)
        # Writes: <stem>.md + <stem>_meta.json + extracted images (figures) referenced by the markdown
        save_output(rendered, str(out_subdir), pdf_path.stem)
        postprocess_marker_output_dir(
            out_subdir,
            pdf_path.stem,
            config=MarkerPostprocessConfig(images_subdir="images", strip_spans=True),
        )
        md_file = out_subdir / f"{pdf_path.stem}.md"
        print(f"  Saved to {md_file} (and extracted images)")
    except Exception as e:
        print(f"  ERROR: {e}")
    sys.stdout.flush()

print("Done!")
PYEOF

echo ""
echo "========================================="
echo "Test completed at $(date)"
echo "Output: ${OUTPUT_DIR}"
echo "========================================="

SUCCESS_COUNT=$(find ${OUTPUT_DIR} -name "*.md" | wc -l)
echo "Successfully converted: ${SUCCESS_COUNT} / ${PDF_COUNT} PDFs"
