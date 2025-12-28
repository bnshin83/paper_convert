# Setting Up PDF-to-Markdown Pipeline on a New Cluster

This guide explains how to set up the paper conversion pipeline on a new HPC cluster.

## Prerequisites

- GPU node access (NVIDIA GPU with CUDA support)
- Python 3.10+
- ~50GB scratch space for models and cache

## Step 1: Clone the Repository

```bash
# Replace with your scratch directory
SCRATCH_DIR="/scratch/<cluster>/<username>"
cd ${SCRATCH_DIR}

git clone https://github.com/bnshin83/paper_convert.git
cd paper_convert
```

## Step 2: Create Cache Directories

Marker downloads large models (~10GB). Store them on scratch to avoid home quota issues.

```bash
CACHE_DIR="${SCRATCH_DIR}/cache"
mkdir -p ${CACHE_DIR}/{pip,huggingface,torch}
```

## Step 3: Set Environment Variables

Add these to your `~/.bashrc` or set them before running:

```bash
export PIP_CACHE_DIR="${SCRATCH_DIR}/cache/pip"
export HF_HOME="${SCRATCH_DIR}/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${SCRATCH_DIR}/cache/huggingface/hub"
export TORCH_HOME="${SCRATCH_DIR}/cache/torch"
```

## Step 4: Create Virtual Environment

```bash
cd ${SCRATCH_DIR}/paper_convert

# Load modules (cluster-specific - adjust as needed)
module load cuda        # or: module load cuda/12.1
module load python      # or: module load python/3.11

# Create venv
python -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

## Step 5: Install PyTorch with CUDA

Choose the appropriate CUDA version for your cluster:

```bash
# For CUDA 12.1 (most common)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify CUDA is working:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

## Step 6: Install Marker

```bash
pip install marker-pdf
```

This will download the Marker models on first run (~10GB to HF_HOME).

## Step 7: Install Additional Dependencies

```bash
pip install requests pdfminer.six
```

## Step 8: Test Installation

```bash
python << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
print("Marker imported successfully!")
print("Creating model dict (this downloads models on first run)...")
models = create_model_dict()
print("Models loaded successfully!")
EOF
```

## Step 9: Adapt SLURM Scripts

Edit the SLURM scripts for your cluster. Key things to change:

### 1. Account and Partition
```bash
#SBATCH --account=<your_account>
#SBATCH --partition=<gpu_partition>   # e.g., gpu, ai, gpu-debug
#SBATCH --qos=<your_qos>              # e.g., normal, standby
```

### 2. Virtual Environment Path
```bash
source /scratch/<cluster>/<username>/paper_convert/.venv/bin/activate
```

### 3. Cache Directory Paths
```bash
export HF_HOME="/scratch/<cluster>/<username>/cache/huggingface"
export TORCH_HOME="/scratch/<cluster>/<username>/cache/torch"
```

### 4. Input/Output Paths
```bash
INPUT_DIR="/scratch/<cluster>/<username>/paper_convert/<conf>_2025/pdfs"
OUTPUT_DIR="/scratch/<cluster>/<username>/paper_convert/<conf>_2025/pdf2md/markdown_output"
```

## Example SLURM Script Template

```bash
#!/bin/bash
#SBATCH --job-name=pdf2md
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

SCRATCH_DIR="/scratch/<cluster>/<username>"
WORK_DIR="${SCRATCH_DIR}/paper_convert"

module load cuda
module load python

export HF_HOME="${SCRATCH_DIR}/cache/huggingface"
export TORCH_HOME="${SCRATCH_DIR}/cache/torch"

source ${WORK_DIR}/.venv/bin/activate

cd ${WORK_DIR}/<conf>_2025/pdf2md

python << 'PYEOF'
# Your conversion code here
PYEOF
```

## Downloading PDFs

Before conversion, download PDFs using the fetch scripts:

```bash
source .venv/bin/activate
cd <conf>_2025

# This downloads from OpenReview (no GPU needed)
python openreview_<conf>2025_to_markdown.py \
    --download-pdfs \
    --pdf-dir ./pdfs \
    --export-dir ./exports \
    --out ./<conf>2025.md
```

## Troubleshooting

### "CUDA not available"
- Check `module list` for loaded CUDA
- Verify GPU allocation: `nvidia-smi`
- Ensure PyTorch CUDA version matches cluster CUDA

### "No module named 'marker'"
- Activate venv: `source .venv/bin/activate`
- Reinstall: `pip install marker-pdf`

### Rate limit errors (429) during PDF download
- The scripts include retry logic with exponential backoff
- Re-run the download script - it skips already-downloaded files

### Out of memory
- Reduce batch size or use smaller GPU
- Marker needs ~20GB GPU memory for optimal performance

### Models not downloading
- Check `HF_HOME` is set correctly
- Ensure scratch has enough space (~10GB for models)
- Check network access to huggingface.co

## Cluster-Specific Notes

### Gilbreth (Purdue)
```bash
module load cuda/12.1
module load python/3.11
#SBATCH --partition=gpu
#SBATCH --account=<your_account>
```

### Gautschi (Purdue)
```bash
module load cuda
module load python
#SBATCH --partition=ai
#SBATCH --account=jhaddock
```

## Directory Structure After Setup

```
/scratch/<cluster>/<username>/
├── paper_convert/
│   ├── .venv/                    # Virtual environment
│   ├── neurips_2025/
│   │   ├── pdfs/                 # Downloaded PDFs
│   │   └── pdf2md/
│   │       └── markdown_output/  # Converted files
│   ├── icml_2025/
│   └── iclr_2025/
└── cache/
    ├── huggingface/              # Marker models
    ├── torch/
    └── pip/
```
