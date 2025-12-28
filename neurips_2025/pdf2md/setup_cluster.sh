#!/bin/bash
# ============================================
# Setup PDF2MD environment on H100 cluster
# Run this ONCE to set up the environment
# ============================================

set -e

WORK_DIR="/scratch/gautschi/shin283/paper/neurips_2025/pdf2md"
CACHE_DIR="/scratch/gautschi/shin283/cache"
mkdir -p ${WORK_DIR}/logs
mkdir -p ${CACHE_DIR}
cd ${WORK_DIR}

echo "Setting up PDF2MD environment..."
echo "Working directory: ${WORK_DIR}"
echo "Cache directory: ${CACHE_DIR}"

# Set cache directories to scratch (avoid filling home quota)
export PIP_CACHE_DIR="${CACHE_DIR}/pip"
export HF_HOME="${CACHE_DIR}/huggingface"
export HUGGINGFACE_HUB_CACHE="${CACHE_DIR}/huggingface/hub"
export TORCH_HOME="${CACHE_DIR}/torch"

# Load modules
module load cuda
module load python

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools

echo ""
echo "========================================="
echo "Installing MinerU (best for academic papers)..."
echo "========================================="

# Install MinerU with CUDA support
pip install magic-pdf[full]

# Download MinerU models (required)
echo "Downloading MinerU models..."
python -c "
from huggingface_hub import snapshot_download
import os

# Download layout model to scratch cache
cache_dir = os.environ.get('HF_HOME', '${CACHE_DIR}/huggingface')
snapshot_download(
    repo_id='opendatalab/PDF-Extract-Kit',
    local_dir=os.path.join(cache_dir, 'hub/PDF-Extract-Kit')
)
print('MinerU models downloaded successfully')
"

echo ""
echo "========================================="
echo "Installing Marker (fastest batch processing)..."
echo "========================================="

# Install Marker
pip install marker-pdf

# Download Marker models
echo "Downloading Marker models..."
python -c "
from marker.models import create_model_dict
models = create_model_dict()
print('Marker models downloaded successfully')
"

echo ""
echo "========================================="
echo "Installing additional dependencies..."
echo "========================================="

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Copy your PDFs to: ${WORK_DIR}/pdfs/"
echo "2. Copy batch_convert.py to: ${WORK_DIR}/"
echo "3. Edit slurm_batch_convert.sh paths if needed"
echo "4. Submit job: sbatch slurm_batch_convert.sh"
echo ""
echo "Virtual environment: ${WORK_DIR}/.venv"
echo "Activate with: source ${WORK_DIR}/.venv/bin/activate"
