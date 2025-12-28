#!/bin/bash
# Setup script for Gilbreth cluster
# Run this once to create the virtual environment and install dependencies

set -e

WORK_DIR="/scratch/gilbreth/shin283/paper_convert"
CACHE_DIR="/scratch/gilbreth/shin283/cache"

echo "========================================="
echo "Setting up PDF2MD environment on Gilbreth"
echo "========================================="

cd ${WORK_DIR}

# Create cache directories
mkdir -p ${CACHE_DIR}/{pip,huggingface,torch}

# Set cache environment variables
export PIP_CACHE_DIR="${CACHE_DIR}/pip"
export HF_HOME="${CACHE_DIR}/huggingface"
export HUGGINGFACE_HUB_CACHE="${CACHE_DIR}/huggingface/hub"
export TORCH_HOME="${CACHE_DIR}/torch"

# Load modules (adjust for Gilbreth)
module load cuda
module load python

echo "Creating virtual environment..."
python -m venv .venv
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Installing Marker..."
pip install marker-pdf

echo "Installing other dependencies..."
pip install requests pdfminer.six

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "To activate the environment:"
echo "  source ${WORK_DIR}/.venv/bin/activate"
echo ""
echo "To test:"
echo "  python -c \"import marker; import torch; print(f'CUDA: {torch.cuda.is_available()}')\""
