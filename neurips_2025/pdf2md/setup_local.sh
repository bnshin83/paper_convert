#!/bin/bash
# ============================================
# Setup PDF2MD environment on local Mac (M4 Max)
# ============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd ${SCRIPT_DIR}

echo "Setting up PDF2MD for Apple Silicon..."

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools

echo ""
echo "========================================="
echo "Installing MinerU..."
echo "========================================="

# Install MinerU (CPU/MPS)
pip install magic-pdf[full]

# Configure MinerU for Mac
python -c "
import json
import os
from pathlib import Path

config_dir = Path.home() / '.magic-pdf'
config_dir.mkdir(exist_ok=True)

config = {
    'device': 'mps',  # Use Apple Metal
    'table-engine': 'rapidocr',
    'formula-engine': 'rapidocr'
}

config_path = config_dir / 'magic-pdf.json'
config_path.write_text(json.dumps(config, indent=2))
print(f'MinerU config saved to {config_path}')
"

# Download models
echo "Downloading MinerU models..."
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='opendatalab/PDF-Extract-Kit', local_dir_use_symlinks=False)
print('Models downloaded')
"

echo ""
echo "========================================="
echo "Installing Marker..."
echo "========================================="

# Install Marker for Mac
pip install marker-pdf

# Download Marker models
python -c "
from marker.models import create_model_dict
models = create_model_dict()
print('Marker models ready')
"

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "Usage examples:"
echo ""
echo "# Single PDF with MinerU:"
echo "magic-pdf -p paper.pdf -o output/"
echo ""
echo "# Single PDF with Marker:"
echo "marker_single paper.pdf --output_dir output/"
echo ""
echo "# Batch convert with MinerU:"
echo "python batch_convert.py -i ./pdfs -o ./markdown -e mineru"
echo ""
echo "# Batch convert with Marker (faster):"
echo "python batch_convert.py -i ./pdfs -o ./markdown -e marker -w 4"
echo ""
echo "# Or use Marker native batch (fastest):"
echo "marker ./pdfs --output_dir ./markdown"
