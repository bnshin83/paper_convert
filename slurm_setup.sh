#!/bin/bash
#SBATCH --job-name=setup_pdf2md
#SBATCH --account=jhaddock
#SBATCH --partition=training
#SBATCH --qos=training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/gilbreth/shin283/paper_convert/logs/setup_%j.out
#SBATCH --error=/scratch/gilbreth/shin283/paper_convert/logs/setup_%j.err

set -e

WORK_DIR="/scratch/gilbreth/shin283/paper_convert"
CACHE_DIR="/scratch/gilbreth/shin283/cache"
VENV_DIR="/scratch/gilbreth/shin283/paper_convert/.venv"

echo "========================================="
echo "Setting up PDF2MD environment on Gilbreth"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "========================================="

cd ${WORK_DIR}

# Create cache directories on scratch
mkdir -p ${CACHE_DIR}/{pip,huggingface,torch}
mkdir -p ${WORK_DIR}/logs

# Set ALL cache/config to scratch (avoid home directory)
export PIP_CACHE_DIR="${CACHE_DIR}/pip"
export HF_HOME="${CACHE_DIR}/huggingface"
export HUGGINGFACE_HUB_CACHE="${CACHE_DIR}/huggingface/hub"
export TORCH_HOME="${CACHE_DIR}/torch"
export XDG_CACHE_HOME="${CACHE_DIR}"
export PYTHONUSERBASE="${CACHE_DIR}/python_user"
export HOME_BACKUP="$HOME"

# Load modules (Gilbreth-specific versions)
module load cuda/12.1
module load python/3.11

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Venv location: ${VENV_DIR}"

# Remove old venv if exists
if [ -d "${VENV_DIR}" ]; then
    echo "Removing old virtual environment..."
    rm -rf ${VENV_DIR}
fi

echo "Creating virtual environment at ${VENV_DIR}..."
python -m venv ${VENV_DIR}
source ${VENV_DIR}/bin/activate

# Verify we're using scratch venv
echo "Active Python: $(which python)"
echo "Active pip: $(which pip)"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Installing Marker..."
pip install marker-pdf

echo "Installing other dependencies..."
pip install requests pdfminer.six

echo ""
echo "Testing installation..."
python -c "import marker; print('Marker: OK')"
python -c "import torch; print(f'PyTorch: OK, CUDA available: {torch.cuda.is_available()}')"
nvidia-smi --query-gpu=name --format=csv,noheader

echo ""
echo "========================================="
echo "Setup complete!"
echo "End time: $(date)"
echo "========================================="
