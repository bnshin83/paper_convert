#!/bin/bash
#SBATCH --job-name=pdf2md_batch_opt
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/gautschi/shin283/paper/neurips_2025/pdf2md/logs/%j_pdf2md_opt.out
#SBATCH --error=/scratch/gautschi/shin283/paper/neurips_2025/pdf2md/logs/%j_pdf2md_opt.err

set -euo pipefail

WORK_DIR="/scratch/gautschi/shin283/paper/neurips_2025/pdf2md"
CACHE_DIR="/scratch/gautschi/shin283/cache"
mkdir -p "${WORK_DIR}/logs"
mkdir -p "${CACHE_DIR}"
cd "${WORK_DIR}"

module load cuda
module load python

# Activate venv
source "${WORK_DIR}/.venv/bin/activate"

# Put caches on scratch
export PIP_CACHE_DIR="${CACHE_DIR}/pip"
export HF_HOME="${CACHE_DIR}/huggingface"
export HUGGINGFACE_HUB_CACHE="${CACHE_DIR}/huggingface/hub"
export TORCH_HOME="${CACHE_DIR}/torch"

# Max CPU usage: use whatever SLURM granted (exclusive node => usually all cores)
CPUS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-1}}"
if [ "${CPUS}" -lt 1 ]; then CPUS=1; fi
export OMP_NUM_THREADS="${CPUS}"
export OPENBLAS_NUM_THREADS="${CPUS}"
export MKL_NUM_THREADS="${CPUS}"
export NUMEXPR_NUM_THREADS="${CPUS}"

# GPU policy (GPU-only by default)
export PDF2MD_DEVICE="${PDF2MD_DEVICE:-cuda}"

echo "========================================="
echo "PDF2MD Batch (optimal CPU usage)"
echo "Job ID: ${SLURM_JOB_ID:-n/a}"
echo "Node: $(hostname)"
echo "CPUs: ${CPUS}"
echo "GPU device policy: ${PDF2MD_DEVICE}"
echo "Start time: $(date)"
echo "========================================="

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# -----------------------------
# EDIT THESE
# -----------------------------
INPUT_DIR="/scratch/gautschi/shin283/paper/neurips_2025/pdfs"
OUTPUT_DIR="${WORK_DIR}/markdown_output"
ENGINE="${ENGINE:-marker}"          # marker | mineru
USE_LLM="${USE_LLM:-0}"             # 1 to enable marker --use_llm
BATCH_MODE="${BATCH_MODE:-1}"       # marker batch mode (fastest): 1/0
IMAGES_SUBDIR="${IMAGES_SUBDIR:-images}"
KEEP_SPANS="${KEEP_SPANS:-0}"       # keep Marker spans in MD (1) or strip (0)

mkdir -p "${OUTPUT_DIR}"

PDF_COUNT=$(find "${INPUT_DIR}" -name "*.pdf" | wc -l)
echo "Found ${PDF_COUNT} PDFs"
echo ""

if [ "${ENGINE}" = "marker" ]; then
  if [ "${BATCH_MODE}" = "1" ]; then
    echo "Marker native batch mode (GPU)"
    # Marker CLI handles its own worker strategy on GPU; our postprocess runs inside batch_convert.py
    ARGS="--batch_mode"
  else
    echo "Marker per-PDF parallel mode (CPU workers drive concurrency; beware GPU contention if >1)"
    ARGS=""
  fi

  LLM_FLAG=""
  if [ "${USE_LLM}" = "1" ]; then LLM_FLAG="--use_llm"; fi

  SPAN_FLAG=""
  if [ "${KEEP_SPANS}" = "1" ]; then SPAN_FLAG="--keep_spans"; fi

  # If not batch_mode, allow parallelism up to CPU count.
  WORKERS="${CPUS}"
  if [ "${BATCH_MODE}" = "1" ]; then WORKERS=1; fi

  python batch_convert.py \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --engine marker \
    --workers "${WORKERS}" \
    ${LLM_FLAG} \
    ${ARGS} \
    --images_subdir "${IMAGES_SUBDIR}" \
    ${SPAN_FLAG} \
    --resume

else
  echo "MinerU (magic-pdf) batch mode"
  # MinerU CLI can be CPU-heavy; parallelize via workers.
  python batch_convert.py \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --engine mineru \
    --workers "${CPUS}" \
    --resume
fi

echo ""
echo "========================================="
echo "Completed at: $(date)"
echo "Output: ${OUTPUT_DIR}"
echo "========================================="


