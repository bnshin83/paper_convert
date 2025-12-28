#!/bin/bash
#SBATCH --job-name=neurips25_analyze
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gautschi/shin283/paper/neurips_2025/logs/%j_analyze.out
#SBATCH --error=/scratch/gautschi/shin283/paper/neurips_2025/logs/%j_analyze.err

set -euo pipefail

WORK_DIR="/scratch/gautschi/shin283/paper/neurips_2025"
mkdir -p "${WORK_DIR}/logs"
cd "${WORK_DIR}"

# Use the maximum CPUs that SLURM assigned.
CPUS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-1}}"
if [ "${CPUS}" -lt 1 ]; then CPUS=1; fi

# Make common math libs behave nicely on large CPU allocations
export OMP_NUM_THREADS="${CPUS}"
export OPENBLAS_NUM_THREADS="${CPUS}"
export MKL_NUM_THREADS="${CPUS}"
export NUMEXPR_NUM_THREADS="${CPUS}"

echo "========================================="
echo "NeurIPS 2025 corpus analysis"
echo "Job ID: ${SLURM_JOB_ID:-n/a}"
echo "Node: $(hostname)"
echo "CPUs: ${CPUS}"
echo "Start: $(date)"
echo "========================================="

# (Optional) If you have a venv here, activate it:
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

python analyze_neurips2025.py \
  --submissions-jsonl exports_all_submissions/submissions.jsonl \
  --md-root pdf2md/markdown_output \
  --out-dir analysis \
  --top-keywords 60

echo "========================================="
echo "Done: $(date)"
echo "Outputs: ${WORK_DIR}/analysis/"
echo "========================================="


