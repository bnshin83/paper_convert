#!/bin/bash
# Submit all 6 PDF conversion jobs

SCRIPT_DIR="/scratch/gilbreth/shin283/paper_convert/icml_2025/pdf2md"

# Create logs directory
mkdir -p ${SCRIPT_DIR}/logs

echo "Submitting 6 ICML 2025 PDF conversion jobs..."
echo "Each job will process ~532 PDFs on a separate A100-80GB GPU"
echo ""

for i in {0..5}; do
    JOB_ID=$(sbatch ${SCRIPT_DIR}/slurm_batch_convert_${i}.sh | awk '{print $4}')
    echo "Submitted part ${i}: Job ID ${JOB_ID}"
done

echo ""
echo "All jobs submitted! Monitor with: squeue -u $USER"
echo "Logs will be in: ${SCRIPT_DIR}/logs/"
