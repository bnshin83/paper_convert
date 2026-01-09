#!/bin/bash
# Submit all 6 batch jobs for PDF to Markdown conversion
# Each job runs on a separate GPU for parallel processing

cd /scratch/gautschi/shin283/paper/others/books/pdf2md

echo "Submitting 6 batch jobs for Books PDF conversion (84 PDFs total)..."
echo ""

for i in 1 2 3 4 5 6; do
    JOB_ID=$(sbatch --parsable slurm_batch_${i}.sh)
    echo "Batch ${i}: Job ID ${JOB_ID} submitted"
done

echo ""
echo "All jobs submitted! Monitor with: squeue -u $USER"
echo "Output logs: /scratch/gautschi/shin283/paper/others/books/pdf2md/logs/"
