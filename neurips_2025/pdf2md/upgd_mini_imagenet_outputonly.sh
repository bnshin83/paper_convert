#!/bin/bash
#SBATCH --job-name=upgd_outputonly_mini_imagenet_seed_2
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=5-00:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_upgd_outputonly_mini_imagenet_seed_2.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_upgd_outputonly_mini_imagenet_seed_2.err

cd /scratch/gautschi/shin283/upgd
module load cuda
module load python

export PYTHONPATH=/scratch/gautschi/shin283/upgd:$PYTHONPATH
source /scratch/gautschi/shin283/upgd/.upgd/bin/activate

# WandB Configuration
export WANDB_PROJECT="upgd"
export WANDB_ENTITY="shin283-purdue-university"
export WANDB_RUN_NAME="${SLURM_JOB_ID}_upgd_outputonly_mini_imagenet_seed_2_lr_0.01_sigma_0.001_beta_0.9_wd_0.0"
export WANDB_MODE="online"

echo "========================================="
echo "Running UPGD Output-Only Gating (Mini-ImageNet)"
echo "Gating applied ONLY to linear_3 (output layer)"
echo "linear_1, linear_2 use fixed scaling at 0.5 (like SGD)"
echo "Start time: $(date)"
echo "========================================="

export CUDA_VISIBLE_DEVICES=0
python3 core/run/run_stats_with_curvature.py \
    --task label_permuted_mini_imagenet_stats \
    --learner upgd_fo_global_outputonly \
    --seed 2 \
    --lr 0.01 \
    --sigma 0.001 \
    --beta_utility 0.9 \
    --weight_decay 0.0 \
    --network fully_connected_relu_with_hooks \
    --n_samples 1000000 \
    --compute_curvature_every 1000000 \
    --save_path logs

echo "========================================="
echo "Output-only gating experiment completed"
echo "End time: $(date)"
echo "Hypothesis: If performance ≈ full UPGD, output layer is critical!"
echo "========================================"
