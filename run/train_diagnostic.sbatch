#!/bin/bash
#SBATCH --job-name=diag_unet3d
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=edward.gaibor001@umb.edu
#SBATCH --output=/home/egaibor/orcd/scratch/experiment/logs/diag_unet3d_%j.out
#SBATCH --error=/home/egaibor/orcd/scratch/experiment/logs/diag_unet3d_%j.err

# ==============================================================================
# DIAGNOSTIC run — 2 epochs with comprehensive per-sample logging.
#
# Purpose: measure exact time breakdown per sample/batch, count timeouts,
#          erode/shallow fallbacks, foreground filter redraws, etc.
#
# Output: full per-sample logs → analyze before committing to 100 epochs.
# ==============================================================================

echo "=== 3D UNet DIAGNOSTIC Run ==="
echo "Start: $(date)"
echo "CPUs: $(nproc)"

module purge
module load StdEnv gcc/12.2.0 intel/2024.2.1 miniforge/24.3.0-0 cuda/12.4.0
source activate synthsplinegpu

PROJECT_DIR="/home/egaibor/orcd/pool/projects/axon-segmentation/experiment"
LABEL_DIR="/home/egaibor/orcd/scratch/experiment/dense_labels/train"
OUTPUT_DIR="/home/egaibor/orcd/scratch/experiment/diag_out"

mkdir -p "${OUTPUT_DIR}" \
         "/home/egaibor/orcd/scratch/experiment/logs"

cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1

python train.py \
    --label_dir       "${LABEL_DIR}" \
    --output_dir      "${OUTPUT_DIR}" \
    --epochs          2 \
    --batch_size      2 \
    --lr              1e-4 \
    --num_workers     44 \
    --val_fraction    0.2 \
    --val_interval    2 \
    --samples_per_vol 5  \
    --max_volumes     400 \
    --roi_size        128 \
    --sw_batch_size   4 \
    --seed            42

echo ""
echo "=== Diagnostic Done! ==="
echo "End: $(date)"

# Quick summary stats from the log
echo ""
echo "=== Summary ==="
LOG_FILE="${OUTPUT_DIR}/../logs/diag_unet3d_${SLURM_JOB_ID}.out"
echo "Total timeouts:     $(grep -c 'synth timeout' "$LOG_FILE" 2>/dev/null || echo 0)"
echo "Raw fallbacks:      $(grep -c 'raw_fallback=True' "$LOG_FILE" 2>/dev/null || echo 0)"
echo "Erode fallbacks:    $(grep -c 'fallback=True' "$LOG_FILE" 2>/dev/null || echo 0)"
echo "FG redraws (>0):    $(grep -oP 'fg_redraws=\K[1-9]\d*' "$LOG_FILE" 2>/dev/null | wc -l || echo 0)"
