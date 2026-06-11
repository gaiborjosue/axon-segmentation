#!/bin/bash
#SBATCH --job-name=train_unet3d
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=50
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=edward.gaibor001@umb.edu
#SBATCH --output=/home/egaibor/orcd/scratch/experiment/logs_higher_range/train_unet3d_%j.out
#SBATCH --error=/home/egaibor/orcd/scratch/experiment/logs_higher_range/train_unet3d_%j.err

# ==============================================================================
# Production training run — higher subset_fraction range (0.3, 0.9)
# Cached training: rebuild synthetic train cache every 3 epochs; fixed cached val every epoch
#
# Label dir:  /scratch/experiment/dense_labels_vl/train/  (0.8 µm voxel)
# --max_volumes 400 → 320 train + 80 val
# Checkpoints:
#   best_model_ep200.pt  — best Dice up to epoch 200
#   best_model.pt        — best Dice over full 500 epochs
# Key change vs training_out_larger_voxel: subset_fraction_hi raised 0.8 → 0.9
#   so model sees densely-packed fascicles during training
# Partition:  mit_preemptable (2-day limit, preemption-safe via SIGTERM checkpoint)
# Resume:     sbatch run/production/train_production.sbatch  (--resume auto-detects latest ckpt)
# ==============================================================================

echo "=== 3D UNet Production Training ==="
echo "Start: $(date)"

module purge
module load StdEnv gcc/12.2.0 intel/2024.2.1 miniforge/24.3.0-0 cuda/12.4.0
source activate synthsplinegpu

PROJECT_DIR="/home/egaibor/orcd/pool/projects/axon-segmentation/experiment"
LABEL_DIR="/home/egaibor/orcd/scratch/experiment/dense_labels_vl/train"
OUTPUT_DIR="/home/egaibor/orcd/scratch/experiment/training_out_higher_range"

mkdir -p "${OUTPUT_DIR}" \
         "/home/egaibor/orcd/scratch/experiment/logs_higher_range"

cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1

# Check if checkpoint exists and add --resume flag
RESUME_FLAG=""
if ls "${OUTPUT_DIR}/checkpoints/epoch_"*.pt 1>/dev/null 2>&1 || \
   [ -f "${OUTPUT_DIR}/checkpoints/best_model.pt" ]; then
    RESUME_FLAG="--resume"
    echo "Checkpoint found — resuming training"
fi

python train.py \
    --label_dir       "${LABEL_DIR}" \
    --output_dir      "${OUTPUT_DIR}" \
    --epochs          500 \
    --batch_size      2 \
    --lr              1e-4 \
    --num_workers     48 \
    --val_fraction    0.2 \
    --val_interval    1 \
    --samples_per_vol 2  \
    --cache_epochs    3 \
    --max_volumes     500 \
    --roi_size        128 \
    --sw_batch_size   4 \
    --seed            42 \
    --subset_fraction_hi 0.9 \
    $RESUME_FLAG

echo ""
echo "=== Done! ==="
echo "End: $(date)"
echo "Output: ${OUTPUT_DIR}"
