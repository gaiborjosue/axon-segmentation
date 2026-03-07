#!/bin/bash
#SBATCH --job-name=train_unet3d_draft
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=edward.gaibor001@umb.edu
#SBATCH --output=/home/egaibor/orcd/scratch/experiment/logs/train_unet3d_draft_%j.out
#SBATCH --error=/home/egaibor/orcd/scratch/experiment/logs/train_unet3d_draft_%j.err

# ==============================================================================
# Draft training run — 3 label volumes, 20 epochs
# Cached training: rebuild synthetic train cache every 3 epochs; fixed cached val every epoch
#
# Purpose: verify DataLoader → UNet → loss decrease → val Dice > 0
#
# Label dir:   /scratch/experiment/draft/dense_labels/  (3 vols: 2 train, 1 val)
# Output dir:  /scratch/experiment/draft/training_out/
#   └── checkpoints/best_model.pt
#   └── tensorboard/
# ==============================================================================

echo "=== 3D UNet Draft Training ==="
echo "Start: $(date)"

module purge
module load StdEnv gcc/12.2.0 intel/2024.2.1 miniforge/24.3.0-0 cuda/12.4.0
source activate synthsplinegpu

PROJECT_DIR="/home/egaibor/orcd/pool/projects/axon-segmentation/experiment"
LABEL_DIR="/home/egaibor/orcd/scratch/experiment/draft/dense_labels"
OUTPUT_DIR="/home/egaibor/orcd/scratch/experiment/draft/training_out"

mkdir -p "${OUTPUT_DIR}" \
         "/home/egaibor/orcd/scratch/experiment/logs"

cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1

python train.py \
    --label_dir      "${LABEL_DIR}" \
    --output_dir     "${OUTPUT_DIR}" \
    --epochs         20 \
    --batch_size     2 \
    --lr             1e-4 \
    --num_workers    4 \
    --val_fraction   0.34 \
    --val_interval   1 \
    --samples_per_vol 30 \
    --cache_epochs   3 \
    --roi_size       128 \
    --sw_batch_size  4 \
    --seed           42

echo ""
echo "=== Done! ==="
echo "End: $(date)"
echo "Output: ${OUTPUT_DIR}"
