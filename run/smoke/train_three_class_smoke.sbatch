#!/bin/bash
#SBATCH --job-name=train_unet3d_3class_smoke
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=edward.gaibor001@umb.edu
#SBATCH --output=/home/egaibor/orcd/scratch/experiment/logs_higher_range_three_class/train_unet3d_3class_smoke_%j.out
#SBATCH --error=/home/egaibor/orcd/scratch/experiment/logs_higher_range_three_class/train_unet3d_3class_smoke_%j.err

set -euo pipefail

echo "=== 3-Class UNet Smoke Training ==="
echo "Start: $(date)"

module purge
module load deprecated-modules
module load StdEnv gcc/12.2.0 intel/2024.2.1 miniforge/24.3.0-0 cuda/12.4.0

# Conda activation scripts in this environment are not nounset-safe.
set +u
source activate synthsplinegpu
set -u

PROJECT_DIR="/home/egaibor/orcd/pool/projects/axon-segmentation/experiment"
LABEL_DIR="/home/egaibor/orcd/scratch/experiment/dense_labels_vl/train"
OUTPUT_DIR="/home/egaibor/orcd/scratch/experiment/training_out_higher_range_three_class_smoke_v6_gpu_cache_1epoch"

mkdir -p "${OUTPUT_DIR}" \
         "/home/egaibor/orcd/scratch/experiment/logs_higher_range_three_class"

cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1

python train.py \
    --label_dir            "${LABEL_DIR}" \
    --output_dir           "${OUTPUT_DIR}" \
    --segmentation_mode    three_class_shell_interior \
    --cache_builder        gpu \
    --three_class_loss_mode nested \
    --three_class_weights  1 1 1 \
    --epochs               1 \
    --batch_size           2 \
    --lr                   1e-4 \
    --num_workers          12 \
    --val_fraction         0.2 \
    --val_interval         1 \
    --samples_per_vol      2 \
    --cache_epochs         1 \
    --max_volumes          24 \
    --roi_size             128 \
    --sw_batch_size        4 \
    --seed                 42 \
    --subset_fraction_hi   0.9

echo ""
echo "=== Done! ==="
echo "End: $(date)"
echo "Output: ${OUTPUT_DIR}"