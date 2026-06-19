#!/bin/bash
#SBATCH --job-name=train_3class_human_ft_smoke
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=edward.gaibor001@umb.edu
#SBATCH --output=/home/egaibor/orcd/scratch/experiment/logs_three_class_human_like/train_3class_human_ft_smoke_%j.out
#SBATCH --error=/home/egaibor/orcd/scratch/experiment/logs_three_class_human_like/train_3class_human_ft_smoke_%j.err

set -euo pipefail

echo "=== Human-like 3-Class Fine-Tune Smoke ==="
echo "Start: $(date)"

module purge
module load deprecated-modules
module load StdEnv gcc/12.2.0 intel/2024.2.1 miniforge/24.3.0-0 cuda/12.4.0

# Conda activation scripts in this environment are not nounset-safe.
set +u
source activate synthsplinegpu
set -u

PROJECT_DIR="/home/egaibor/orcd/pool/projects/axon-segmentation/experiment"
LABEL_DIR="${LABEL_DIR:-/home/egaibor/orcd/scratch/experiment/dense_labels_vl/train}"
JOB_TAG="${SLURM_JOB_ID:-manual}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/egaibor/orcd/scratch/experiment/training_out_three_class_human_like_finetune_smoke_${JOB_TAG}}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/home/egaibor/orcd/scratch/experiment/training_out_higher_range_three_class_v2/checkpoints/best_model.pt}"
LOG_DIR="/home/egaibor/orcd/scratch/experiment/logs_three_class_human_like"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1

echo "Initializing from: ${INIT_CHECKPOINT}"
echo "Output: ${OUTPUT_DIR}"

python train.py \
    --label_dir                     "${LABEL_DIR}" \
    --output_dir                    "${OUTPUT_DIR}" \
    --init_checkpoint               "${INIT_CHECKPOINT}" \
    --segmentation_mode             three_class_shell_interior \
    --cache_builder                 gpu \
    --three_class_loss_mode         nested \
    --three_class_weights           1.5 1 1 \
    --nested_foreground_weight      2.0 \
    --nested_core_weight            1.0 \
    --three_class_selection_weights 0.75 0.125 0.125 \
    --epochs                        1 \
    --batch_size                    2 \
    --lr                            3e-5 \
    --num_workers                   12 \
    --val_fraction                  0.2 \
    --val_interval                  1 \
    --samples_per_vol               2 \
    --cache_epochs                  1 \
    --max_volumes                   24 \
    --roi_size                      128 \
    --sw_batch_size                 4 \
    --seed                          42 \
    --subset_fraction_lo            0.05 \
    --subset_fraction_hi            0.60 \
    --density_low_range             0.02 0.12 \
    --density_high_range            0.20 0.65 \
    --density_uniform_range         0.05 0.45 \
    --background                    0.15 \
    --fibers_lower_lo               0.55 \
    --fibers_lower_hi               0.75 \
    --bg_upper_lo                   0.05 \
    --bg_upper_hi                   0.18

echo ""
echo "=== Done! ==="
echo "End: $(date)"
echo "Output: ${OUTPUT_DIR}"
