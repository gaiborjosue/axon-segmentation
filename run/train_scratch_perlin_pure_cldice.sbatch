#!/bin/bash
#SBATCH --job-name=train_pcld_scratch
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=50
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=edward.gaibor001@umb.edu
#SBATCH --output=/home/egaibor/orcd/scratch/experiment/logs_scratch_perlin_pure_cldice/train_unet3d_%j.out
#SBATCH --error=/home/egaibor/orcd/scratch/experiment/logs_scratch_perlin_pure_cldice/train_unet3d_%j.err

# ==============================================================================
# From-scratch higher_range training with Perlin-like noise + pure soft-clDice
#
# Scientific intent:
# - train from clean initialization (no init checkpoint)
# - keep the higher_range synthetic data recipe aligned with the baseline
# - add structured Perlin-like noise augmentation
# - replace DiceCE entirely with soft-clDice by setting --cldice_weight 1.0
# - run for 200 epochs as a controlled ablation
#
# Controlled differences vs train_production.sbatch:
#   --epochs 200
#   --perlin_noise_prob 0.30
#   --perlin_noise_amplitude_[0.02, 0.08]
#   --experimental_cldice --cldice_weight 1.0
#   no --init_checkpoint                  (clean weights)
#   no --hipct_structures flag            (use default off)
#
# Re-submit the same script if preempted: it will resume from this run's output dir.
# ==============================================================================

set -euo pipefail

echo "=== 3D UNet Scratch Training: Perlin + Pure clDice ==="
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
OUTPUT_DIR="/home/egaibor/orcd/scratch/experiment/training_out_higher_range_scratch_perlin_pure_cldice"
LOG_DIR="/home/egaibor/orcd/scratch/experiment/logs_scratch_perlin_pure_cldice"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1

RESUME_FLAG=""
if ls "${OUTPUT_DIR}/checkpoints/epoch_"*.pt 1>/dev/null 2>&1 || \
   [[ -f "${OUTPUT_DIR}/checkpoints/best_model.pt" ]]; then
    RESUME_FLAG="--resume"
    echo "Checkpoint found in scratch output dir — resuming run"
else
    echo "Starting fresh training from scratch"
fi

python train.py \
    --label_dir                     "${LABEL_DIR}" \
    --output_dir                    "${OUTPUT_DIR}" \
    --epochs                        200 \
    --batch_size                    2 \
    --lr                            1e-4 \
    --num_workers                   48 \
    --val_fraction                  0.2 \
    --val_interval                  1 \
    --samples_per_vol               2 \
    --cache_epochs                  3 \
    --max_volumes                   500 \
    --roi_size                      128 \
    --sw_batch_size                 4 \
    --seed                          42 \
    --subset_fraction_hi            0.9 \
    --perlin_noise_prob             0.30 \
    --perlin_noise_grid_min         4 \
    --perlin_noise_grid_max         16 \
    --perlin_noise_amplitude_min    0.02 \
    --perlin_noise_amplitude_max    0.08 \
    --perlin_noise_octaves          2 \
    --experimental_cldice \
    --cldice_weight                 1.0 \
    --cldice_iters                  10 \
    ${RESUME_FLAG}

echo ""
echo "=== Done! ==="
echo "End: $(date)"
echo "Output: ${OUTPUT_DIR}"