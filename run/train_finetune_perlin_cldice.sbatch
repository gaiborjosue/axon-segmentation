#!/bin/bash
#SBATCH --job-name=train_ft_pcld
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=50
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=edward.gaibor001@umb.edu
#SBATCH --output=/home/egaibor/orcd/scratch/experiment/logs_finetune_perlin_cldice/train_unet3d_%j.out
#SBATCH --error=/home/egaibor/orcd/scratch/experiment/logs_finetune_perlin_cldice/train_unet3d_%j.err

# ==============================================================================
# Fine-tune higher_range baseline with Perlin-like noise + experimental clDice
#
# Scientific intent:
# - start from the best epoch-200 higher_range checkpoint
# - keep the synthetic data recipe aligned with the baseline
# - add only moderate structured-noise augmentation and a low-weight clDice term
# - run as a fresh experiment (new optimizer/scheduler state), but resume safely
#   if this fine-tune itself is preempted
#
# Key differences vs train_production.sbatch:
#   --init_checkpoint best_model_ep200.pt   (fresh fine-tune, not resume of baseline)
#   --epochs 60                            (short first-pass adaptation run)
#   --lr 3e-5                              (conservative fine-tune learning rate)
#   --perlin_noise_prob 0.30
#   --perlin_noise_amplitude_[0.02, 0.08]
#   --experimental_cldice --cldice_weight 0.2
#   no --hipct_structures flag             (use default off)
#
# Re-submit the same script if preempted: it will resume from this run's output dir.
# ==============================================================================

set -euo pipefail

echo "=== 3D UNet Fine-Tune: Perlin + clDice ==="
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
BASELINE_CKPT="/home/egaibor/orcd/scratch/experiment/training_out_higher_range/checkpoints/best_model_ep200.pt"
OUTPUT_DIR="/home/egaibor/orcd/scratch/experiment/training_out_higher_range_finetune_perlin_cldice02"
LOG_DIR="/home/egaibor/orcd/scratch/experiment/logs_finetune_perlin_cldice"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

if [[ ! -f "${BASELINE_CKPT}" ]]; then
    echo "Missing baseline checkpoint: ${BASELINE_CKPT}" >&2
    exit 1
fi

cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1

EXTRA_INIT_ARGS=()
if ls "${OUTPUT_DIR}/checkpoints/epoch_"*.pt 1>/dev/null 2>&1 || \
   [[ -f "${OUTPUT_DIR}/checkpoints/best_model.pt" ]]; then
    EXTRA_INIT_ARGS+=(--resume)
    echo "Checkpoint found in fine-tune output dir — resuming fine-tune run"
else
    EXTRA_INIT_ARGS+=(--init_checkpoint "${BASELINE_CKPT}")
    echo "Starting fresh fine-tune from baseline checkpoint"
fi

python train.py \
    --label_dir                     "${LABEL_DIR}" \
    --output_dir                    "${OUTPUT_DIR}" \
    --epochs                        60 \
    --batch_size                    2 \
    --lr                            3e-5 \
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
    --cldice_weight                 0.2 \
    --cldice_iters                  10 \
    "${EXTRA_INIT_ARGS[@]}"

echo ""
echo "=== Done! ==="
echo "End: $(date)"
echo "Output: ${OUTPUT_DIR}"