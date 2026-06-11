#!/bin/bash
#SBATCH --job-name=save_hipct_crops
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=edward.gaibor001@umb.edu
#SBATCH --output=/home/egaibor/orcd/scratch/experiment/logs/save_hipct_crops_%j.out
#SBATCH --error=/home/egaibor/orcd/scratch/experiment/logs/save_hipct_crops_%j.err

# ==============================================================================
# Export review crops from an existing HiP-CT inference directory.
#
# Optional overrides when submitting:
#   sbatch --export=ALL,INFERENCE_DIR=/path/to/inference_out,PATCH=/path/to/patch.raw run/production/save_hipct_crops.sbatch
#   sbatch --export=ALL,CHECKPOINT=/path/to/3class.pt,SEGMENTATION_MODE=auto,N_CROPS=8 run/production/save_hipct_crops.sbatch
# ==============================================================================

set -euo pipefail

echo "=== HiP-CT Crop Export ==="
echo "Start: $(date)"

module purge
module load deprecated-modules
module load StdEnv gcc/12.2.0 intel/2024.2.1 miniforge/24.3.0-0

PROJECT_DIR="/home/egaibor/orcd/pool/projects/axon-segmentation/experiment"
PYTHON_BIN="${PYTHON_BIN:-/home/egaibor/orcd/scratch/experiment/envs/synthsplinegpufix/bin/python}"
PATCH="${PATCH:-/home/egaibor/orcd/scratch/experiment/hipct/patch_I74_IC_zoom01.raw}"
CHECKPOINT="${CHECKPOINT:-/home/egaibor/orcd/scratch/experiment/training_out/checkpoints/best_model.pt}"
PATCH_BASENAME="$(basename "${PATCH}")"
PATCH_STEM="${PATCH_BASENAME%.raw}"
PATCH_STEM="${PATCH_STEM%.npy}"
PATCH_NAME="${PATCH_NAME:-${PATCH_STEM}}"
DEFAULT_MODEL_NAME="$(basename "$(dirname "$(dirname "${CHECKPOINT}")")")"
MODEL_NAME="${MODEL_NAME:-${DEFAULT_MODEL_NAME}}"
NORM_MODE="${NORM_MODE:-mean_shift}"
SEGMENTATION_MODE="${SEGMENTATION_MODE:-auto}"
VOXEL_SIZE="${VOXEL_SIZE:-0.857}"
N_CROPS="${N_CROPS:-5}"
CROP_SIZE="${CROP_SIZE:-128}"
CROP_SEED="${CROP_SEED:-42}"
MIN_POSITIVE_FRACTION="${MIN_POSITIVE_FRACTION:-0.05}"
INFERENCE_ROOT="${INFERENCE_ROOT:-/home/egaibor/orcd/scratch/experiment/hipct/inference}"
INFERENCE_DIR="${INFERENCE_DIR:-${INFERENCE_ROOT}/${PATCH_NAME}/${MODEL_NAME}_${NORM_MODE}}"

mkdir -p "/home/egaibor/orcd/scratch/experiment/logs"

if [[ ! -f "${PATCH}" ]]; then
    echo "Missing patch: ${PATCH}" >&2
    exit 1
fi

if [[ ! -d "${INFERENCE_DIR}" ]]; then
    echo "Missing inference directory: ${INFERENCE_DIR}" >&2
    exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing python interpreter: ${PYTHON_BIN}" >&2
    exit 1
fi

echo "Python:                ${PYTHON_BIN}"
echo "Patch:                 ${PATCH}"
echo "Inference dir:         ${INFERENCE_DIR}"
echo "Segmentation mode:     ${SEGMENTATION_MODE}"
echo "Crop count:            ${N_CROPS}"
echo "Crop size:             ${CROP_SIZE}"
echo "Crop seed:             ${CROP_SEED}"
echo "Min positive fraction: ${MIN_POSITIVE_FRACTION}"
echo "Voxel size:            ${VOXEL_SIZE} um"

cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1

if command -v timeout >/dev/null 2>&1; then
    echo "Running import preflight ..."
    timeout 60s "${PYTHON_BIN}" -c "import inference.utils.save_crops"
else
    echo "Running import preflight without timeout ..."
    "${PYTHON_BIN}" -c "import inference.utils.save_crops"
fi

cmd=(
    "${PYTHON_BIN}" inference/utils/save_crops.py
    --inference_dir "${INFERENCE_DIR}"
    --patch_raw "${PATCH}"
    --n_crops "${N_CROPS}"
    --crop_size "${CROP_SIZE}"
    --seed "${CROP_SEED}"
    --segmentation_mode "${SEGMENTATION_MODE}"
    --min_positive_fraction "${MIN_POSITIVE_FRACTION}"
    --voxel_size "${VOXEL_SIZE}"
)

echo "Command: ${cmd[*]}"
"${cmd[@]}"

echo ""
echo "=== Done! ==="
echo "End: $(date)"
echo "Outputs: ${INFERENCE_DIR}/crops"