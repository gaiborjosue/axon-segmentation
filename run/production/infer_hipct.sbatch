#!/bin/bash
#SBATCH --job-name=infer_hipct
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=edward.gaibor001@umb.edu
#SBATCH --output=/home/egaibor/orcd/scratch/experiment/logs/infer_hipct_%j.out
#SBATCH --error=/home/egaibor/orcd/scratch/experiment/logs/infer_hipct_%j.err

# ==============================================================================
# HiP-CT Axon Segmentation Inference
#
# Input:  Downloaded .npy patch from LINCBrain dandiset 000005
#         sub-I74_sample-IC_acq-zoom_chunk-01_XPCT.ome.zarr
#         Bounding box: (1461,210,3295) → (3323,1180,4889)
#         Patch size: 1862 × 970 × 1594, uint16, 0.857 µm isotropic
#
# Override settings at submit time if needed, for example:
#   sbatch --export=ALL,CHECKPOINT=/path/to/best_model.pt run/production/infer_hipct.sbatch
#   sbatch --export=ALL,CHECKPOINT=/path/to/3class.pt,SEGMENTATION_MODE=auto run/production/infer_hipct.sbatch
#   sbatch --export=ALL,NORM_MODE=clahe,OUTPUT_ROOT=/path/out run/production/infer_hipct.sbatch
#
# ==============================================================================

set -euo pipefail

echo "=== HiP-CT Axon Segmentation Inference ==="
echo "Start: $(date)"

module purge
module load deprecated-modules
module load StdEnv gcc/12.2.0 intel/2024.2.1 miniforge/24.3.0-0 cuda/12.4.0

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

# norm_mode options: percentile (original), mean_shift, clahe
NORM_MODE="${NORM_MODE:-mean_shift}"

SEGMENTATION_MODE="${SEGMENTATION_MODE:-auto}"
VOXEL_SIZE="${VOXEL_SIZE:-0.857}"
ROI_SIZE="${ROI_SIZE:-128}"
SW_BATCH_SIZE="${SW_BATCH_SIZE:-4}"
OVERLAP="${OVERLAP:-0.5}"
THRESHOLD="${THRESHOLD:-0.5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/egaibor/orcd/scratch/experiment/hipct/inference}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${PATCH_NAME}/${MODEL_NAME}_${NORM_MODE}}"

mkdir -p "${OUTPUT_DIR}" \
         "/home/egaibor/orcd/scratch/experiment/logs"

if [[ ! -f "${PATCH}" ]]; then
    echo "Missing patch: ${PATCH}" >&2
    exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Missing checkpoint: ${CHECKPOINT}" >&2
    exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing python interpreter: ${PYTHON_BIN}" >&2
    exit 1
fi

echo "Python:            ${PYTHON_BIN}"
echo "Patch:             ${PATCH}"
echo "Checkpoint:        ${CHECKPOINT}"
echo "Segmentation mode: ${SEGMENTATION_MODE}"
echo "Output dir:        ${OUTPUT_DIR}"
echo "Norm mode:         ${NORM_MODE}"
echo "Voxel size:        ${VOXEL_SIZE} um"
echo "ROI size:          ${ROI_SIZE}"
echo "SW batch:          ${SW_BATCH_SIZE}"
echo "Overlap:           ${OVERLAP}"
echo "Threshold:         ${THRESHOLD}"

cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1

if command -v timeout >/dev/null 2>&1; then
    echo "Running import preflight ..."
    timeout 60s "${PYTHON_BIN}" -c "import inference.infer_hipct"
else
    echo "Running import preflight without timeout ..."
    "${PYTHON_BIN}" -c "import inference.infer_hipct"
fi

"${PYTHON_BIN}" inference/infer_hipct.py \
    --input         "${PATCH}" \
    --checkpoint    "${CHECKPOINT}" \
    --output_dir    "${OUTPUT_DIR}" \
    --segmentation_mode "${SEGMENTATION_MODE}" \
    --voxel_size    "${VOXEL_SIZE}" \
    --roi_size      "${ROI_SIZE}" \
    --sw_batch_size "${SW_BATCH_SIZE}" \
    --overlap       "${OVERLAP}" \
    --threshold     "${THRESHOLD}" \
    --norm_mode     "${NORM_MODE}"

echo ""
echo "=== Done! ==="
echo "End: $(date)"
echo "Outputs: ${OUTPUT_DIR}"
