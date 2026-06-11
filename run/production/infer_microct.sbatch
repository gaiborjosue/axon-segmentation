#!/bin/bash
#SBATCH --job-name=infer_microct
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=edward.gaibor001@umb.edu
#SBATCH --output=/home/egaibor/orcd/scratch/experiment/logs/infer_microct_%j.out
#SBATCH --error=/home/egaibor/orcd/scratch/experiment/logs/infer_microct_%j.err

# ==============================================================================
# microCT Axon Segmentation Inference
#
# Default target: DANDI asset
#   sourcedata/raw/sub-Hb1/micr/
#   sub-Hb1_sample-lefthemi+ICblock+punch1+piece1+trim9_uCT.ome.zarr
#
# Downloaded patch bbox (x,y,z): (269,246,809) -> (1486,1282,1147)
# Saved patch size: 1217 x 1036 x 338, uint8
# Native voxel size: 0.3644713433459401 um isotropic
#
# Override settings at submit time if needed, for example:
#   sbatch --export=ALL,NORM_MODE=clahe run/production/infer_microct.sbatch
#   sbatch --export=ALL,INVERT_CONTRAST=0 run/production/infer_microct.sbatch
#   sbatch --export=ALL,CHECKPOINT=/path/to/3class.pt,SEGMENTATION_MODE=auto run/production/infer_microct.sbatch
#   sbatch --export=ALL,PATCH=/path/to/other.raw,OUTPUT_ROOT=/path/out run/production/infer_microct.sbatch
# ==============================================================================

set -euo pipefail

echo "=== microCT Axon Segmentation Inference ==="
echo "Start: $(date)"

module purge
module load deprecated-modules
module load StdEnv gcc/12.2.0 intel/2024.2.1 miniforge/24.3.0-0 cuda/12.4.0

PROJECT_DIR="/home/egaibor/orcd/pool/projects/axon-segmentation/experiment"
PYTHON_BIN="${PYTHON_BIN:-/home/egaibor/orcd/scratch/experiment/envs/synthsplinegpufix/bin/python}"
PATCH="${PATCH:-/home/egaibor/orcd/scratch/experiment/microct/sub-Hb1_sample-lefthemi+ICblock+punch1+piece1+trim9_uCT_patch.raw}"
CHECKPOINT="${CHECKPOINT:-/home/egaibor/orcd/scratch/experiment/training_out_higher_range/checkpoints/best_model_ep200.pt}"
PATCH_BASENAME="$(basename "${PATCH}")"
PATCH_STEM="${PATCH_BASENAME%.raw}"
PATCH_STEM="${PATCH_STEM%.npy}"
PATCH_NAME="${PATCH_NAME:-${PATCH_STEM}}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-${PATCH_NAME}}"
MODEL_NAME="${MODEL_NAME:-higher_range_ep200}"

SEGMENTATION_MODE="${SEGMENTATION_MODE:-auto}"
VOXEL_SIZE="${VOXEL_SIZE:-0.3644713433459401}"
NORM_MODE="${NORM_MODE:-percentile}"
INVERT_CONTRAST="${INVERT_CONTRAST:-1}"
ROI_SIZE="${ROI_SIZE:-128}"
SW_BATCH_SIZE="${SW_BATCH_SIZE:-4}"
OVERLAP="${OVERLAP:-0.5}"
THRESHOLD="${THRESHOLD:-0.5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/egaibor/orcd/scratch/experiment/microct/inference}"
CONTRAST_TAG=""
if [[ "${INVERT_CONTRAST}" == "1" || "${INVERT_CONTRAST}" == "true" || "${INVERT_CONTRAST}" == "TRUE" ]]; then
    CONTRAST_TAG="_inverted"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${PATCH_NAME}/${MODEL_NAME}_${NORM_MODE}${CONTRAST_TAG}}"

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
echo "Output prefix:     ${OUTPUT_PREFIX}"
echo "Norm mode:         ${NORM_MODE}"
echo "Invert contrast:   ${INVERT_CONTRAST}"
echo "Voxel size:        ${VOXEL_SIZE} um"
echo "ROI size:          ${ROI_SIZE}"
echo "SW batch:          ${SW_BATCH_SIZE}"
echo "Overlap:           ${OVERLAP}"
echo "Threshold:         ${THRESHOLD}"

cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1

if command -v timeout >/dev/null 2>&1; then
    echo "Running import preflight ..."
    timeout 60s "${PYTHON_BIN}" -c "import inference.infer_lsm"
else
    echo "Running import preflight without timeout ..."
    "${PYTHON_BIN}" -c "import inference.infer_lsm"
fi

INVERT_CONTRAST_ARG=()
if [[ -n "${CONTRAST_TAG}" ]]; then
    INVERT_CONTRAST_ARG=(--invert_contrast)
fi

"${PYTHON_BIN}" inference/infer_lsm.py \
    --input             "${PATCH}" \
    --checkpoint        "${CHECKPOINT}" \
    --output_dir        "${OUTPUT_DIR}" \
    --output_prefix     "${OUTPUT_PREFIX}" \
    --segmentation_mode "${SEGMENTATION_MODE}" \
    --voxel_size        "${VOXEL_SIZE}" \
    --roi_size          "${ROI_SIZE}" \
    --sw_batch_size     "${SW_BATCH_SIZE}" \
    --overlap           "${OVERLAP}" \
    --threshold         "${THRESHOLD}" \
    --norm_mode         "${NORM_MODE}" \
    "${INVERT_CONTRAST_ARG[@]}"

echo ""
echo "=== Done! ==="
echo "End: $(date)"
echo "Outputs: ${OUTPUT_DIR}"
