#!/bin/bash
#SBATCH --job-name=infer_lsm
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=edward.gaibor001@umb.edu
#SBATCH --output=/home/egaibor/orcd/scratch/experiment/logs/infer_lsm_%j.out
#SBATCH --error=/home/egaibor/orcd/scratch/experiment/logs/infer_lsm_%j.err

# ==============================================================================
# LSM Axon Segmentation Inference
#
# Default target: macaque NEFH white-matter patch downloaded from WebKnossos
# annotation 696148730100001001be9620.
#
# Requested box: (-2, 919, 2645) size (114, 250, 250)
# Saved patch preserves that full shape with zero padding on the x margin.
#
# Model default: higher_range epoch-200 checkpoint.
#
# Override settings at submit time if needed, for example:
#   sbatch --export=ALL,NORM_MODE=clahe run/infer_lsm.sbatch
#   sbatch --export=ALL,PATCH=/path/to/other.npy,OUTPUT_PREFIX=other run/infer_lsm.sbatch
# ==============================================================================

set -euo pipefail

echo "=== LSM Axon Segmentation Inference ==="
echo "Start: $(date)"

module purge
module load deprecated-modules
module load StdEnv gcc/12.2.0 intel/2024.2.1 miniforge/24.3.0-0 cuda/12.4.0

# Conda activation scripts in this environment are not nounset-safe.
set +u
source activate synthsplinegpu
set -u

PROJECT_DIR="/home/egaibor/orcd/pool/projects/axon-segmentation/experiment"
PATCH="${PATCH:-/home/egaibor/orcd/scratch/experiment/webknossos/macaque_NEFH_WM.npy}"
CHECKPOINT="${CHECKPOINT:-/home/egaibor/orcd/scratch/experiment/training_out_higher_range/checkpoints/best_model_ep200.pt}"
PATCH_NAME="${PATCH_NAME:-macaque_NEFH_WM}"
MODEL_NAME="${MODEL_NAME:-higher_range_ep200}"
NORM_MODE="${NORM_MODE:-percentile}"

# This only affects the NIfTI affine used for visualization. Update if a trusted
# physical voxel size for this LSM dataset becomes available.
VOXEL_SIZE="${VOXEL_SIZE:-1.0}"

ROI_SIZE="${ROI_SIZE:-128}"
SW_BATCH_SIZE="${SW_BATCH_SIZE:-4}"
OVERLAP="${OVERLAP:-0.5}"
THRESHOLD="${THRESHOLD:-0.5}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-${PATCH_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/egaibor/orcd/scratch/experiment/webknossos/inference}"
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

echo "Patch:         ${PATCH}"
echo "Checkpoint:    ${CHECKPOINT}"
echo "Output dir:    ${OUTPUT_DIR}"
echo "Output prefix: ${OUTPUT_PREFIX}"
echo "Norm mode:     ${NORM_MODE}"
echo "Voxel size:    ${VOXEL_SIZE} um"
echo "ROI size:      ${ROI_SIZE}"
echo "SW batch:      ${SW_BATCH_SIZE}"
echo "Overlap:       ${OVERLAP}"
echo "Threshold:     ${THRESHOLD}"

cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1

python inference/infer_lsm.py \
    --input         "${PATCH}" \
    --checkpoint    "${CHECKPOINT}" \
    --output_dir    "${OUTPUT_DIR}" \
    --output_prefix "${OUTPUT_PREFIX}" \
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