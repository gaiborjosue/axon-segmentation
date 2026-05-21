#!/bin/bash
#SBATCH --job-name=infer_lsm_eval
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=edward.gaibor001@umb.edu
#SBATCH --output=/home/egaibor/orcd/scratch/experiment/logs/infer_lsm_eval_%j.out
#SBATCH --error=/home/egaibor/orcd/scratch/experiment/logs/infer_lsm_eval_%j.err

set -euo pipefail

echo "=== LSM Inference + Eval ==="
echo "Start: $(date)"

module purge
module load deprecated-modules
module load StdEnv gcc/12.2.0 intel/2024.2.1 miniforge/24.3.0-0 cuda/12.4.0

set +u
source activate synthsplinegpu
set -u

PROJECT_DIR="/home/egaibor/orcd/pool/projects/axon-segmentation/experiment"
WEBK_DIR="/home/egaibor/orcd/scratch/experiment/webknossos"

PATCH_NAME="${PATCH_NAME:-macaque_NEFH_WM}"
PATCH="${PATCH:-${WEBK_DIR}/${PATCH_NAME}.npy}"
TARGET="${TARGET:-${WEBK_DIR}/${PATCH_NAME}_binary.npy}"
VALID_MASK="${VALID_MASK:-${WEBK_DIR}/${PATCH_NAME}_valid_mask.npy}"
CHECKPOINT="${CHECKPOINT:-/home/egaibor/orcd/scratch/experiment/training_out_higher_range_three_class_v2/checkpoints/best_model.pt}"
MODEL_NAME="${MODEL_NAME:-three_class_v2_best}"
NORM_MODE="${NORM_MODE:-percentile}"
VOXEL_SIZE="${VOXEL_SIZE:-1.0}"
ROI_SIZE="${ROI_SIZE:-128}"
SW_BATCH_SIZE="${SW_BATCH_SIZE:-4}"
OVERLAP="${OVERLAP:-0.5}"
INFER_THRESHOLD="${INFER_THRESHOLD:-0.5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/egaibor/orcd/scratch/experiment/webknossos/inference_3class}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-${PATCH_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${PATCH_NAME}/${MODEL_NAME}_${NORM_MODE}}"
SKIP_INFERENCE="${SKIP_INFERENCE:-0}"

SWEEP_START="${SWEEP_START:-0.05}"
SWEEP_STOP="${SWEEP_STOP:-0.95}"
SWEEP_STEP="${SWEEP_STEP:-0.01}"
CORRECT_NEIGHBOR_ROUNDS="${CORRECT_NEIGHBOR_ROUNDS:-2}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda}"

mkdir -p "${OUTPUT_DIR}" \
         "/home/egaibor/orcd/scratch/experiment/logs"

cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1

echo "Patch:         ${PATCH}"
echo "Target:        ${TARGET}"
echo "Valid mask:    ${VALID_MASK}"
echo "Checkpoint:    ${CHECKPOINT}"
echo "Output dir:    ${OUTPUT_DIR}"
echo "Skip inference:${SKIP_INFERENCE}"
echo "Sweep:         ${SWEEP_START}..${SWEEP_STOP} step ${SWEEP_STEP}"
echo "Eval device:   ${EVAL_DEVICE}"

if [[ ! -f "${PATCH}" ]]; then
    echo "Missing patch: ${PATCH}" >&2
    exit 1
fi
if [[ ! -f "${TARGET}" ]]; then
    echo "Missing target: ${TARGET}" >&2
    exit 1
fi
if [[ ! -f "${VALID_MASK}" ]]; then
    echo "Missing valid mask: ${VALID_MASK}" >&2
    exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Missing checkpoint: ${CHECKPOINT}" >&2
    exit 1
fi

if [[ "${SKIP_INFERENCE}" != "1" ]]; then
    python inference/infer_lsm.py \
        --input         "${PATCH}" \
        --checkpoint    "${CHECKPOINT}" \
        --output_dir    "${OUTPUT_DIR}" \
        --output_prefix "${OUTPUT_PREFIX}" \
        --voxel_size    "${VOXEL_SIZE}" \
        --roi_size      "${ROI_SIZE}" \
        --sw_batch_size "${SW_BATCH_SIZE}" \
        --overlap       "${OVERLAP}" \
        --threshold     "${INFER_THRESHOLD}" \
        --norm_mode     "${NORM_MODE}"
else
    echo "Skipping inference and reusing existing outputs"
fi

SWEEP_JSON="${OUTPUT_DIR}/${OUTPUT_PREFIX}_eval_sweep.json"
python inference/utils/evaluate_masked_prediction.py \
    --prediction              "${OUTPUT_DIR}/${OUTPUT_PREFIX}_pred_prob.nii.gz" \
    --target                  "${TARGET}" \
    --valid-mask              "${VALID_MASK}" \
    --sweep-start             "${SWEEP_START}" \
    --sweep-stop              "${SWEEP_STOP}" \
    --sweep-step              "${SWEEP_STEP}" \
    --correct-neighbors \
    --correct-neighbor-rounds "${CORRECT_NEIGHBOR_ROUNDS}" \
    --device                  "${EVAL_DEVICE}" \
    --output-json             "${SWEEP_JSON}"

BEST_JSON=$(python inference/utils/materialize_best_threshold_prediction.py \
    --probability   "${OUTPUT_DIR}/${OUTPUT_PREFIX}_pred_prob.nii.gz" \
    --sweep-json    "${SWEEP_JSON}" \
    --output-dir    "${OUTPUT_DIR}" \
    --output-prefix "${OUTPUT_PREFIX}")

echo "${BEST_JSON}"
BEST_PRED=$(printf '%s\n' "${BEST_JSON}" | python -c 'import json,sys; print(json.load(sys.stdin)["output_path"])')

python inference/utils/evaluate_masked_prediction.py \
    --prediction              "${BEST_PRED}" \
    --target                  "${TARGET}" \
    --valid-mask              "${VALID_MASK}" \
    --correct-neighbors \
    --correct-neighbor-rounds "${CORRECT_NEIGHBOR_ROUNDS}" \
    --device                  "${EVAL_DEVICE}" \
    --output-json             "${OUTPUT_DIR}/${OUTPUT_PREFIX}_eval_corrected.json"

echo ""
echo "=== Done! ==="
echo "End: $(date)"
echo "Outputs: ${OUTPUT_DIR}"