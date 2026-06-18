#!/bin/bash
#SBATCH --job-name=infer_lsm_annotated
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=edward.gaibor001@umb.edu
#SBATCH --output=/home/egaibor/orcd/scratch/experiment/logs/infer_lsm_annotated_%j.out
#SBATCH --error=/home/egaibor/orcd/scratch/experiment/logs/infer_lsm_annotated_%j.err

# ==============================================================================
# Inference + complete evaluation for one annotated LSM NIfTI patch.
#
# Required override:
#   PATCH=/path/to/<patch>_raw.nii.gz
#
# Optional common overrides:
#   MODEL_KIND=binary|threeclass
#   TARGET=/path/to/<patch>_gt.nii.gz
#   OUTPUT_DIR=/path/to/output
#   SKIP_INFERENCE=1
# ==============================================================================

set -euo pipefail

echo "=== Annotated LSM Patch Inference + Eval ==="
echo "Start: $(date)"

module purge
module load deprecated-modules
module load StdEnv gcc/12.2.0 intel/2024.2.1 miniforge/24.3.0-0 cuda/12.4.0

PROJECT_DIR="/home/egaibor/orcd/pool/projects/axon-segmentation/experiment"
PYTHON_BIN="${PYTHON_BIN:-/home/egaibor/orcd/scratch/experiment/envs/synthsplinegpufix/bin/python}"
PATCH="${PATCH:?Set PATCH=/path/to/<patch>_raw.nii.gz}"
PATCH_DIR="$(dirname "${PATCH}")"
PATCH_BASENAME="$(basename "${PATCH}")"
PATCH_STEM="${PATCH_BASENAME%.nii.gz}"
PATCH_STEM="${PATCH_STEM%.nii}"
PATCH_NAME="${PATCH_NAME:-${PATCH_STEM%_raw}}"
TARGET="${TARGET:-${PATCH_DIR}/${PATCH_NAME}_gt.nii.gz}"
VALID_MASK="${VALID_MASK:-}"

MODEL_KIND="${MODEL_KIND:-binary}"
if [[ "${MODEL_KIND}" == "binary" ]]; then
    CHECKPOINT="${CHECKPOINT:-/home/egaibor/orcd/scratch/experiment/training_out_higher_range/checkpoints/best_model_ep200.pt}"
elif [[ "${MODEL_KIND}" == "threeclass" ]]; then
    CHECKPOINT="${CHECKPOINT:-/home/egaibor/orcd/scratch/experiment/training_out_higher_range_three_class_v2/checkpoints/best_model.pt}"
else
    echo "MODEL_KIND must be 'binary' or 'threeclass', got: ${MODEL_KIND}" >&2
    exit 1
fi

SEGMENTATION_MODE="${SEGMENTATION_MODE:-auto}"
NORM_MODE="${NORM_MODE:-percentile}"
VOXEL_SIZE="${VOXEL_SIZE:-1.0}"
ROI_SIZE="${ROI_SIZE:-128}"
SW_BATCH_SIZE="${SW_BATCH_SIZE:-4}"
OVERLAP="${OVERLAP:-0.5}"
INFER_THRESHOLD="${INFER_THRESHOLD:-0.5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PATCH_DIR}/inference}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-${PATCH_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${PATCH_NAME}/${MODEL_KIND}}"
SKIP_INFERENCE="${SKIP_INFERENCE:-0}"

EVAL_TAG="${EVAL_TAG:-eval_complete}"
BEST_THRESHOLD_CRITERION="${BEST_THRESHOLD_CRITERION:-corrected_dice}"
SWEEP_START="${SWEEP_START:-0.05}"
SWEEP_STOP="${SWEEP_STOP:-0.95}"
SWEEP_STEP="${SWEEP_STEP:-0.01}"
CORRECT_NEIGHBOR_ROUNDS="${CORRECT_NEIGHBOR_ROUNDS:-2}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda}"
TOPOLOGY_METRICS="${TOPOLOGY_METRICS:-1}"
TOPOLOGY_ON_SWEEP="${TOPOLOGY_ON_SWEEP:-0}"
TOPOLOGY_CONNECTIVITY="${TOPOLOGY_CONNECTIVITY:-6}"
CORRECTED_TOPOLOGY_METRICS="${CORRECTED_TOPOLOGY_METRICS:-0}"

mkdir -p "${OUTPUT_DIR}" \
         "/home/egaibor/orcd/scratch/experiment/logs"

if [[ ! -f "${PATCH}" ]]; then
    echo "Missing patch: ${PATCH}" >&2
    exit 1
fi
if [[ ! -f "${TARGET}" ]]; then
    echo "Missing target: ${TARGET}" >&2
    exit 1
fi
if [[ -n "${VALID_MASK}" && ! -f "${VALID_MASK}" ]]; then
    echo "Missing valid mask: ${VALID_MASK}" >&2
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

echo "Python:        ${PYTHON_BIN}"
echo "Patch:         ${PATCH}"
echo "Target:        ${TARGET}"
echo "Valid mask:    ${VALID_MASK:-<all voxels>}"
echo "Model kind:    ${MODEL_KIND}"
echo "Checkpoint:    ${CHECKPOINT}"
echo "Output dir:    ${OUTPUT_DIR}"
echo "Output prefix: ${OUTPUT_PREFIX}"
echo "Skip inference:${SKIP_INFERENCE}"
echo "Eval tag:      ${EVAL_TAG}"
echo "Best criterion:${BEST_THRESHOLD_CRITERION}"
echo "Sweep:         ${SWEEP_START}..${SWEEP_STOP} step ${SWEEP_STEP}"
echo "Topology:      ${TOPOLOGY_METRICS} connectivity=${TOPOLOGY_CONNECTIVITY} sweep=${TOPOLOGY_ON_SWEEP} corrected=${CORRECTED_TOPOLOGY_METRICS}"

cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1

if [[ "${SKIP_INFERENCE}" != "1" ]]; then
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
        --threshold         "${INFER_THRESHOLD}" \
        --norm_mode         "${NORM_MODE}"
else
    echo "Skipping inference and reusing existing outputs"
fi

VALID_MASK_ARGS=()
if [[ -n "${VALID_MASK}" ]]; then
    VALID_MASK_ARGS=(--valid-mask "${VALID_MASK}")
fi

TOPOLOGY_ARGS=()
if [[ "${TOPOLOGY_METRICS}" == "1" || "${TOPOLOGY_METRICS}" == "true" || "${TOPOLOGY_METRICS}" == "TRUE" ]]; then
    TOPOLOGY_ARGS=(--topology-metrics --topology-connectivity "${TOPOLOGY_CONNECTIVITY}")
    if [[ "${CORRECTED_TOPOLOGY_METRICS}" == "1" || "${CORRECTED_TOPOLOGY_METRICS}" == "true" || "${CORRECTED_TOPOLOGY_METRICS}" == "TRUE" ]]; then
        TOPOLOGY_ARGS+=(--corrected-topology-metrics)
    fi
fi

TOPOLOGY_SWEEP_ARGS=()
if [[ ${#TOPOLOGY_ARGS[@]} -gt 0 && ( "${TOPOLOGY_ON_SWEEP}" == "1" || "${TOPOLOGY_ON_SWEEP}" == "true" || "${TOPOLOGY_ON_SWEEP}" == "TRUE" ) ]]; then
    TOPOLOGY_SWEEP_ARGS=("${TOPOLOGY_ARGS[@]}" --topology-on-sweep)
fi

BEST_OUTPUT_PREFIX="${OUTPUT_PREFIX}"
if [[ "${EVAL_TAG}" != "eval" ]]; then
    BEST_OUTPUT_PREFIX="${OUTPUT_PREFIX}_${EVAL_TAG}"
fi

SWEEP_JSON="${OUTPUT_DIR}/${OUTPUT_PREFIX}_${EVAL_TAG}_sweep.json"
"${PYTHON_BIN}" inference/utils/evaluate_masked_prediction.py \
    --prediction              "${OUTPUT_DIR}/${OUTPUT_PREFIX}_pred_prob.nii.gz" \
    --target                  "${TARGET}" \
    "${VALID_MASK_ARGS[@]}" \
    --sweep-start             "${SWEEP_START}" \
    --sweep-stop              "${SWEEP_STOP}" \
    --sweep-step              "${SWEEP_STEP}" \
    --correct-neighbors \
    --correct-neighbor-rounds "${CORRECT_NEIGHBOR_ROUNDS}" \
    --device                  "${EVAL_DEVICE}" \
    --output-json             "${SWEEP_JSON}" \
    "${TOPOLOGY_SWEEP_ARGS[@]}"

CLASS_MAP_ARGS=()
if [[ -f "${OUTPUT_DIR}/${OUTPUT_PREFIX}_pred_class.nii.gz" ]]; then
    CLASS_MAP_ARGS=(--class-map "${OUTPUT_DIR}/${OUTPUT_PREFIX}_pred_class.nii.gz")
fi

BEST_JSON=$("${PYTHON_BIN}" inference/utils/materialize_best_threshold_prediction.py \
    --probability   "${OUTPUT_DIR}/${OUTPUT_PREFIX}_pred_prob.nii.gz" \
    --sweep-json    "${SWEEP_JSON}" \
    --output-dir    "${OUTPUT_DIR}" \
    --output-prefix "${BEST_OUTPUT_PREFIX}" \
    --criterion     "${BEST_THRESHOLD_CRITERION}" \
    "${CLASS_MAP_ARGS[@]}")

echo "${BEST_JSON}"
BEST_PRED=$(printf '%s\n' "${BEST_JSON}" | "${PYTHON_BIN}" -c 'import json,sys; print(json.load(sys.stdin)["output_path"])')

"${PYTHON_BIN}" inference/utils/evaluate_masked_prediction.py \
    --prediction              "${BEST_PRED}" \
    --target                  "${TARGET}" \
    "${VALID_MASK_ARGS[@]}" \
    --correct-neighbors \
    --correct-neighbor-rounds "${CORRECT_NEIGHBOR_ROUNDS}" \
    --device                  "${EVAL_DEVICE}" \
    --output-json             "${OUTPUT_DIR}/${OUTPUT_PREFIX}_${EVAL_TAG}_corrected.json" \
    "${TOPOLOGY_ARGS[@]}"

echo ""
echo "=== Done! ==="
echo "End: $(date)"
echo "Outputs: ${OUTPUT_DIR}"
