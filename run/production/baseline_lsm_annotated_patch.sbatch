#!/bin/bash
#SBATCH --job-name=baseline_lsm_annotated
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=edward.gaibor001@umb.edu
#SBATCH --output=/home/egaibor/orcd/scratch/experiment/logs/baseline_lsm_annotated_%j.out
#SBATCH --error=/home/egaibor/orcd/scratch/experiment/logs/baseline_lsm_annotated_%j.err

# ==============================================================================
# CPU intensity-threshold baseline + complete evaluation for one annotated LSM patch.
#
# Required override:
#   PATCH=/path/to/<patch>_raw.nii.gz
# ==============================================================================

set -euo pipefail

echo "=== Annotated LSM Threshold Baseline + Eval ==="
echo "Start: $(date)"

module purge
module load deprecated-modules
module load StdEnv gcc/12.2.0 intel/2024.2.1 miniforge/24.3.0-0

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

BASELINE_KIND="${BASELINE_KIND:-threshold}"
NORM_MODE="${NORM_MODE:-percentile}"
VOXEL_SIZE="${VOXEL_SIZE:-1.0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PATCH_DIR}/baseline}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-${PATCH_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${PATCH_NAME}/${BASELINE_KIND}}"
SKIP_BASELINE="${SKIP_BASELINE:-0}"

EVAL_TAG="${EVAL_TAG:-eval_complete}"
BEST_THRESHOLD_CRITERION="${BEST_THRESHOLD_CRITERION:-corrected_dice}"
SWEEP_START="${SWEEP_START:-0.05}"
SWEEP_STOP="${SWEEP_STOP:-0.95}"
SWEEP_STEP="${SWEEP_STEP:-0.01}"
CORRECT_NEIGHBOR_ROUNDS="${CORRECT_NEIGHBOR_ROUNDS:-2}"
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
TOPOLOGY_METRICS="${TOPOLOGY_METRICS:-1}"
TOPOLOGY_ON_SWEEP="${TOPOLOGY_ON_SWEEP:-0}"
TOPOLOGY_CONNECTIVITY="${TOPOLOGY_CONNECTIVITY:-6}"
CORRECTED_TOPOLOGY_METRICS="${CORRECTED_TOPOLOGY_METRICS:-0}"
SKIP_SWEEP_CLDICE="${SKIP_SWEEP_CLDICE:-1}"

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
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing python interpreter: ${PYTHON_BIN}" >&2
    exit 1
fi

echo "Python:        ${PYTHON_BIN}"
echo "Patch:         ${PATCH}"
echo "Target:        ${TARGET}"
echo "Valid mask:    ${VALID_MASK:-<all voxels>}"
echo "Baseline kind: ${BASELINE_KIND}"
echo "Output dir:    ${OUTPUT_DIR}"
echo "Output prefix: ${OUTPUT_PREFIX}"
echo "Skip baseline: ${SKIP_BASELINE}"
echo "Eval tag:      ${EVAL_TAG}"
echo "Best criterion:${BEST_THRESHOLD_CRITERION}"
echo "Sweep:         ${SWEEP_START}..${SWEEP_STOP} step ${SWEEP_STEP}"
echo "Sweep clDice:  $([[ "${SKIP_SWEEP_CLDICE}" == "1" ]] && printf 'disabled' || printf 'enabled')"
echo "Topology:      ${TOPOLOGY_METRICS} connectivity=${TOPOLOGY_CONNECTIVITY} sweep=${TOPOLOGY_ON_SWEEP} corrected=${CORRECTED_TOPOLOGY_METRICS}"

cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1

if [[ "${SKIP_BASELINE}" != "1" ]]; then
    "${PYTHON_BIN}" inference/baselines/threshold_lsm.py \
        --input         "${PATCH}" \
        --output-dir    "${OUTPUT_DIR}" \
        --output-prefix "${OUTPUT_PREFIX}" \
        --norm-mode     "${NORM_MODE}" \
        --voxel-size    "${VOXEL_SIZE}"
else
    echo "Skipping baseline score-map generation and reusing existing outputs"
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

SWEEP_CLDICE_ARGS=()
if [[ "${SKIP_SWEEP_CLDICE}" == "1" || "${SKIP_SWEEP_CLDICE}" == "true" || "${SKIP_SWEEP_CLDICE}" == "TRUE" ]]; then
    SWEEP_CLDICE_ARGS=(--skip-cldice)
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
    "${SWEEP_CLDICE_ARGS[@]}" \
    --output-json             "${SWEEP_JSON}" \
    "${TOPOLOGY_SWEEP_ARGS[@]}"

BEST_JSON=$("${PYTHON_BIN}" inference/utils/materialize_best_threshold_prediction.py \
    --probability   "${OUTPUT_DIR}/${OUTPUT_PREFIX}_pred_prob.nii.gz" \
    --sweep-json    "${SWEEP_JSON}" \
    --output-dir    "${OUTPUT_DIR}" \
    --output-prefix "${BEST_OUTPUT_PREFIX}" \
    --criterion     "${BEST_THRESHOLD_CRITERION}")

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
