#!/bin/bash
set -euo pipefail

# Submit binary and 3-class inference/eval jobs for every annotated LSM patch.
#
# Usage:
#   bash run/production/submit_lsm_annotated_all.sh
#   DRY_RUN=1 bash run/production/submit_lsm_annotated_all.sh

DATA_ROOT="${DATA_ROOT:-/home/egaibor/orcd/scratch/LSM_axonal_marker_annotated_patches}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-run/production/infer_lsm_annotated_patch.sbatch}"
MODEL_KINDS="${MODEL_KINDS:-binary threeclass}"

EVAL_TAG="${EVAL_TAG:-eval_complete}"
TOPOLOGY_METRICS="${TOPOLOGY_METRICS:-1}"
BEST_THRESHOLD_CRITERION="${BEST_THRESHOLD_CRITERION:-corrected_dice}"
NORM_MODE="${NORM_MODE:-percentile}"

shopt -s nullglob

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "Missing data root: ${DATA_ROOT}" >&2
    exit 1
fi

if [[ ! -f "${SBATCH_SCRIPT}" ]]; then
    echo "Missing sbatch script: ${SBATCH_SCRIPT}" >&2
    exit 1
fi

raw_patches=("${DATA_ROOT}"/*/*_raw.nii.gz)
if [[ ${#raw_patches[@]} -eq 0 ]]; then
    echo "No *_raw.nii.gz patches found under ${DATA_ROOT}" >&2
    exit 1
fi

submitted=0
for patch in "${raw_patches[@]}"; do
    target="${patch%_raw.nii.gz}_gt.nii.gz"
    if [[ ! -f "${target}" ]]; then
        echo "Skipping ${patch}; missing target ${target}" >&2
        continue
    fi

    for model_kind in ${MODEL_KINDS}; do
        export_arg="ALL,PATCH=${patch},MODEL_KIND=${model_kind},EVAL_TAG=${EVAL_TAG},TOPOLOGY_METRICS=${TOPOLOGY_METRICS},BEST_THRESHOLD_CRITERION=${BEST_THRESHOLD_CRITERION},NORM_MODE=${NORM_MODE}"
        if [[ "${DRY_RUN:-0}" == "1" ]]; then
            echo "sbatch --export=${export_arg} ${SBATCH_SCRIPT}"
        else
            job_id=$(sbatch --parsable --export="${export_arg}" "${SBATCH_SCRIPT}")
            echo "Submitted ${job_id}: ${model_kind} ${patch}"
        fi
        submitted=$((submitted + 1))
    done
done

echo "Prepared ${submitted} job submissions."
