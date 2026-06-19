#!/bin/bash
#SBATCH --job-name=stage_microct_dandi
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=edward.gaibor001@umb.edu
#SBATCH --output=/home/egaibor/orcd/scratch/experiment/logs/stage_microct_dandi_%j.out
#SBATCH --error=/home/egaibor/orcd/scratch/experiment/logs/stage_microct_dandi_%j.err

set -euo pipefail

echo "=== Stage microCT DANDI derivative ==="
echo "Start: $(date)"

module purge
module load deprecated-modules
module load StdEnv gcc/12.2.0 intel/2024.2.1 miniforge/24.3.0-0

PROJECT_DIR="/home/egaibor/orcd/pool/projects/axon-segmentation/experiment"
PYTHON_BIN="${PYTHON_BIN:-/home/egaibor/orcd/scratch/experiment/envs/synthsplinegpufix/bin/python}"
DANDI_BIN="${DANDI_BIN:-/home/egaibor/orcd/scratch/experiment/envs/synthsplinegpufix/bin/dandi}"
INFER_DIR="${INFER_DIR:-/home/egaibor/orcd/scratch/experiment/microct/inference/sub-Hb1_sample-lefthemi+ICblock+punch1+piece1+trim9_uCT_patch/three_class_v2_best_percentile_inverted}"
OME_DIR="${OME_DIR:-/home/egaibor/orcd/scratch/experiment/microct/inference/sub-Hb1_sample-lefthemi+ICblock+punch1+piece1+trim9_uCT_patch/three_class_v2_best_percentile_inverted_ome}"
STAGE_ROOT="${STAGE_ROOT:-/home/egaibor/orcd/scratch/experiment/dandi_staging/001769}"
BRANCH_NAME="${BRANCH_NAME:-sub-Hb1_sample-lhicp1p1t9_desc-v2bestpctinv}"
STAGE_DIR="${STAGE_ROOT}/derivatives/${BRANCH_NAME}"
SOURCE_STEM="${SOURCE_STEM:-sub-Hb1_sample-lefthemi+ICblock+punch1+piece1+trim9_uCT_patch}"
SAMPLE_STEM="${SAMPLE_STEM:-sub-Hb1_sample-lhicp1p1t9}"
RUN_DESC="${RUN_DESC:-v2bestpctinv}"
VOXEL_SIZE_UM="${VOXEL_SIZE_UM:-0.3644713433459401}"
CHUNK_SHAPE="${CHUNK_SHAPE:-64,64,64}"
LOG_DIR="/home/egaibor/orcd/scratch/experiment/logs"
TMP_STAGE_DIR="${STAGE_DIR}.tmp.${SLURM_JOB_ID:-$$}"

INPUT_NIFTI="${SOURCE_STEM}_input.nii.gz"
PRED_NIFTI="${SOURCE_STEM}_pred.nii.gz"
PRED_CLASS_NIFTI="${SOURCE_STEM}_pred_class.nii.gz"
PRED_PROB_NIFTI="${SOURCE_STEM}_pred_prob.nii.gz"

INPUT_ASSET="${SAMPLE_STEM}_desc-input_micr.ome.zarr"
MASK_ASSET="${SAMPLE_STEM}_label-axon_mask.ome.zarr"
DSEG_ASSET="${SAMPLE_STEM}_desc-shellinterior_dseg.ome.zarr"
PROBSEG_ASSET="${SAMPLE_STEM}_label-axon_probseg.ome.zarr"

mkdir -p "${LOG_DIR}"

if [[ ! -d "${INFER_DIR}" ]]; then
    echo "Missing inference directory: ${INFER_DIR}" >&2
    exit 1
fi

for required in \
    "${INFER_DIR}/${INPUT_NIFTI}" \
    "${INFER_DIR}/${PRED_NIFTI}" \
    "${INFER_DIR}/${PRED_CLASS_NIFTI}" \
    "${INFER_DIR}/${PRED_PROB_NIFTI}"; do
    if [[ ! -f "${required}" ]]; then
        echo "Missing inference output: ${required}" >&2
        exit 1
    fi
done

if [[ ! -f "${STAGE_ROOT}/dandiset.yaml" ]]; then
    echo "Missing dandiset root: ${STAGE_ROOT}/dandiset.yaml" >&2
    exit 1
fi

if [[ -e "${OME_DIR}" && -n "$(find "${OME_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Refusing to reuse non-empty OME export directory: ${OME_DIR}" >&2
    exit 1
fi

if [[ -e "${STAGE_DIR}" ]]; then
    echo "Refusing to overwrite existing stage directory: ${STAGE_DIR}" >&2
    exit 1
fi

if [[ -e "${TMP_STAGE_DIR}" ]]; then
    echo "Temporary stage directory already exists: ${TMP_STAGE_DIR}" >&2
    exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing python interpreter: ${PYTHON_BIN}" >&2
    exit 1
fi

if [[ ! -x "${DANDI_BIN}" ]]; then
    echo "Missing dandi executable: ${DANDI_BIN}" >&2
    exit 1
fi

echo "Python:      ${PYTHON_BIN}"
echo "DANDI:       ${DANDI_BIN}"
echo "Inference:   ${INFER_DIR}"
echo "OME output:  ${OME_DIR}"
echo "Stage root:  ${STAGE_ROOT}"
echo "Stage dir:   ${STAGE_DIR}"
echo "Sample stem: ${SAMPLE_STEM}"
echo "Run desc:    ${RUN_DESC}"
echo "Voxel size:  ${VOXEL_SIZE_UM} um"

cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1

if command -v timeout >/dev/null 2>&1; then
    echo "Running import preflight ..."
    timeout 60s "${PYTHON_BIN}" -c "import inference.utils.export_nifti_to_ome_zarr"
else
    echo "Running import preflight without timeout ..."
    "${PYTHON_BIN}" -c "import inference.utils.export_nifti_to_ome_zarr"
fi

"${PYTHON_BIN}" inference/utils/export_nifti_to_ome_zarr.py \
    --input_dir "${INFER_DIR}" \
    --output_dir "${OME_DIR}" \
    --inputs \
        "${INPUT_NIFTI}" \
        "${PRED_NIFTI}" \
        "${PRED_CLASS_NIFTI}" \
        "${PRED_PROB_NIFTI}" \
    --chunk_shape "${CHUNK_SHAPE}"

mkdir -p "${STAGE_ROOT}/derivatives" "${TMP_STAGE_DIR}"

cp -al "${OME_DIR}/${SOURCE_STEM}_input.ome.zarr"      "${TMP_STAGE_DIR}/${INPUT_ASSET}"
cp -al "${OME_DIR}/${SOURCE_STEM}_pred.ome.zarr"       "${TMP_STAGE_DIR}/${MASK_ASSET}"
cp -al "${OME_DIR}/${SOURCE_STEM}_pred_class.ome.zarr" "${TMP_STAGE_DIR}/${DSEG_ASSET}"
cp -al "${OME_DIR}/${SOURCE_STEM}_pred_prob.ome.zarr"  "${TMP_STAGE_DIR}/${PROBSEG_ASSET}"

cat > "${TMP_STAGE_DIR}/README.md" <<EOF
microCT derivative upload for ${SAMPLE_STEM}.

- Raw source stem: ${SOURCE_STEM}
- Inference directory: ${INFER_DIR}
- Run description: ${RUN_DESC}
- Native voxel size: ${VOXEL_SIZE_UM} um isotropic
- Assets:
  - ${INPUT_ASSET}
  - ${MASK_ASSET}
  - ${DSEG_ASSET}
  - ${PROBSEG_ASSET}
EOF

mv "${TMP_STAGE_DIR}" "${STAGE_DIR}"

"${DANDI_BIN}" validate "${STAGE_ROOT}"

echo ""
echo "=== Done! ==="
echo "End: $(date)"
echo "Staged derivative: ${STAGE_DIR}"
echo "Upload command:"
echo "  cd ${STAGE_ROOT} && ${DANDI_BIN} upload derivatives/${BRANCH_NAME}"
