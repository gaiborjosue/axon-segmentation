#!/bin/bash
#SBATCH --job-name=dense_3class_labels
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=edward.gaibor001@umb.edu
#SBATCH --output=/home/egaibor/orcd/scratch/experiment/logs_dense_three_class/export_dense_3class_%j.out
#SBATCH --error=/home/egaibor/orcd/scratch/experiment/logs_dense_three_class/export_dense_3class_%j.err

# ==============================================================================
# Precompute dense 3-class shell/interior targets from saved dense instance
# labels. This is a lightweight CPU-only export: it writes one
# ``*_gt_class.nii.gz`` per ``*_label.nii.gz`` into a parallel directory.
#
# Default source: /scratch/experiment/dense_labels_vl/train
# Default out:    /scratch/experiment/dense_labels_vl_three_class/train
#
# Optional overrides when submitting:
#   sbatch --export=ALL,LABEL_DIR=/path/in,OUTPUT_DIR=/path/out run/export_dense_three_class_labels.sbatch
#   sbatch --export=ALL,MAX_VOLUMES=8,OVERWRITE=1 run/export_dense_three_class_labels.sbatch
# ==============================================================================

set -euo pipefail

echo "=== Dense 3-Class Label Export ==="
echo "Start: $(date)"

module purge
module load StdEnv gcc/12.2.0 intel/2024.2.1 miniforge/24.3.0-0

set +u
source activate synthsplinegpu
set -u

PROJECT_DIR="/home/egaibor/orcd/pool/projects/axon-segmentation/experiment"
LABEL_DIR="${LABEL_DIR:-/home/egaibor/orcd/scratch/experiment/dense_labels_vl/train}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/egaibor/orcd/scratch/experiment/dense_labels_vl_three_class/train}"
LOG_DIR="/home/egaibor/orcd/scratch/experiment/logs_dense_three_class"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1

cmd=(
  python export_dense_three_class_labels.py
  --label_dir "${LABEL_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --copy_meta
)

if [[ -n "${MAX_VOLUMES:-}" ]]; then
  cmd+=(--max_volumes "${MAX_VOLUMES}")
fi

if [[ "${OVERWRITE:-0}" == "1" ]]; then
  cmd+=(--overwrite)
fi

echo "Command: ${cmd[*]}"
"${cmd[@]}"

echo ""
echo "=== Done! ==="
echo "End: $(date)"
echo "Output: ${OUTPUT_DIR}"