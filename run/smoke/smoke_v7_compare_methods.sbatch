#!/bin/bash
#SBATCH --job-name=smoke_v7_methods
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=edward.gaibor001@umb.edu
#SBATCH --output=/home/egaibor/orcd/scratch/experiment/logs/smoke_v7_methods_%j.out
#SBATCH --error=/home/egaibor/orcd/scratch/experiment/logs/smoke_v7_methods_%j.err

set -euo pipefail

echo "=== smoke_v7 method comparison ==="
echo "Start: $(date)"

module purge
module load deprecated-modules
module load StdEnv gcc/12.2.0 intel/2024.2.1 miniforge/24.3.0-0 cuda/12.4.0

set +u
source activate synthsplinegpu
set -u

PROJECT_DIR="/home/egaibor/orcd/pool/projects/axon-segmentation/experiment"
LABEL_DIR="${LABEL_DIR:-/home/egaibor/orcd/scratch/experiment/dense_labels_vl/train}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/egaibor/orcd/scratch/experiment/inference_val_higher_range_three_class/smoke_v7}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
SAMPLE_IDX="${SAMPLE_IDX:-0}"
NUM_SAMPLES_PER_VOLUME="${NUM_SAMPLES_PER_VOLUME:-8}"
MAX_VOLUMES="${MAX_VOLUMES:-1}"
CROP_SIZE="${CROP_SIZE:-128}"
SEED="${SEED:-0}"
WALL_MODE="${WALL_MODE:-thin}"
DEBUG_PROGRESS="${DEBUG_PROGRESS:-0}"

mkdir -p "${OUTPUT_DIR}" \
         "/home/egaibor/orcd/scratch/experiment/logs"

cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"

cmd=(
  python inference/export_three_class_method_smoke.py
  --label_dir "${LABEL_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --sample_idx "${SAMPLE_IDX}"
  --val_fraction "${VAL_FRACTION}"
  --num_samples_per_volume "${NUM_SAMPLES_PER_VOLUME}"
  --max_volumes "${MAX_VOLUMES}"
  --crop_size "${CROP_SIZE}"
  --seed "${SEED}"
  --wall_mode "${WALL_MODE}"
)

if [[ "${DEBUG_PROGRESS}" != "0" ]]; then
  cmd+=(--debug_progress)
fi

printf 'Running command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

echo ""
echo "=== Done! ==="
echo "End: $(date)"
echo "Outputs: ${OUTPUT_DIR}"