#!/bin/bash
#SBATCH --job-name=bench_gpu_cache
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=edward.gaibor001@umb.edu
#SBATCH --output=/home/egaibor/orcd/scratch/experiment/logs/bench_gpu_cache_%j.out
#SBATCH --error=/home/egaibor/orcd/scratch/experiment/logs/bench_gpu_cache_%j.err

set -euo pipefail

echo "=== GPU Cache Builder Benchmark ==="
echo "Start: $(date)"

module purge
module load deprecated-modules
module load StdEnv gcc/12.2.0 intel/2024.2.1 miniforge/24.3.0-0 cuda/12.4.0

set +u
source activate synthsplinegpu
set -u

PROJECT_DIR="/home/egaibor/orcd/pool/projects/axon-segmentation/experiment"
LABEL_DIR="${LABEL_DIR:-/home/egaibor/orcd/scratch/experiment/dense_labels_vl/train}"
OUTPUT_JSON="${OUTPUT_JSON:-/home/egaibor/orcd/scratch/experiment/bench_gpu_cache_builder/summary.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/egaibor/orcd/scratch/experiment/bench_gpu_cache_builder/samples}"
N_SAMPLES="${N_SAMPLES:-1}"
MAX_VOLUMES="${MAX_VOLUMES:-1}"
GPU_LABEL_BLOCK_SIZE="${GPU_LABEL_BLOCK_SIZE:-8}"

mkdir -p "$(dirname "${OUTPUT_JSON}")" "${OUTPUT_DIR}"

cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1

python tools/benchmark_gpu_cache_builder.py \
  --label_dir "${LABEL_DIR}" \
  --split val \
  --segmentation_mode three_class_shell_interior \
  --n_samples "${N_SAMPLES}" \
  --num_samples_per_volume 8 \
  --max_volumes "${MAX_VOLUMES}" \
  --subset_fraction_lo 0.3 \
  --subset_fraction_hi 0.9 \
  --gpu_label_block_size "${GPU_LABEL_BLOCK_SIZE}" \
  --output_dir "${OUTPUT_DIR}" \
  --output_json "${OUTPUT_JSON}"

echo ""
echo "=== Done! ==="
echo "End: $(date)"
echo "Summary: ${OUTPUT_JSON}"
echo "Samples: ${OUTPUT_DIR}"