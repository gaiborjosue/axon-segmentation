#!/bin/bash
#SBATCH --job-name=infer_val
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=edward.gaibor001@umb.edu
#SBATCH --output=/home/egaibor/orcd/scratch/experiment/logs/infer_val_%j.out
#SBATCH --error=/home/egaibor/orcd/scratch/experiment/logs/infer_val_%j.err

set -euo pipefail

echo "=== Val Inference ==="
echo "Start: $(date)"

module purge
module load deprecated-modules
module load StdEnv gcc/12.2.0 intel/2024.2.1 miniforge/24.3.0-0 cuda/12.4.0

# Conda activation scripts in this environment are not nounset-safe.
set +u
source activate synthsplinegpu
set -u

PROJECT_DIR="/home/egaibor/orcd/pool/projects/axon-segmentation/experiment"
CHECKPOINT="${CHECKPOINT:-/home/egaibor/orcd/scratch/experiment/training_out_higher_range_three_class/checkpoints/best_model.pt}"
LABEL_DIR="${LABEL_DIR:-/home/egaibor/orcd/scratch/experiment/dense_labels_vl/train}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/egaibor/orcd/scratch/experiment/inference_val_higher_range_three_class/best_model}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
N_SAMPLES="${N_SAMPLES:-8}"
ROI_SIZE="${ROI_SIZE:-128}"
SW_BATCH_SIZE="${SW_BATCH_SIZE:-4}"
N_LABEL_GROUPS="${N_LABEL_GROUPS:-8}"
THRESHOLD="${THRESHOLD:-0.5}"

mkdir -p "${OUTPUT_DIR}" \
         "/home/egaibor/orcd/scratch/experiment/logs"

cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1

python inference/infer_val.py \
    --checkpoint     "${CHECKPOINT}" \
    --label_dir      "${LABEL_DIR}" \
    --output_dir     "${OUTPUT_DIR}" \
    --val_fraction   "${VAL_FRACTION}" \
    --n_samples      "${N_SAMPLES}" \
    --roi_size       "${ROI_SIZE}" \
    --sw_batch_size  "${SW_BATCH_SIZE}" \
    --n_label_groups "${N_LABEL_GROUPS}" \
    --threshold      "${THRESHOLD}"

echo ""
echo "=== Done! ==="
echo "End: $(date)"
echo "Outputs: ${OUTPUT_DIR}"
