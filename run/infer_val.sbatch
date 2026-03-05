#!/bin/bash
#SBATCH --job-name=infer_val_draft
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=edward.gaibor001@umb.edu
#SBATCH --output=/home/egaibor/orcd/scratch/experiment/logs/infer_val_draft_%j.out
#SBATCH --error=/home/egaibor/orcd/scratch/experiment/logs/infer_val_draft_%j.err

echo "=== Val Inference ==="
echo "Start: $(date)"

module purge
module load StdEnv gcc/12.2.0 intel/2024.2.1 miniforge/24.3.0-0 cuda/12.4.0
source activate synthsplinegpu

PROJECT_DIR="/home/egaibor/orcd/pool/projects/axon-segmentation/experiment"
CHECKPOINT="/home/egaibor/orcd/scratch/experiment/draft/training_out/checkpoints/best_model.pt"
LABEL_DIR="/home/egaibor/orcd/scratch/experiment/draft/dense_labels"
OUTPUT_DIR="/home/egaibor/orcd/scratch/experiment/draft/inference_out"

mkdir -p "${OUTPUT_DIR}" \
         "/home/egaibor/orcd/scratch/experiment/logs"

cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1

python infer_val.py \
    --checkpoint     "${CHECKPOINT}" \
    --label_dir      "${LABEL_DIR}" \
    --output_dir     "${OUTPUT_DIR}" \
    --val_fraction   0.34 \
    --n_samples      5 \
    --roi_size       128 \
    --sw_batch_size  4 \
    --n_label_groups 8 \
    --threshold      0.5

echo ""
echo "=== Done! ==="
echo "End: $(date)"
echo "Outputs: ${OUTPUT_DIR}"
