#!/bin/bash
#SBATCH --job-name=infer_hipct
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=edward.gaibor001@umb.edu
#SBATCH --output=/home/egaibor/orcd/scratch/experiment/logs/infer_hipct_%j.out
#SBATCH --error=/home/egaibor/orcd/scratch/experiment/logs/infer_hipct_%j.err

# ==============================================================================
# HiP-CT Axon Segmentation Inference
#
# Input:  Downloaded .npy patch from LINCBrain dandiset 000005
#         sub-I74_sample-IC_acq-zoom_chunk-01_XPCT.ome.zarr
#         Bounding box: (1461,210,3295) → (3323,1180,4889)
#         Patch size: 1862 × 970 × 1594, uint16, 0.857 µm isotropic
#
# Model:  3D UNet trained on synthetic axon data (best Dice 0.8828)
#
# Pre-requisite: Run download_patch.py on login node first.
# ==============================================================================

echo "=== HiP-CT Axon Segmentation Inference ==="
echo "Start: $(date)"

module purge
module load StdEnv gcc/12.2.0 intel/2024.2.1 miniforge/24.3.0-0 cuda/12.4.0
source activate synthsplinegpu

PROJECT_DIR="/home/egaibor/orcd/pool/projects/axon-segmentation/experiment"
PATCH="/home/egaibor/orcd/scratch/experiment/hipct/patch_I74_IC_zoom01.raw"
CHECKPOINT="/home/egaibor/orcd/scratch/experiment/training_out/checkpoints/best_model.pt"
OUTPUT_DIR="/home/egaibor/orcd/scratch/experiment/hipct/inference_out"

# norm_mode options: percentile (original), mean_shift, clahe
NORM_MODE="mean_shift"

mkdir -p "${OUTPUT_DIR}" \
         "/home/egaibor/orcd/scratch/experiment/logs"

cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1

python inference/infer_hipct.py \
    --input         "${PATCH}" \
    --checkpoint    "${CHECKPOINT}" \
    --output_dir    "${OUTPUT_DIR}_${NORM_MODE}" \
    --voxel_size    0.857 \
    --roi_size      128 \
    --sw_batch_size 4 \
    --overlap       0.5 \
    --threshold     0.5 \
    --norm_mode     "${NORM_MODE}"

echo ""
echo "=== Done! ==="
echo "End: $(date)"
echo "Outputs: ${OUTPUT_DIR}"
