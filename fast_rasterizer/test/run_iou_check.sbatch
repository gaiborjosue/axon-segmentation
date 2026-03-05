#!/bin/bash
#SBATCH --job-name=iou_sanity_check
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/egaibor/orcd/scratch/experiment/logs/iou_sanity_check_%j.out
#SBATCH --error=/home/egaibor/orcd/scratch/experiment/logs/iou_sanity_check_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# IoU sanity check: fast_rasterize vs synthspline
#
# Generates 1 volume with both methods (identical curves, seed=0),
# thresholds at prob > 0.5, computes IoU on binary masks.
#
# Submit:
#   sbatch experiment/fast_rasterizer/test/run_iou_check.sbatch
#
# Monitor:
#   tail -f /home/egaibor/orcd/scratch/experiment/logs/iou_sanity_check_<JOB_ID>.out
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PYTHON=/home/egaibor/orcd/pool/projects/synthspline/envs/synthsplinegpu/bin/python
SCRIPT=/home/egaibor/orcd/pool/projects/axon-segmentation/experiment/fast_rasterizer/test/iou_sanity_check.py

echo "============================================================"
echo "  IoU Sanity Check — job ${SLURM_JOB_ID}"
echo "============================================================"
echo "  Node: $(hostname)"
echo "  Date: $(date)"
echo ""

$PYTHON -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"

echo ""
echo "  Running check with 2500 curves (realistic density) + slice plot …"
$PYTHON "$SCRIPT" \
    --n_curves   2500 \
    --shape      128  \
    --seg_length 1.5  \
    --batch_size 2048 \
    --threshold  0.5  \
    --device     cuda \
    --save_slices

echo ""
echo "Done!  $(date)"
