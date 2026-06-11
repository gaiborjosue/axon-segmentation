#!/bin/bash
#SBATCH --job-name=bench_fast_rasterizer
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/egaibor/orcd/scratch/experiment/logs/bench_fast_rasterizer_%j.out
#SBATCH --error=/home/egaibor/orcd/scratch/experiment/logs/bench_fast_rasterizer_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# bench_fast_rasterizer.sbatch
#
# Benchmarks fast_rasterize vs synthspline on a GPU node.
#
# Submit:
#   sbatch run/debug/bench_fast_rasterizer.sbatch
#
# Monitor:
#   tail -f /home/egaibor/orcd/scratch/experiment/logs/bench_fast_rasterizer_<JOB_ID>.out
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PYTHON=/home/egaibor/orcd/pool/projects/synthspline/envs/synthsplinegpu/bin/python
BENCH=/home/egaibor/orcd/pool/projects/axon-segmentation/experiment/fast_rasterizer/benchmark.py

echo "============================================================"
echo "  bench_fast_rasterizer  —  job ${SLURM_JOB_ID}"
echo "============================================================"
echo "  Node:   $(hostname)"
echo "  Date:   $(date)"
echo "  GPU(s): $CUDA_VISIBLE_DEVICES"
echo ""

# Confirm GPU is visible
$PYTHON -c "import torch; print('CUDA available:', torch.cuda.is_available()); \
            print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo ""
echo "============================================================"
echo "  RUN 1 — 200 curves, 128^3  (quick correctness check)"
echo "============================================================"
$PYTHON "$BENCH" \
    --n_curves   200 \
    --shape      128 \
    --seg_length 1.5 \
    --batch_size 2048 \
    --device     cuda

echo ""
echo "============================================================"
echo "  RUN 2 — 500 curves, 128^3  (medium scale timing)"
echo "============================================================"
$PYTHON "$BENCH" \
    --n_curves   500 \
    --shape      128 \
    --seg_length 1.5 \
    --batch_size 2048 \
    --device     cuda

echo ""
echo "============================================================"
echo "  RUN 3 — 1000 curves, 128^3 (fast_rasterize only, no ref)"
echo "  (reference would be too slow; just checks our throughput)"
echo "============================================================"
$PYTHON "$BENCH" \
    --n_curves   1000 \
    --shape      128  \
    --seg_length 1.5  \
    --batch_size 512  \
    --device     cuda \
    --skip_ref

echo ""
echo "============================================================"
echo "  RUN 4 — 1000 curves, batch_size=2048  (bigger batch)"
echo "============================================================"
$PYTHON "$BENCH" \
    --n_curves   1000 \
    --shape      128  \
    --seg_length 1.5  \
    --batch_size 2048 \
    --device     cuda \
    --skip_ref

echo ""
echo "Done!  $(date)"
