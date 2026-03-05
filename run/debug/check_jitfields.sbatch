#!/bin/bash
#SBATCH --job-name=debug_jitfields
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/home/egaibor/orcd/scratch/experiment/logs/debug_jitfields_%j.out
#SBATCH --error=/home/egaibor/orcd/scratch/experiment/logs/debug_jitfields_%j.err

echo "=== jitfields / cupy debug ==="
echo "Start: $(date)"
echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'nvidia-smi not found')"
echo ""

PYTHON=/home/egaibor/orcd/pool/projects/synthspline/envs/synthsplinegpu/bin/python
SCRIPT=/home/egaibor/orcd/pool/projects/axon-segmentation/experiment/run/debug/check_jitfields.py

$PYTHON $SCRIPT

echo ""
echo "=== Done! ==="
echo "End: $(date)"
