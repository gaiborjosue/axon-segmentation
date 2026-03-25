#!/bin/bash
#SBATCH --job-name=axon_labels
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=edward.gaibor001@umb.edu
#SBATCH --output=/home/egaibor/orcd/scratch/experiment/logs_dense_larger/axon_labels_%A_%a.out
#SBATCH --error=/home/egaibor/orcd/scratch/experiment/logs_dense_larger/axon_labels_%A_%a.err

# ==============================================================================
# Job array: generate one 128³ dense label volume per task
#
# Array:    0–599   (600 volumes total, 0.8 µm voxel)
# %50:      ≤50 tasks running simultaneously
# Output:   /scratch/experiment/dense_labels_vl/{vol0000–vol0599}
#            └── train/ 0–509    (510 volumes, 85%)
#            └── val/   510–569  ( 60 volumes, 10%)
#            └── test/  570–599  ( 30 volumes,  5%)
# Resume:   if output files already exist the task exits 0 immediately
# ==============================================================================

VOL_ID=$SLURM_ARRAY_TASK_ID
VOL_STR=$(printf "%04d" $VOL_ID)

BASE_DIR="/home/egaibor/orcd/scratch/experiment/dense_labels_vl"
LOG_DIR="/home/egaibor/orcd/scratch/experiment/logs_dense_larger"

# Assign to split
if   [[ $VOL_ID -lt 510 ]]; then SPLIT="train"
elif [[ $VOL_ID -lt 570 ]]; then SPLIT="val"
else                             SPLIT="test"
fi

OUT_DIR="${BASE_DIR}/${SPLIT}"
LABEL_FILE="${OUT_DIR}/vol${VOL_STR}_label.nii.gz"

# ── Skip if already generated (resume safety) ──────────────────────────────
if [[ -f "$LABEL_FILE" ]]; then
    echo "vol${VOL_STR} already exists — skipping"
    exit 0
fi

echo "=== vol${VOL_STR} (${SPLIT}) | node=$(hostname) | $(date) ==="

module purge
module load StdEnv gcc/12.2.0 intel/2024.2.1 miniforge/24.3.0-0 cuda/12.4.0
source activate synthsplinegpu

mkdir -p "$OUT_DIR" "$LOG_DIR"

PROJECT_DIR="/home/egaibor/orcd/pool/projects/axon-segmentation/experiment"
PYTHON="/home/egaibor/orcd/pool/projects/synthspline/envs/synthsplinegpu/bin/python"

cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1

$PYTHON - <<EOF
import sys, os
sys.path.insert(0, '.')

import torch
import numpy as np
import nibabel as nib

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device, '|', torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU')

vol_id   = ${VOL_ID}
vol_str  = f'{vol_id:04d}'
out_dir  = '${OUT_DIR}'
affine   = np.eye(4) * 0.0008
affine[3, 3] = 1

# Seed for reproducibility — unique per volume but deterministic
rng_seed = 42_000 + vol_id
np.random.seed(rng_seed % (2**31))
torch.manual_seed(rng_seed % (2**31))

from datagen.axon_labels_full_density import FullDensityUnidirectionalAxon

density_exp  = np.random.uniform(6.1, 6.6)
tree_density = 10 ** density_exp
print(f'Density: 10^{density_exp:.3f} = {tree_density:.2e} trees/mm³')

synth = FullDensityUnidirectionalAxon(
    shape=(128, 128, 128),
    voxel_size=0.0008,
    tree_density=tree_density,
).to(device)

import time
t0 = time.time()
with torch.no_grad():
    result = synth()
elapsed = time.time() - t0

labels_np = result[1][0, 0].cpu().numpy().astype(np.int32)
prob_np   = result[0][0, 0].cpu().numpy().astype(np.float32)

n_axons = len(np.unique(labels_np)) - 1
print(f'n_axons={n_axons}  shape={labels_np.shape}  elapsed={elapsed:.1f}s')

nib.save(nib.Nifti1Image(labels_np, affine), f'{out_dir}/vol{vol_str}_label.nii.gz')
nib.save(nib.Nifti1Image(prob_np,   affine), f'{out_dir}/vol{vol_str}_prob.nii.gz')

with open(f'{out_dir}/vol{vol_str}_meta.txt', 'w') as fh:
    fh.write(f'vol_id: {vol_id}\n')
    fh.write(f'split: ${SPLIT}\n')
    fh.write(f'density_exponent: {density_exp:.4f}\n')
    fh.write(f'tree_density: {tree_density:.6e}\n')
    fh.write(f'n_axons: {n_axons}\n')
    fh.write(f'shape: 128x128x128\n')
    fh.write(f'voxel_size: 0.0008\n')
    fh.write(f'elapsed_s: {elapsed:.2f}\n')
    fh.write(f'rng_seed: {rng_seed}\n')

print(f'Saved vol{vol_str} → {out_dir}/')
EOF

echo "=== Done: vol${VOL_STR} | $(date) ==="
