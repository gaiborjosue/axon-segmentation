#!/bin/bash
#SBATCH --job-name=vol0507
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --exclude=node2804
#SBATCH --output=/home/egaibor/orcd/scratch/experiment/logs/vol0507_%j.out
#SBATCH --error=/home/egaibor/orcd/scratch/experiment/logs/vol0507_%j.err

VOL_ID=507
VOL_STR="0507"
SPLIT="train"
BASE_DIR="/home/egaibor/orcd/scratch/experiment/dense_labels"
OUT_DIR="${BASE_DIR}/${SPLIT}"
LABEL_FILE="${OUT_DIR}/vol${VOL_STR}_label.nii.gz"

if [[ -f "$LABEL_FILE" ]]; then
    echo "vol${VOL_STR} already exists — nothing to do"; exit 0
fi

echo "=== vol${VOL_STR} | node=$(hostname) | $(date) ==="

module purge
module load StdEnv gcc/12.2.0 intel/2024.2.1 miniforge/24.3.0-0 cuda/12.4.0
source activate synthsplinegpu

PROJECT_DIR="/home/egaibor/orcd/pool/projects/axon-segmentation/experiment"
PYTHON="/home/egaibor/orcd/pool/projects/synthspline/envs/synthsplinegpu/bin/python"

cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1

$PYTHON - <<EOF
import sys, time
sys.path.insert(0, '.')

import torch
import numpy as np
import nibabel as nib

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device, '|', torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU')

vol_id  = 507
vol_str = '0507'
out_dir = '${OUT_DIR}'
affine  = np.eye(4) * 0.0005
affine[3, 3] = 1

rng_seed = 99_000 + vol_id  # alternate seed: 42507 causes pathological synthspline hang
np.random.seed(rng_seed % (2**31))
torch.manual_seed(rng_seed % (2**31))

from datagen.axon_labels_full_density import FullDensityUnidirectionalAxon

density_exp  = np.random.uniform(6.75, 7.1)  # alternate seed 99507, capped at 7.1 to be safe
tree_density = 10 ** density_exp
print(f'Density: 10^{density_exp:.3f} = {tree_density:.2e} trees/mm³')

synth = FullDensityUnidirectionalAxon(
    shape=(128, 128, 128),
    voxel_size=0.0005,
    tree_density=tree_density,
).to(device)

t0 = time.time()
with torch.no_grad():
    result = synth()
elapsed = time.time() - t0

labels_np = result[1][0, 0].cpu().numpy().astype(np.int32)
prob_np   = result[0][0, 0].cpu().numpy().astype(np.float32)
n_axons   = len(np.unique(labels_np)) - 1
print(f'n_axons={n_axons}  elapsed={elapsed:.1f}s')

nib.save(nib.Nifti1Image(labels_np, affine), f'{out_dir}/vol{vol_str}_label.nii.gz')
nib.save(nib.Nifti1Image(prob_np,   affine), f'{out_dir}/vol{vol_str}_prob.nii.gz')

with open(f'{out_dir}/vol{vol_str}_meta.txt', 'w') as fh:
    fh.write(f'vol_id: {vol_id}\nsplit: train\n')
    fh.write(f'density_exponent: {density_exp:.4f}\ntree_density: {tree_density:.6e}\n')
    fh.write(f'n_axons: {n_axons}\nshape: 128x128x128\nvoxel_size: 0.0005\n')
    fh.write(f'elapsed_s: {elapsed:.2f}\nrng_seed: {rng_seed}\n')

print(f'Saved vol{vol_str}')
EOF

echo "=== Done | $(date) ==="
