#!/bin/bash
#SBATCH --job-name=draft_3labels_128
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=edward.gaibor001@umb.edu
#SBATCH --output=/home/egaibor/orcd/scratch/experiment/logs/draft_3labels_128_%j.out
#SBATCH --error=/home/egaibor/orcd/scratch/experiment/logs/draft_3labels_128_%j.err

# ==============================================================================
# Generate 3 dense label volumes (128³) for pipeline testing
#
# Output: /scratch/experiment/draft/dense_labels/vol000{0,1,2}_{label,prob}.nii.gz
# Split:  2 train + 1 val  (val_fraction=0.2 → max(1, int(3*0.2)) = 1)
# ==============================================================================

echo "=== 3× Dense Labels (128³) — Pipeline Test ==="
echo "Start: $(date)"

module purge
module load StdEnv gcc/12.2.0 intel/2024.2.1 miniforge/24.3.0-0 cuda/12.4.0
source activate synthsplinegpu

PROJECT_DIR="/home/egaibor/orcd/pool/projects/axon-segmentation/experiment"
OUTPUT_DIR="/home/egaibor/orcd/scratch/experiment/draft/dense_labels"

mkdir -p "${OUTPUT_DIR}" \
         "/home/egaibor/orcd/scratch/experiment/logs"

cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1

python -c "
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
import nibabel as nib

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

output_dir = '${OUTPUT_DIR}'
affine     = np.eye(4) * 0.0005
affine[3, 3] = 1

from datagen.axon_labels_full_density import FullDensityUnidirectionalAxon

N_VOLUMES = 3

for vi in range(N_VOLUMES):
    print()
    print(f'=== Volume {vi+1}/{N_VOLUMES} ===')

    density_exp  = np.random.uniform(6.75, 7.25)
    tree_density = 10 ** density_exp
    print(f'Density: 10^{density_exp:.2f} = {tree_density:.2e} trees/mm³')

    synth = FullDensityUnidirectionalAxon(
        shape=(128, 128, 128),
        voxel_size=0.0005,
        tree_density=tree_density,
    ).to(device)

    with torch.no_grad():
        result = synth()

    labels_np = result[1][0, 0].cpu().numpy().astype(np.int32)
    prob_np   = result[0][0, 0].cpu().numpy().astype(np.float32)

    n_axons = len(np.unique(labels_np)) - 1
    print(f'{n_axons} axons, shape: {labels_np.shape}')

    nib.save(nib.Nifti1Image(labels_np, affine), f'{output_dir}/vol{vi:04d}_label.nii.gz')
    nib.save(nib.Nifti1Image(prob_np,   affine), f'{output_dir}/vol{vi:04d}_prob.nii.gz')
    with open(f'{output_dir}/vol{vi:04d}_meta.txt', 'w') as f:
        f.write(f'density_exponent: {density_exp:.4f}\n')
        f.write(f'tree_density: {tree_density:.6e}\n')
        f.write(f'n_axons: {n_axons}\n')
        f.write(f'shape: 128x128x128\n')
        f.write(f'voxel_size: 0.0005\n')

    print(f'Saved vol{vi:04d}')

    # Free GPU memory between volumes
    del synth, result
    torch.cuda.empty_cache()

print()
print(f'All {N_VOLUMES} volumes saved to {output_dir}/')
print('Done!')
"

echo ""
echo "=== Done! ==="
echo "End: $(date)"
