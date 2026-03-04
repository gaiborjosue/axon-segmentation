#!/bin/bash
#SBATCH --job-name=datagen_64_contrast
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/home/egaibor/orcd/scratch/synthspline/datagen/logs/demo_64_contrast_%j.out
#SBATCH --error=/home/egaibor/orcd/scratch/synthspline/datagen/logs/demo_64_contrast_%j.err

# ==============================================================================
# Data Generation (64³) — Controlled Contrast
#
# Step 1: Generate 1 dense label volume  (shape 64³, voxel 0.5 μm)
# Step 2: Synthesise 5 images with controlled contrast via LogNormal subsetting
#
# Axons      → [Uniform(0.3, 0.5),  1.0]
# Background → [0,                  Uniform(0.2, 0.4)]
# ==============================================================================

echo "=== Datagen (64³) — Controlled Contrast ==="
echo "Start: $(date)"

module purge
module load StdEnv gcc/12.2.0 intel/2024.2.1 miniforge/24.3.0-0 cuda/12.4.0
source activate synthsplinegpu

PROJECT_DIR="/home/egaibor/orcd/pool/projects/axon-segmentation/experiment"
OUTPUT_DIR="/home/egaibor/orcd/scratch/synthspline/datagen"

mkdir -p "${OUTPUT_DIR}/dense_labels_64" \
         "${OUTPUT_DIR}/samples_64" \
         "${OUTPUT_DIR}/logs"

cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1

python -c "
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
import nibabel as nib
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

output_dir = '${OUTPUT_DIR}'
affine     = np.eye(4) * 0.0005
affine[3, 3] = 1

# ===== STEP 1: Generate dense label volume =====
print()
print('='*50)
print('STEP 1: Generating dense label volume (64³)...')
print('='*50)

from datagen.axon_labels_full_density import FullDensityUnidirectionalAxon

density_exp  = np.random.uniform(6.75, 7.25)
tree_density = 10 ** density_exp
print(f'Sampled density: 10^{density_exp:.2f} = {tree_density:.2e} trees/mm³')

synth = FullDensityUnidirectionalAxon(
    shape=(64, 64, 64),
    voxel_size=0.0005,
    tree_density=tree_density,
).to(device)

with torch.no_grad():
    result = synth()

labels_np = result[1][0, 0].cpu().numpy().astype(np.int32)
prob_np   = result[0][0, 0].cpu().numpy().astype(np.float32)

n_axons = len(np.unique(labels_np)) - 1
print(f'Generated {n_axons} axons, shape: {labels_np.shape}')

nib.save(nib.Nifti1Image(labels_np, affine), f'{output_dir}/dense_labels_64/vol0000_label.nii.gz')
nib.save(nib.Nifti1Image(prob_np,   affine), f'{output_dir}/dense_labels_64/vol0000_prob.nii.gz')
with open(f'{output_dir}/dense_labels_64/vol0000_meta.txt', 'w') as f:
    f.write(f'density_exponent: {density_exp:.4f}\n')
    f.write(f'tree_density: {tree_density:.6e}\n')
    f.write(f'n_axons: {n_axons}\n')
print(f'Saved to {output_dir}/dense_labels_64/')

# ===== STEP 2: Synthesise 5 images with controlled contrast =====
print()
print('='*50)
print('STEP 2: Generating 5 images with controlled contrast...')
print('='*50)
print('  Axons:      [Uniform(0.3,0.5), 1.0]')
print('  Background: [0,               Uniform(0.2,0.4)]')
print()

from datagen.axon_image_controlled_contrast import ControlledContrastAxonImage

image_synth = ControlledContrastAxonImage(
    background=0.5,
    fibers_lower_range=(0.3, 0.5),
    background_upper_range=(0.2, 0.4),
).to(device)

unique_axons  = np.unique(labels_np[labels_np > 0])
keep_mean_log = -0.3   # exp(-0.3) ≈ 0.74
keep_std_log  = 0.5
print(f'Total axons: {len(unique_axons)}')

for i in range(5):
    keep_prob = float(np.clip(np.exp(np.random.normal(keep_mean_log, keep_std_log)), 0.1, 1.0))
    print(f'  Image {i+1}/5 (keep_prob={keep_prob:.2f})...', end=' ')

    subset_labels = labels_np.copy()
    subset_prob   = prob_np.copy()
    for aid in unique_axons:
        if random.random() > keep_prob:
            mask = labels_np == aid
            subset_labels[mask] = 0
            subset_prob[mask]   = 0

    n_kept  = len(np.unique(subset_labels)) - 1
    label_t = torch.from_numpy(subset_labels).unsqueeze(0).unsqueeze(0).to(device)
    prob_t  = torch.from_numpy(subset_prob).unsqueeze(0).unsqueeze(0).float().to(device)
    print(f'{n_kept}/{len(unique_axons)} axons...', end=' ')

    with torch.no_grad():
        image, out_prob = image_synth(label_t, prob_t)

    nib.save(nib.Nifti1Image(subset_labels.astype(np.int32),               affine), f'{output_dir}/samples_64/vol{i:04d}_label.nii.gz')
    nib.save(nib.Nifti1Image(image[0,0].cpu().numpy().astype(np.float32),  affine), f'{output_dir}/samples_64/vol{i:04d}_image.nii.gz')
    nib.save(nib.Nifti1Image(out_prob[0,0].cpu().numpy().astype(np.float32), affine), f'{output_dir}/samples_64/vol{i:04d}_prob.nii.gz')
    with open(f'{output_dir}/samples_64/vol{i:04d}_meta.txt', 'w') as f:
        f.write(f'keep_prob: {keep_prob:.4f}\nn_axons_kept: {n_kept}\nn_axons_total: {len(unique_axons)}\n')
    print('done')

print()
print('Dense label: ${OUTPUT_DIR}/dense_labels_64/')
print('Images:      ${OUTPUT_DIR}/samples_64/')
"

echo ""
echo "=== Done! ==="
echo "End: $(date)"
