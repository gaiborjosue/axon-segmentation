#!/bin/bash
#SBATCH --job-name=threshold_sweep
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=edward.gaibor001@umb.edu
#SBATCH --output=/home/egaibor/orcd/scratch/experiment/logs/threshold_sweep_%j.out
#SBATCH --error=/home/egaibor/orcd/scratch/experiment/logs/threshold_sweep_%j.err

# ==============================================================================
# Threshold sweep on HiP-CT probability map
#
# Generates 512³ center crops from hipct_pred_prob.nii.gz at multiple thresholds
# to diagnose over-segmentation and find the best operating point.
#
# Pre-requisite: inference job must have completed (hipct_pred_prob.nii.gz exists)
# ==============================================================================

echo "=== HiP-CT Threshold Sweep ==="
echo "Start: $(date)"

module purge
module load StdEnv gcc/12.2.0 intel/2024.2.1 miniforge/24.3.0-0 cuda/12.4.0
source activate synthsplinegpu

INF_DIR="/home/egaibor/orcd/scratch/experiment/hipct/inference_out"
RAW_DIR="/home/egaibor/orcd/scratch/experiment/hipct"
CROP_DIR="${INF_DIR}/crops"
CROP_SIZE=256

mkdir -p "${CROP_DIR}"

export PYTHONUNBUFFERED=1

python - <<'PYEOF'
import json
import nibabel as nib
import numpy as np
from pathlib import Path

inf_dir   = Path("/home/egaibor/orcd/scratch/experiment/hipct/inference_out_mean_shift")
raw_dir   = Path("/home/egaibor/orcd/scratch/experiment/hipct")
crop_dir  = inf_dir / "crops"
C         = 256

crop_dir.mkdir(parents=True, exist_ok=True)

affine = np.diag([0.857] * 3 + [1.0])

# Load probability map — get_fdata with float32 avoids an extra copy
print("Loading probability map ...")
prob_nii  = nib.load(str(inf_dir / "hipct_pred_prob.nii.gz"))
prob      = prob_nii.get_fdata(dtype=np.float32, caching='unchanged')
shape     = prob.shape
print(f"  Shape: {shape}  range: [{prob.min():.3f}, {prob.max():.3f}]")

# Load input via raw memmap (uint16, seekable)
meta = json.loads((raw_dir / "patch_I74_IC_zoom01.raw.json").read_text())
inp  = np.memmap(str(raw_dir / "patch_I74_IC_zoom01.raw"),
                 dtype=meta["dtype"], mode="r", shape=tuple(meta["shape"]))

# Center crop origin
ox = shape[0] // 2 - C // 2
oy = shape[1] // 2 - C // 2
oz = shape[2] // 2 - C // 2
print(f"Crop origin: ({ox}, {oy}, {oz})  size: {C}³")

# Save input crop once (uint8)
inp_crop = inp[ox:ox+C, oy:oy+C, oz:oz+C].astype(np.float32)
p_lo, p_hi = np.percentile(inp_crop, [0.5, 99.5])
inp_u8 = ((inp_crop - p_lo) / (p_hi - p_lo + 1e-8) * 255).clip(0, 255).astype(np.uint8)
nib.save(nib.Nifti1Image(inp_u8, affine), str(crop_dir / "center_input.nii.gz"))
print(f"Saved center_input.nii.gz")

# Crop probability map once
prob_crop = prob[ox:ox+C, oy:oy+C, oz:oz+C]
print(f"  Prob crop range: [{prob_crop.min():.3f}, {prob_crop.max():.3f}]")
print(f"  >0.5:  {(prob_crop > 0.5).mean()*100:.1f}%")
print(f"  >0.6:  {(prob_crop > 0.6).mean()*100:.1f}%")
print(f"  >0.7:  {(prob_crop > 0.7).mean()*100:.1f}%")
print(f"  >0.8:  {(prob_crop > 0.8).mean()*100:.1f}%")
print(f"  >0.9:  {(prob_crop > 0.9).mean()*100:.1f}%")
print(f"  >0.95: {(prob_crop > 0.95).mean()*100:.1f}%")

# Save binary crops at each threshold
for t in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    pred = (prob_crop >= t).astype(np.uint8)
    fname = crop_dir / f"center_pred_t{int(t*100):02d}.nii.gz"
    nib.save(nib.Nifti1Image(pred, affine), str(fname))
    print(f"  t={t:.2f}: {pred.mean()*100:.1f}% positive  -> {fname.name}")

print(f"\nDone. Open in NiiVue:")
print(f"  {crop_dir}/center_input.nii.gz")
print(f"  {crop_dir}/center_pred_t??.nii.gz")
PYEOF

echo ""
echo "=== Done! ==="
echo "End: $(date)"
