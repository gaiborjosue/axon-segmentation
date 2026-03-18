"""
Extract 128³ crops from HiP-CT inference output for quick preview in NiiVue.

Saves input + pred_prob + pred_binary crops to <output_dir>/crops/.

Usage
-----
    python inference/save_crops.py \
        --inference_dir /scratch/experiment/hipct/inference_out \
        --patch_raw     /scratch/experiment/hipct/patch_I74_IC_zoom01.raw \
        --n_crops       5

Crops are placed at:
  - volume center
  - 4 random locations with >5% predicted positive voxels (interesting regions)
"""

import argparse
import json
import random
from pathlib import Path

import nibabel as nib
import numpy as np


CROP = 128


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inference_dir", required=True)
    p.add_argument("--patch_raw",     required=True,
                   help="Raw memmap file (patch_*.raw); .json sidecar must exist alongside it")
    p.add_argument("--n_crops",  type=int, default=5)
    p.add_argument("--crop_size",type=int, default=CROP)
    p.add_argument("--seed",     type=int, default=42)
    return p.parse_args()


def save_crop_nii(arr, path, voxel_size=0.857):
    affine = np.diag([voxel_size] * 3 + [1.0])
    nib.save(nib.Nifti1Image(arr, affine=affine), str(path))


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    C = args.crop_size
    inf_dir  = Path(args.inference_dir)
    crop_dir = inf_dir / "crops"
    crop_dir.mkdir(exist_ok=True)

    raw_path  = Path(args.patch_raw)
    meta_path = Path(str(raw_path) + ".json")
    meta      = json.loads(meta_path.read_text())
    shape     = tuple(meta["shape"])  # (X, Y, Z)

    print(f"Volume shape: {shape}")
    print(f"Crop size: {C}³")

    # Load prediction fully as uint8 (~2.9 GB, not float32 which would be ~11.6 GB)
    print("Loading prediction ...")
    pred_nii = nib.load(str(inf_dir / "hipct_pred.nii.gz"))
    pred = np.asarray(pred_nii.dataobj).astype(np.uint8)

    # Load prob map fully (float32, ~11.5 GB) — skip if too large, use pred only
    prob_path = inf_dir / "hipct_pred_prob.nii.gz"

    # Input via memmap (uint16, seekable)
    inp = np.memmap(str(raw_path), dtype=meta["dtype"], mode="r", shape=shape)

    # --- Pick crop origins ---
    max_origin = [s - C for s in shape]

    # 1. Volume center
    origins = [tuple(s // 2 - C // 2 for s in shape)]

    # 2. Random crops that have >threshold positive voxels
    threshold = 0.05
    attempts  = 0
    while len(origins) < args.n_crops and attempts < 500:
        ox = random.randint(0, max_origin[0])
        oy = random.randint(0, max_origin[1])
        oz = random.randint(0, max_origin[2])
        crop_pred = pred[ox:ox+C, oy:oy+C, oz:oz+C]
        if crop_pred.mean() >= threshold:
            origins.append((ox, oy, oz))
        attempts += 1

    # fill remaining with any random origin if not enough high-density found
    while len(origins) < args.n_crops:
        origins.append(tuple(random.randint(0, m) for m in max_origin))

    print(f"Saving {len(origins)} crops ...")

    for i, (ox, oy, oz) in enumerate(origins):
        tag = "center" if i == 0 else f"crop{i:02d}"

        # Input crop: uint16 → float32 normalised
        inp_crop = inp[ox:ox+C, oy:oy+C, oz:oz+C].astype(np.float32)
        p_lo, p_hi = np.percentile(inp_crop, [0.5, 99.5])
        inp_norm = np.clip((inp_crop - p_lo) / (p_hi - p_lo + 1e-8), 0, 1)

        # Pred crop
        pred_crop = pred[ox:ox+C, oy:oy+C, oz:oz+C]
        pos_frac = pred_crop.mean() * 100

        # Save as uint8 (4x smaller than float32, NiiVue handles it well)
        inp_u8 = (inp_norm * 255).clip(0, 255).astype(np.uint8)
        save_crop_nii(inp_u8,    crop_dir / f"{tag}_input.nii.gz")
        save_crop_nii(pred_crop, crop_dir / f"{tag}_pred.nii.gz")

        print(f"  [{i+1}/{len(origins)}] {tag}  origin=({ox},{oy},{oz})  "
              f"positive={pos_frac:.1f}%")

    print(f"\nDone. Crops saved to {crop_dir}/")


if __name__ == "__main__":
    main()
