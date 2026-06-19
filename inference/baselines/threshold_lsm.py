#!/usr/bin/env python3
"""Create an intensity-threshold baseline score map for an annotated LSM patch."""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from skimage.filters import threshold_otsu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create normalized-intensity baseline outputs for one LSM NIfTI patch."
    )
    parser.add_argument("--input", type=Path, required=True, help="Raw *_raw.nii.gz patch")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output prefix. Default: input stem with trailing _raw removed.",
    )
    parser.add_argument(
        "--norm-mode",
        default="percentile",
        choices=["percentile", "mean_shift", "clahe"],
        help="Normalization used before thresholding. Default matches LSM inference.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=None,
        help="Optional isotropic voxel size for output affine. Default: preserve input affine.",
    )
    return parser.parse_args()


def default_output_prefix(input_path: Path) -> str:
    name = input_path.name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    else:
        name = input_path.stem
    return name[:-4] if name.endswith("_raw") else name


def normalize_patch(patch: np.ndarray, norm_mode: str) -> tuple[np.ndarray, dict]:
    patch_f = patch.astype(np.float32)
    p_lo, p_hi = np.percentile(patch_f, [0.5, 99.5])
    patch_f = np.clip(patch_f, p_lo, p_hi)
    patch_f = (patch_f - p_lo) / (p_hi - p_lo + 1e-8)
    metadata = {
        "norm_mode": norm_mode,
        "percentile_clip": [float(p_lo), float(p_hi)],
    }

    if norm_mode == "percentile":
        return patch_f.astype(np.float32, copy=False), metadata

    if norm_mode == "mean_shift":
        mean_before_shift = float(patch_f.mean())
        patch_f = np.clip(patch_f - mean_before_shift + 0.2, 0.0, 1.0)
        metadata["mean_before_shift"] = mean_before_shift
        metadata["target_mean"] = 0.2
        return patch_f.astype(np.float32, copy=False), metadata

    from skimage.exposure import equalize_adapthist

    for z in range(patch_f.shape[2]):
        patch_f[:, :, z] = equalize_adapthist(
            patch_f[:, :, z], kernel_size=64, clip_limit=0.02
        )
    mean_before_shift = float(patch_f.mean())
    patch_f = np.clip(patch_f - mean_before_shift + 0.2, 0.0, 1.0)
    metadata["mean_before_shift"] = mean_before_shift
    metadata["target_mean"] = 0.2
    return patch_f.astype(np.float32, copy=False), metadata


def output_affine(input_img: nib.Nifti1Image, voxel_size: float | None) -> np.ndarray:
    if voxel_size is None:
        return input_img.affine
    return np.diag([voxel_size, voxel_size, voxel_size, 1.0])


def save_nii(arr: np.ndarray, path: Path, affine: np.ndarray):
    nib.save(nib.Nifti1Image(arr, affine=affine), str(path))


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = args.output_prefix or default_output_prefix(args.input)

    img = nib.load(str(args.input))
    raw = np.asarray(img.dataobj)
    score, norm_metadata = normalize_patch(raw, args.norm_mode)
    otsu_threshold = float(threshold_otsu(score))
    otsu_pred = (score >= otsu_threshold).astype(np.uint8)
    affine = output_affine(img, args.voxel_size)

    input_path = args.output_dir / f"{output_prefix}_input.nii.gz"
    prob_path = args.output_dir / f"{output_prefix}_pred_prob.nii.gz"
    otsu_path = args.output_dir / f"{output_prefix}_pred_otsu.nii.gz"
    metadata_path = args.output_dir / f"{output_prefix}_baseline_metadata.json"

    save_nii(score, input_path, affine)
    save_nii(score, prob_path, affine)
    save_nii(otsu_pred, otsu_path, affine)

    metadata = {
        "method": "intensity_threshold_baseline",
        "input_path": str(args.input),
        "output_dir": str(args.output_dir),
        "output_prefix": output_prefix,
        "shape": list(raw.shape),
        "input_dtype": str(raw.dtype),
        "input_intensity_range": [float(np.min(raw)), float(np.max(raw))],
        "normalized_intensity_range": [float(np.min(score)), float(np.max(score))],
        "normalized_mean": float(np.mean(score)),
        "normalized_std": float(np.std(score)),
        "otsu_threshold": otsu_threshold,
        "otsu_positive_fraction": float(otsu_pred.mean()),
        **norm_metadata,
        "outputs": {
            "input": str(input_path),
            "pred_prob": str(prob_path),
            "pred_otsu": str(otsu_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
