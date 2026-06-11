"""Extract NIfTI crops from inference outputs for quick review.

The utility supports both binary and 3-class outputs saved either with the
legacy ``hipct_*`` names or with another shared prefix such as the raw patch
stem used by ``infer_lsm.py``.

Saved crops always include:
    - normalized input crop
    - foreground probability crop
    - thresholded foreground prediction crop

For 3-class inference outputs, it also saves:
    - class-label crop
    - shell-only mask crop
    - interior-only mask crop

Usage
-----
        python inference/utils/save_crops.py \
                --inference_dir /scratch/experiment/hipct/inference_out_mean_shift \
                --patch_raw     /scratch/experiment/hipct/patch_I74_IC_zoom01.raw \
                --n_crops       5

Crops are placed at:
    - volume center
    - additional random locations with foreground fraction above a threshold
"""

import argparse
import json
import random
from pathlib import Path

import nibabel as nib
import numpy as np


VALID_THREE_CLASS_LABELS = {0, 1, 2}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inference_dir", required=True)
    p.add_argument("--patch_raw",     required=True,
                   help="Raw memmap file (patch_*.raw); .json sidecar must exist alongside it")
    p.add_argument(
        "--output_prefix",
        default=None,
        help=(
            "Inference file prefix. Default: auto-detect from a top-level "
            "*_pred.nii.gz file inside inference_dir."
        ),
    )
    p.add_argument("--n_crops",  type=int, default=5)
    p.add_argument("--crop_size",type=int, default=128)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument(
        "--segmentation_mode",
        default="auto",
        choices=["auto", "binary", "three_class_shell_interior"],
        help="Crop export mode. 'auto' detects a 3-class run when hipct_pred_class.nii.gz exists.",
    )
    p.add_argument(
        "--min_positive_fraction",
        type=float,
        default=0.05,
        help="Minimum fraction of foreground-positive voxels for a random crop to count as interesting.",
    )
    p.add_argument(
        "--voxel_size",
        type=float,
        default=0.857,
        help="Isotropic voxel size in µm for saved crop affines.",
    )
    return p.parse_args()


def save_crop_nii(arr, path, voxel_size=0.857):
    affine = np.diag([voxel_size] * 3 + [1.0])
    nib.save(nib.Nifti1Image(arr, affine=affine), str(path))


def find_prediction_prefixes(inference_dir: Path) -> list[str]:
    suffix = "_pred.nii.gz"
    prefixes = sorted(
        path.name[: -len(suffix)]
        for path in inference_dir.glob(f"*{suffix}")
        if path.is_file()
    )
    return prefixes


def resolve_output_prefix(requested_prefix: str | None, inference_dir: Path) -> str:
    if requested_prefix:
        return requested_prefix

    prefixes = find_prediction_prefixes(inference_dir)
    if not prefixes:
        raise FileNotFoundError(
            f"Could not find any top-level *_pred.nii.gz files in {inference_dir}"
        )

    if len(prefixes) == 1:
        return prefixes[0]

    if "hipct" in prefixes:
        return "hipct"

    raise RuntimeError(
        "Multiple prediction prefixes found in inference_dir; "
        f"please pass --output_prefix explicitly. Found: {prefixes}"
    )


def build_prediction_paths(inference_dir: Path, output_prefix: str) -> dict[str, Path]:
    return {
        "pred": inference_dir / f"{output_prefix}_pred.nii.gz",
        "pred_prob": inference_dir / f"{output_prefix}_pred_prob.nii.gz",
        "pred_class": inference_dir / f"{output_prefix}_pred_class.nii.gz",
    }


def resolve_segmentation_mode(requested_mode: str, pred_class_path: Path) -> str:
    if requested_mode != "auto":
        return requested_mode
    if pred_class_path.exists():
        return "three_class_shell_interior"
    return "binary"


def normalize_input_crop(inp_crop: np.ndarray) -> np.ndarray:
    p_lo, p_hi = np.percentile(inp_crop, [0.5, 99.5])
    return np.clip((inp_crop - p_lo) / (p_hi - p_lo + 1e-8), 0, 1)


def extract_crop(volume, ox: int, oy: int, oz: int, crop_size: int, dtype=None) -> np.ndarray:
    crop = np.asarray(volume[ox:ox+crop_size, oy:oy+crop_size, oz:oz+crop_size])
    if dtype is not None:
        crop = crop.astype(dtype, copy=False)
    return crop


def validate_three_class_labels(pred_class_crop: np.ndarray) -> list[int]:
    unique_classes = [int(v) for v in np.unique(pred_class_crop).tolist()]
    unexpected = sorted(set(unique_classes) - VALID_THREE_CLASS_LABELS)
    if unexpected:
        raise ValueError(
            "pred_class crop contains unexpected labels "
            f"{unexpected}; expected labels within {sorted(VALID_THREE_CLASS_LABELS)}"
        )
    return unique_classes


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    C = args.crop_size
    inf_dir  = Path(args.inference_dir)
    crop_dir = inf_dir / "crops"
    crop_dir.mkdir(exist_ok=True)
    output_prefix = resolve_output_prefix(args.output_prefix, inf_dir)
    prediction_paths = build_prediction_paths(inf_dir, output_prefix)
    segmentation_mode = resolve_segmentation_mode(
        args.segmentation_mode,
        prediction_paths["pred_class"],
    )

    raw_path  = Path(args.patch_raw)
    meta_path = Path(str(raw_path) + ".json")
    meta      = json.loads(meta_path.read_text())
    shape     = tuple(meta["shape"])  # (X, Y, Z)

    print(f"Volume shape: {shape}")
    print(f"Crop size: {C}³")
    print(f"Output prefix: {output_prefix}")
    print(f"Segmentation mode: {segmentation_mode}")

    pred_path = prediction_paths["pred"]
    prob_path = prediction_paths["pred_prob"]
    pred_class_path = prediction_paths["pred_class"]

    if not pred_path.exists():
        raise FileNotFoundError(f"Missing foreground prediction: {pred_path}")
    if not prob_path.exists():
        raise FileNotFoundError(f"Missing foreground probability map: {prob_path}")
    if segmentation_mode == "three_class_shell_interior" and not pred_class_path.exists():
        raise FileNotFoundError(f"Missing 3-class prediction map: {pred_class_path}")

    # Load foreground mask fully for crop selection; keep heavier arrays proxied.
    print("Loading foreground prediction for crop selection ...")
    pred_nii = nib.load(str(pred_path))
    pred = np.asarray(pred_nii.dataobj).astype(np.uint8)
    pred_prob_nii = nib.load(str(prob_path))
    pred_prob = pred_prob_nii.dataobj
    pred_class = None
    if segmentation_mode == "three_class_shell_interior":
        pred_class = nib.load(str(pred_class_path)).dataobj

    # Input via memmap (uint16, seekable)
    inp = np.memmap(str(raw_path), dtype=meta["dtype"], mode="r", shape=shape)

    # --- Pick crop origins ---
    max_origin = [s - C for s in shape]

    # 1. Volume center
    origins = [tuple(s // 2 - C // 2 for s in shape)]

    # 2. Random crops that have enough predicted foreground to be worth inspecting.
    attempts  = 0
    while len(origins) < args.n_crops and attempts < 500:
        ox = random.randint(0, max_origin[0])
        oy = random.randint(0, max_origin[1])
        oz = random.randint(0, max_origin[2])
        crop_pred = pred[ox:ox+C, oy:oy+C, oz:oz+C]
        if crop_pred.mean() >= args.min_positive_fraction:
            origins.append((ox, oy, oz))
        attempts += 1

    # fill remaining with any random origin if not enough high-density found
    while len(origins) < args.n_crops:
        origins.append(tuple(random.randint(0, m) for m in max_origin))

    print(f"Saving {len(origins)} crops ...")
    summary = {
        "inference_dir": str(inf_dir),
        "patch_raw": str(raw_path),
        "output_prefix": output_prefix,
        "segmentation_mode": segmentation_mode,
        "crop_size": C,
        "seed": args.seed,
        "min_positive_fraction": args.min_positive_fraction,
        "voxel_size_um": args.voxel_size,
        "crops": [],
    }

    for i, (ox, oy, oz) in enumerate(origins):
        tag = "center" if i == 0 else f"crop{i:02d}"

        # Input crop: uint16 → float32 normalised
        inp_crop = inp[ox:ox+C, oy:oy+C, oz:oz+C].astype(np.float32)
        inp_norm = normalize_input_crop(inp_crop)

        # Foreground crops
        pred_crop = pred[ox:ox+C, oy:oy+C, oz:oz+C]
        prob_crop = extract_crop(pred_prob, ox, oy, oz, C, dtype=np.float32)
        pos_frac = pred_crop.mean() * 100

        # Save input as uint8; save predictions with their native interpretation.
        inp_u8 = (inp_norm * 255).clip(0, 255).astype(np.uint8)
        save_crop_nii(inp_u8,    crop_dir / f"{tag}_input.nii.gz", voxel_size=args.voxel_size)
        save_crop_nii(prob_crop, crop_dir / f"{tag}_pred_prob.nii.gz", voxel_size=args.voxel_size)
        save_crop_nii(pred_crop, crop_dir / f"{tag}_pred.nii.gz", voxel_size=args.voxel_size)

        crop_record = {
            "tag": tag,
            "origin": [int(ox), int(oy), int(oz)],
            "foreground_fraction": float(pred_crop.mean()),
            "foreground_probability_mean": float(prob_crop.mean()),
            "foreground_probability_max": float(prob_crop.max()),
        }

        if segmentation_mode == "three_class_shell_interior" and pred_class is not None:
            pred_class_crop = extract_crop(pred_class, ox, oy, oz, C, dtype=np.uint8)
            unique_classes = validate_three_class_labels(pred_class_crop)
            shell_crop = (pred_class_crop == 1).astype(np.uint8)
            interior_crop = (pred_class_crop == 2).astype(np.uint8)
            save_crop_nii(pred_class_crop, crop_dir / f"{tag}_pred_class.nii.gz", voxel_size=args.voxel_size)
            save_crop_nii(shell_crop, crop_dir / f"{tag}_pred_shell.nii.gz", voxel_size=args.voxel_size)
            save_crop_nii(interior_crop, crop_dir / f"{tag}_pred_interior.nii.gz", voxel_size=args.voxel_size)
            crop_record.update({
                "shell_fraction": float(shell_crop.mean()),
                "interior_fraction": float(interior_crop.mean()),
                "predicted_classes_present": unique_classes,
            })
            print(f"      classes={unique_classes}")

        summary["crops"].append(crop_record)

        print(f"  [{i+1}/{len(origins)}] {tag}  origin=({ox},{oy},{oz})  "
              f"positive={pos_frac:.1f}%")

    summary_path = crop_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nDone. Crops saved to {crop_dir}/")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
