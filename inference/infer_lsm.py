"""
Run axon segmentation on a real LSM patch.

Usage
-----
    python infer_lsm.py \
        --input         /scratch/experiment/webknossos/macaque_NEFH_WM.npy \
        --checkpoint    /scratch/experiment/training_out_higher_range/checkpoints/best_model_ep200.pt \
        --output_dir    /scratch/experiment/webknossos/inference/macaque_NEFH_WM \
        --output_prefix macaque_NEFH_WM \
        --voxel_size    1.0

"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.layers import Norm
from monai.networks.nets import UNet


def parse_args():
    p = argparse.ArgumentParser(description="Axon segmentation on an LSM patch")
    p.add_argument("--input", required=True, help="Path to .npy uint16 patch")
    p.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    p.add_argument("--output_dir", required=True, help="Output directory for NIfTI files")
    p.add_argument(
        "--output_prefix",
        default=None,
        help="Prefix for saved NIfTI files (default: input filename stem)",
    )
    p.add_argument(
        "--voxel_size",
        type=float,
        default=1.0,
        help="Isotropic voxel size in µm for the NIfTI affine",
    )
    p.add_argument("--roi_size", type=int, default=128, help="Sliding window ROI size")
    p.add_argument("--sw_batch_size", type=int, default=4, help="Sliding window batch size")
    p.add_argument("--overlap", type=float, default=0.5, help="Sliding window overlap")
    p.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold")
    p.add_argument(
        "--norm_mode",
        default="percentile",
        choices=["percentile", "mean_shift", "clahe"],
        help="Intensity normalization strategy",
    )
    return p.parse_args()


def save_nii(arr: np.ndarray, path: Path, voxel_size: float):
    affine = np.diag([voxel_size, voxel_size, voxel_size, 1.0])
    nib.save(nib.Nifti1Image(arr, affine=affine), str(path))


def load_patch(input_path: Path) -> np.ndarray:
    if input_path.suffix == ".npy":
        return np.load(input_path)

    meta = json.loads(Path(str(input_path) + ".json").read_text())
    return np.memmap(str(input_path), dtype=meta["dtype"], mode="r", shape=tuple(meta["shape"]))


def normalize_patch(patch: np.ndarray, norm_mode: str) -> np.ndarray:
    patch_f = patch.astype(np.float32)

    if norm_mode == "percentile":
        p_lo, p_hi = np.percentile(patch_f, [0.5, 99.5])
        print(f"  Percentile clip: [{p_lo:.1f}, {p_hi:.1f}]")
        patch_f = np.clip(patch_f, p_lo, p_hi)
        patch_f = (patch_f - p_lo) / (p_hi - p_lo + 1e-8)
        return patch_f.astype(np.float32, copy=False)

    if norm_mode == "mean_shift":
        p_lo, p_hi = np.percentile(patch_f, [0.5, 99.5])
        patch_f = np.clip(patch_f, p_lo, p_hi)
        patch_f = (patch_f - p_lo) / (p_hi - p_lo + 1e-8)
        mu = float(patch_f.mean())
        patch_f = patch_f - mu + 0.2
        patch_f = np.clip(patch_f, 0.0, 1.0)
        print(f"  Mean before shift: {mu:.3f} -> shifted to 0.2")
        return patch_f.astype(np.float32, copy=False)

    from skimage.exposure import equalize_adapthist

    p_lo, p_hi = np.percentile(patch_f, [0.5, 99.5])
    patch_f = np.clip(patch_f, p_lo, p_hi)
    patch_f = (patch_f - p_lo) / (p_hi - p_lo + 1e-8)
    print("  Applying CLAHE per Z-slice ...")
    for z in range(patch_f.shape[2]):
        patch_f[:, :, z] = equalize_adapthist(
            patch_f[:, :, z], kernel_size=64, clip_limit=0.02
        )
    mu = float(patch_f.mean())
    patch_f = patch_f - mu + 0.2
    patch_f = np.clip(patch_f, 0.0, 1.0)
    print(f"  CLAHE done. Mean -> shifted to 0.2")
    return patch_f.astype(np.float32, copy=False)


def build_model(device: torch.device) -> UNet:
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
        dropout=0.1,
    ).to(device)


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_prefix = args.output_prefix or input_path.stem

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading {args.input} ...")
    patch = load_patch(input_path)
    print(f"  Shape: {patch.shape}, dtype: {patch.dtype}")
    print(f"  Intensity range: [{patch.min()}, {patch.max()}]")
    print(f"  Norm mode: {args.norm_mode}")

    patch_f = normalize_patch(patch, args.norm_mode)
    print(
        f"  Normalized range: [{patch_f.min():.3f}, {patch_f.max():.3f}] "
        f" mean={patch_f.mean():.3f}"
    )

    tensor = torch.from_numpy(patch_f).unsqueeze(0).unsqueeze(0).to(
        device=device,
        dtype=torch.float32,
    )
    print(
        f"  Tensor: {tensor.shape}, dtype={tensor.dtype}, "
        f"range [{tensor.min():.3f}, {tensor.max():.3f}]"
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    roi_size = (args.roi_size,) * 3
    print(
        f"Running sliding-window inference "
        f"(roi={args.roi_size}, overlap={args.overlap}, sw_batch={args.sw_batch_size}) ..."
    )

    autocast_enabled = device.type == "cuda"
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=autocast_enabled):
        pred_logits = sliding_window_inference(
            tensor,
            roi_size,
            args.sw_batch_size,
            model,
            overlap=args.overlap,
            mode="gaussian",
        )

    del tensor
    pred_logits_cpu = pred_logits.cpu()
    del pred_logits
    torch.cuda.empty_cache()

    prob_np = torch.sigmoid(pred_logits_cpu)[0, 0].numpy()
    del pred_logits_cpu
    pred_np = (prob_np >= args.threshold).astype(np.uint8)

    save_nii(patch_f, output_dir / f"{output_prefix}_input.nii.gz", args.voxel_size)
    save_nii(prob_np, output_dir / f"{output_prefix}_pred_prob.nii.gz", args.voxel_size)
    save_nii(pred_np, output_dir / f"{output_prefix}_pred.nii.gz", args.voxel_size)

    print(f"\nDone! Saved to {output_dir}/")
    print(
        f"{output_dir}/{output_prefix}_input.nii.gz "
        f"{output_dir}/{output_prefix}_pred.nii.gz"
    )


if __name__ == "__main__":
    main()