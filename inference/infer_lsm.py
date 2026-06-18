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
    p.add_argument("--input", required=True, help="Path to .npy, .nii/.nii.gz, or raw memmap patch")
    p.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    p.add_argument("--output_dir", required=True, help="Output directory for NIfTI files")
    p.add_argument(
        "--segmentation_mode",
        default="auto",
        choices=["auto", "binary", "three_class_shell_interior"],
        help="Segmentation mode. 'auto' uses the checkpoint args when available.",
    )
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
    p.add_argument(
        "--invert_contrast",
        action="store_true",
        help="Invert normalized intensities before inference, for data where axons are dark.",
    )
    return p.parse_args()


def save_nii(arr: np.ndarray, path: Path, voxel_size: float):
    affine = np.diag([voxel_size, voxel_size, voxel_size, 1.0])
    nib.save(nib.Nifti1Image(arr, affine=affine), str(path))


def load_patch(input_path: Path) -> np.ndarray:
    if input_path.suffix == ".npy":
        return np.load(input_path)
    if input_path.suffix == ".nii" or input_path.name.endswith(".nii.gz"):
        return np.asarray(nib.load(str(input_path)).dataobj)

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


def resolve_segmentation_mode(requested_mode: str, checkpoint: dict) -> str:
    if requested_mode != "auto":
        return requested_mode
    checkpoint_args = checkpoint.get("args") or {}
    return checkpoint_args.get("segmentation_mode", "binary")


def build_model(device: torch.device, segmentation_mode: str) -> UNet:
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1 if segmentation_mode == "binary" else 3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
        dropout=0.1,
    ).to(device)


def postprocess_logits(
    pred_logits: torch.Tensor,
    segmentation_mode: str,
    threshold: float,
) -> dict[str, np.ndarray]:
    if segmentation_mode == "binary":
        pred_prob = torch.sigmoid(pred_logits)[0, 0].numpy()
        pred_mask = (pred_prob >= threshold).astype(np.uint8)
        return {
            "pred_prob": pred_prob,
            "pred": pred_mask,
        }

    probabilities = torch.softmax(pred_logits, dim=1)[0]
    foreground_prob = probabilities[1:].sum(dim=0).numpy()
    foreground_mask = (foreground_prob >= threshold).astype(np.uint8)
    pred_class = probabilities.argmax(dim=0).to(torch.uint8).numpy()
    return {
        "pred_prob": foreground_prob,
        "pred": foreground_mask,
        "pred_class": pred_class,
        "pred_shell": (pred_class == 1).astype(np.uint8),
        "pred_interior": (pred_class == 2).astype(np.uint8),
    }


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
    if args.invert_contrast:
        patch_f = 1.0 - patch_f
        print(
            f"  Inverted contrast range: [{patch_f.min():.3f}, {patch_f.max():.3f}] "
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
    segmentation_mode = resolve_segmentation_mode(args.segmentation_mode, ckpt)
    print(f"  Segmentation mode: {segmentation_mode}")

    model = build_model(device, segmentation_mode)
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

    outputs = postprocess_logits(pred_logits_cpu, segmentation_mode, args.threshold)
    del pred_logits_cpu

    save_nii(patch_f, output_dir / f"{output_prefix}_input.nii.gz", args.voxel_size)
    save_nii(outputs["pred_prob"], output_dir / f"{output_prefix}_pred_prob.nii.gz", args.voxel_size)
    save_nii(outputs["pred"], output_dir / f"{output_prefix}_pred.nii.gz", args.voxel_size)
    if segmentation_mode == "three_class_shell_interior":
        save_nii(
            outputs["pred_class"],
            output_dir / f"{output_prefix}_pred_class.nii.gz",
            args.voxel_size,
        )
        save_nii(
            outputs["pred_shell"],
            output_dir / f"{output_prefix}_pred_shell.nii.gz",
            args.voxel_size,
        )
        save_nii(
            outputs["pred_interior"],
            output_dir / f"{output_prefix}_pred_interior.nii.gz",
            args.voxel_size,
        )

    print(f"\nDone! Saved to {output_dir}/")
    print(
        f"{output_dir}/{output_prefix}_input.nii.gz "
        f"{output_dir}/{output_prefix}_pred.nii.gz"
    )


if __name__ == "__main__":
    main()
