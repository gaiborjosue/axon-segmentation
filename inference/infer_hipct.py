"""
Run axon segmentation on the HiP-CT patch.

Loads a .npy uint16 volume, normalises to [0, 1], runs sliding-window
inference with the trained 3D UNet, and saves predictions as NIfTI.

Usage
-----
    python infer_hipct.py \
        --input      /scratch/experiment/hipct/patch_I74_IC_zoom01.npy \
        --checkpoint /scratch/experiment/training_out/checkpoints/best_model.pt \
        --output_dir /scratch/experiment/hipct/inference_out \
        --voxel_size 0.857

Outputs
-------
    <output_dir>/
        hipct_input.nii.gz        — normalised input volume
        hipct_pred.nii.gz         — binary segmentation
        hipct_pred_prob.nii.gz    — sigmoid probability map
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.networks.layers import Norm


def parse_args():
    p = argparse.ArgumentParser(description="Axon segmentation on HiP-CT patch")
    p.add_argument("--input", required=True, help="Path to .npy uint16 patch")
    p.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    p.add_argument("--output_dir", required=True, help="Output directory for NIfTI files")
    p.add_argument("--voxel_size", type=float, default=0.857,
                   help="Isotropic voxel size in µm (for NIfTI affine)")
    p.add_argument("--roi_size", type=int, default=128, help="Sliding window ROI size")
    p.add_argument("--sw_batch_size", type=int, default=4, help="Sliding window batch size")
    p.add_argument("--overlap", type=float, default=0.5, help="Sliding window overlap")
    p.add_argument("--threshold", type=float, default=0.8, help="Binarisation threshold")
    p.add_argument("--norm_mode", default="percentile",
                   choices=["percentile", "mean_shift", "clahe"],
                   help="Intensity normalisation strategy:\n"
                        "  percentile  — clip [p0.5, p99.5] → [0,1] (original)\n"
                        "  mean_shift  — shift mean → 0.2 (background level in training)\n"
                        "  clahe       — local contrast normalisation then shift")
    return p.parse_args()


def save_nii(arr: np.ndarray, path: Path, voxel_size: float):
    """Save numpy array as NIfTI with isotropic voxel-size affine."""
    affine = np.diag([voxel_size, voxel_size, voxel_size, 1.0])
    nib.save(nib.Nifti1Image(arr, affine=affine), str(path))


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load patch ---
    print(f"Loading {args.input} ...")
    input_path = Path(args.input)
    if input_path.suffix == ".npy":
        patch = np.load(args.input)
    else:
        # Raw memmap with sidecar .json metadata
        meta = json.loads(Path(str(input_path) + ".json").read_text())
        patch = np.memmap(str(input_path), dtype=meta["dtype"],
                          mode="r", shape=tuple(meta["shape"]))
    print(f"  Shape: {patch.shape}, dtype: {patch.dtype}")

    # Normalise uint16 → float32 using chosen strategy
    print(f"  Intensity range: [{patch.min()}, {patch.max()}]")
    print(f"  Norm mode: {args.norm_mode}")
    patch_f = patch.astype(np.float32)

    if args.norm_mode == "percentile":
        # Original: clip [p0.5, p99.5] → [0, 1]
        p_lo, p_hi = np.percentile(patch_f, [0.5, 99.5])
        print(f"  Percentile clip: [{p_lo:.1f}, {p_hi:.1f}]")
        patch_f = np.clip(patch_f, p_lo, p_hi)
        patch_f = (patch_f - p_lo) / (p_hi - p_lo + 1e-8)

    elif args.norm_mode == "mean_shift":
        # Shift mean to 0.2 (background level in training distribution)
        # so the real data occupies the same intensity range the model was trained on.
        p_lo, p_hi = np.percentile(patch_f, [0.5, 99.5])
        patch_f = np.clip(patch_f, p_lo, p_hi)
        patch_f = (patch_f - p_lo) / (p_hi - p_lo + 1e-8)  # → [0, 1]
        mu = float(patch_f.mean())
        # Shift so the mean sits at 0.2 (training background centre)
        patch_f = patch_f - mu + 0.2
        patch_f = np.clip(patch_f, 0.0, 1.0)
        print(f"  Mean before shift: {mu:.3f}  → shifted to 0.2")

    elif args.norm_mode == "clahe":
        # Per-slice CLAHE to boost local axon/background contrast,
        # then apply mean_shift so the global distribution matches training.
        from skimage.exposure import equalize_adapthist
        p_lo, p_hi = np.percentile(patch_f, [0.5, 99.5])
        patch_f = np.clip(patch_f, p_lo, p_hi)
        patch_f = (patch_f - p_lo) / (p_hi - p_lo + 1e-8)  # → [0, 1]
        print("  Applying CLAHE per Z-slice ...")
        for z in range(patch_f.shape[2]):
            patch_f[:, :, z] = equalize_adapthist(
                patch_f[:, :, z], kernel_size=64, clip_limit=0.02
            )
        mu = float(patch_f.mean())
        patch_f = patch_f - mu + 0.2
        patch_f = np.clip(patch_f, 0.0, 1.0)
        print(f"  CLAHE done. Mean → shifted to 0.2")

    print(f"  Normalised range: [{patch_f.min():.3f}, {patch_f.max():.3f}]  mean={patch_f.mean():.3f}")

    tensor = torch.from_numpy(patch_f).unsqueeze(0).unsqueeze(0).to(device)
    print(f"  Tensor: {tensor.shape}, range [{tensor.min():.3f}, {tensor.max():.3f}]")

    # --- Load model ---
    print(f"Loading checkpoint {args.checkpoint} ...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    print(f"  Epoch {ckpt['epoch']}, val Dice {ckpt['val_dice']:.4f}")

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
        dropout=0.1,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # --- Inference ---
    roi_size = (args.roi_size,) * 3

    print(f"Running sliding-window inference "
          f"(roi={args.roi_size}, overlap={args.overlap}, sw_batch={args.sw_batch_size}) ...")

    with torch.no_grad(), torch.amp.autocast("cuda"):
        pred_logits = sliding_window_inference(
            tensor, roi_size, args.sw_batch_size, model,
            overlap=args.overlap, mode="gaussian",
        )

    # Move to CPU and free GPU memory before post-processing
    del tensor
    pred_logits_cpu = pred_logits.cpu()
    del pred_logits
    torch.cuda.empty_cache()

    prob_np = torch.sigmoid(pred_logits_cpu)[0, 0].numpy()
    del pred_logits_cpu
    pred_np = (prob_np >= args.threshold).astype(np.uint8)

    save_nii(patch_f, output_dir / "hipct_input.nii.gz", args.voxel_size)

    save_nii(prob_np, output_dir / "hipct_pred_prob.nii.gz", args.voxel_size)

    save_nii(pred_np, output_dir / "hipct_pred.nii.gz", args.voxel_size)

    print(f"\nDone! Saved to {output_dir}/")
    print(f"{output_dir}/hipct_input.nii.gz {output_dir}/hipct_pred.nii.gz")


if __name__ == "__main__":
    main()
