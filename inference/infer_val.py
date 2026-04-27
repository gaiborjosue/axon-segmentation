"""
Inference on validation split — saves image, prediction, and ground truth as NIfTI.

Usage
-----
    python infer_val.py \
        --checkpoint /scratch/experiment/draft/training_out/checkpoints/best_model.pt \
        --label_dir  /scratch/experiment/draft/dense_labels \
        --output_dir /scratch/experiment/draft/inference_out

Outputs per sample
------------------
    <output_dir>/
        sample_000_image.nii.gz      — synthesised input image
        sample_000_pred.nii.gz       — binary prediction (threshold 0.5)
        sample_000_pred_prob.nii.gz  — raw sigmoid probability map
        sample_000_gt.nii.gz         — ground truth segmentation
        ...
"""

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.networks.layers import Norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datagen import create_dataloader


def parse_args():
    p = argparse.ArgumentParser(description='Run inference on val split and save NIfTI outputs')
    p.add_argument('--checkpoint',      required=True,  help='Path to best_model.pt')
    p.add_argument('--label_dir',       required=True,  help='Directory with *_label.nii.gz volumes')
    p.add_argument('--output_dir',      required=True,  help='Where to save NIfTI outputs')
    p.add_argument('--segmentation_mode', default='auto',
                   choices=['auto', 'binary', 'three_class_shell_interior'],
                   help="Segmentation mode. 'auto' uses the checkpoint args when available.")
    p.add_argument('--val_fraction',    type=float, default=0.34)
    p.add_argument('--n_samples',       type=int,   default=5,
                   help='Number of val samples to run inference on')
    p.add_argument('--roi_size',        type=int,   default=128)
    p.add_argument('--sw_batch_size',   type=int,   default=4)
    p.add_argument('--n_label_groups',  type=int,   default=8)
    p.add_argument('--threshold',       type=float, default=0.5)
    # Synthesis params (should match training)
    p.add_argument('--background',      type=float, default=0.5)
    p.add_argument('--fibers_lower_lo', type=float, default=0.3)
    p.add_argument('--fibers_lower_hi', type=float, default=0.5)
    p.add_argument('--bg_upper_lo',     type=float, default=0.2)
    p.add_argument('--bg_upper_hi',     type=float, default=0.4)
    return p.parse_args()


def save_nii(arr: np.ndarray, path: Path):
    """Save a float32 or uint8 numpy array as NIfTI."""
    nib.save(nib.Nifti1Image(arr, affine=np.eye(4)), str(path))


def resolve_segmentation_mode(requested_mode: str, checkpoint: dict) -> str:
    if requested_mode != 'auto':
        return requested_mode
    checkpoint_args = checkpoint.get('args') or {}
    return checkpoint_args.get('segmentation_mode', 'binary')


def build_model(device: torch.device, segmentation_mode: str) -> UNet:
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1 if segmentation_mode == 'binary' else 3,
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
    if segmentation_mode == 'binary':
        pred_prob = torch.sigmoid(pred_logits)[0, 0].cpu().float().numpy()
        pred_mask = (pred_prob >= threshold).astype(np.uint8)
        return {
            'pred_prob': pred_prob,
            'pred': pred_mask,
        }

    probabilities = torch.softmax(pred_logits, dim=1)[0].cpu().float()
    foreground_prob = probabilities[1:].sum(dim=0).numpy()
    foreground_mask = (foreground_prob >= threshold).astype(np.uint8)
    pred_class = probabilities.argmax(dim=0).to(torch.uint8).numpy()
    return {
        'pred_prob': foreground_prob,
        'pred': foreground_mask,
        'pred_class': pred_class,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # --- Load checkpoint ---
    ckpt = torch.load(args.checkpoint, map_location=device)
    metric_name = ckpt.get('val_metric_name', 'val_dice')
    metric_value = ckpt.get('val_metric_value', ckpt.get('val_dice', float('nan')))
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}, {metric_name}={metric_value:.4f}")
    segmentation_mode = resolve_segmentation_mode(args.segmentation_mode, ckpt)
    print(f'Segmentation mode: {segmentation_mode}')

    # Rebuild model with same architecture as train.py
    model = build_model(device, segmentation_mode)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'Model loaded: {sum(p.numel() for p in model.parameters()):,} params')

    # --- Val DataLoader (batch_size=1, fixed seed for reproducibility) ---
    val_loader = create_dataloader(
        label_dir=args.label_dir,
        batch_size=1,
        num_workers=2,
        split='val',
        generate_images=True,
        num_samples_per_volume=args.n_samples,
        val_fraction=args.val_fraction,
        fibers_lower_range=(args.fibers_lower_lo, args.fibers_lower_hi),
        background_upper_range=(args.bg_upper_lo, args.bg_upper_hi),
        background=args.background,
        segmentation_mode=segmentation_mode,
    )

    roi_size = (args.roi_size,) * 3

    print(f'Running inference on {args.n_samples} samples → {output_dir}')

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= args.n_samples:
                break

            image = batch['image'].to(device)
            seg = batch['seg'].to(device)

            # Inference
            pred_logits = sliding_window_inference(
                image, roi_size, args.sw_batch_size, model
            )
            outputs = postprocess_logits(pred_logits, segmentation_mode, args.threshold)

            # Convert to numpy (D, H, W)
            img_np    = image[0, 0].cpu().float().numpy()
            prob_np   = outputs['pred_prob']
            pred_np   = outputs['pred']
            if segmentation_mode == 'binary':
                gt_np = seg[0, 0].cpu().float().numpy()
            else:
                gt_np = (seg[0, 0] > 0).cpu().float().numpy()
                gt_class_np = seg[0, 0].cpu().numpy().astype(np.uint8)

            prefix = output_dir / f'sample_{i:03d}'
            save_nii(img_np,  Path(f'{prefix}_image.nii.gz'))
            save_nii(prob_np, Path(f'{prefix}_pred_prob.nii.gz'))
            save_nii(pred_np, Path(f'{prefix}_pred.nii.gz'))
            save_nii(gt_np,   Path(f'{prefix}_gt.nii.gz'))
            if segmentation_mode == 'three_class_shell_interior':
                save_nii(outputs['pred_class'], Path(f'{prefix}_pred_class.nii.gz'))
                save_nii(gt_class_np, Path(f'{prefix}_gt_class.nii.gz'))

            print(f'  [{i+1}/{args.n_samples}] saved {prefix.name}_*.nii.gz')

    print(f'\nDone. Open in ITK-SNAP or napari:')
    print(f'  napari {output_dir}/sample_000_image.nii.gz')


if __name__ == '__main__':
    main()
