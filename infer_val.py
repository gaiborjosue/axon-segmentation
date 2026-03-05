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
from monai.transforms import Activations, AsDiscrete, Compose

sys.path.insert(0, str(Path(__file__).parent))
from datagen import create_dataloader
from datagen.axon_image_controlled_contrast import ControlledContrastAxonImage
from train import collapse_labels


def parse_args():
    p = argparse.ArgumentParser(description='Run inference on val split and save NIfTI outputs')
    p.add_argument('--checkpoint',      required=True,  help='Path to best_model.pt')
    p.add_argument('--label_dir',       required=True,  help='Directory with *_label.nii.gz volumes')
    p.add_argument('--output_dir',      required=True,  help='Where to save NIfTI outputs')
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


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # --- Load checkpoint ---
    ckpt = torch.load(args.checkpoint, map_location=device)
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}, val Dice={ckpt['val_dice']:.4f}")

    # Rebuild model with same architecture
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'Model loaded: {sum(p.numel() for p in model.parameters()):,} params')

    # --- Synthesizer ---
    synth = ControlledContrastAxonImage(
        background=args.background,
        fibers_lower_range=(args.fibers_lower_lo, args.fibers_lower_hi),
        background_upper_range=(args.bg_upper_lo, args.bg_upper_hi),
    )

    # --- Val DataLoader (batch_size=1, fixed seed for reproducibility) ---
    val_loader = create_dataloader(
        label_dir=args.label_dir,
        batch_size=1,
        num_workers=2,
        split='val',
        generate_images=False,
        num_samples_per_volume=args.n_samples,
        val_fraction=args.val_fraction,
        fibers_lower_range=(args.fibers_lower_lo, args.fibers_lower_hi),
        background_upper_range=(args.bg_upper_lo, args.bg_upper_hi),
        background=args.background,
    )

    roi_size = (args.roi_size,) * 3
    post_sigmoid = Activations(sigmoid=True)
    post_binary  = AsDiscrete(threshold=args.threshold)

    print(f'Running inference on {args.n_samples} samples → {output_dir}')

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= args.n_samples:
                break

            label = batch['label'].to(device)
            prob  = batch['prob'].to(device)

            label_g = collapse_labels(label, n_groups=args.n_label_groups)
            image, seg = synth(label_g, prob)

            # Inference
            pred_logits = sliding_window_inference(
                image, roi_size, args.sw_batch_size, model
            )
            pred_prob   = post_sigmoid(pred_logits)
            pred_binary = post_binary(pred_prob)

            # Convert to numpy (D, H, W)
            img_np    = image[0, 0].cpu().float().numpy()
            prob_np   = pred_prob[0, 0].cpu().float().numpy()
            pred_np   = pred_binary[0, 0].cpu().float().numpy().astype(np.uint8)
            gt_np     = seg[0, 0].cpu().float().numpy()

            prefix = output_dir / f'sample_{i:03d}'
            save_nii(img_np,  Path(f'{prefix}_image.nii.gz'))
            save_nii(prob_np, Path(f'{prefix}_pred_prob.nii.gz'))
            save_nii(pred_np, Path(f'{prefix}_pred.nii.gz'))
            save_nii(gt_np,   Path(f'{prefix}_gt.nii.gz'))

            print(f'  [{i+1}/{args.n_samples}] saved {prefix.name}_*.nii.gz')

    print(f'\nDone. Open in ITK-SNAP or napari:')
    print(f'  napari {output_dir}/sample_000_image.nii.gz')


if __name__ == '__main__':
    main()
