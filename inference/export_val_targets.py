"""Export validation images and ground truth targets without model predictions.

This is a lightweight smoke/export utility for inspecting synthetic val samples.

Two geometry modes are supported:
    clean : disable the noisy hard-label perturbation inside XForm so image and
            GT come from a cleaner shared geometry stage.
    live  : keep the current live hard-label perturbation path.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datagen.axon_image_controlled_contrast import ControlledContrastAxonImage
from datagen.axon_subset_dataset import AxonSubsetDataset, build_shell_interior_target, create_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Export validation image/gt/gt_class samples without predictions.'
    )
    parser.add_argument('--label_dir', required=True, help='Directory with *_label.nii.gz volumes')
    parser.add_argument('--output_dir', required=True, help='Where to save NIfTI outputs')
    parser.add_argument('--n_samples', type=int, default=3, help='Number of samples to export')
    parser.add_argument('--sample_start', type=int, default=0, help='Starting val sample index')
    parser.add_argument('--val_fraction', type=float, default=0.34, help='Validation fraction')
    parser.add_argument('--num_samples_per_volume', type=int, default=8,
                        help='Dataset num_samples_per_volume used for deterministic sampling')
    parser.add_argument('--max_volumes', type=int, default=None,
                        help='Optional cap on source volumes for faster smoke runs')
    parser.add_argument('--crop_size', type=int, default=128,
                        help='Crop around foreground before synthesis; <= 0 uses the full sample')
    parser.add_argument('--source_mode', choices=['dataset', 'manual'], default='dataset',
                        help='dataset uses the patched AxonSubsetDataset image/GT path; '
                             'manual re-synthesizes from raw label/prob pairs')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of DataLoader workers in dataset source mode')
    parser.add_argument('--debug_progress', action='store_true',
                        help='Print exporter and dataset progress markers for debugging stalls')
    parser.add_argument('--geometry_mode', choices=['clean', 'live'], default='clean',
                        help='clean disables noisy hard-label perturbation; live keeps the current path')
    parser.add_argument('--seed', type=int, default=0, help='Base random seed for reproducible synthesis')
    parser.add_argument('--background', type=float, default=0.5)
    parser.add_argument('--fibers_lower_lo', type=float, default=0.3)
    parser.add_argument('--fibers_lower_hi', type=float, default=0.5)
    parser.add_argument('--bg_upper_lo', type=float, default=0.2)
    parser.add_argument('--bg_upper_hi', type=float, default=0.4)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_nii(array: np.ndarray, path: Path) -> None:
    nib.save(nib.Nifti1Image(array, affine=np.eye(4)), str(path))


def progress(enabled: bool, t0: float, message: str) -> None:
    if enabled:
        print(f'[export-progress +{time.monotonic() - t0:7.2f}s] {message}', flush=True)


def crop_around_foreground(
    label: torch.Tensor,
    prob: torch.Tensor,
    crop_size: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    label_np = label.squeeze(0).cpu().numpy()
    coords = np.argwhere(label_np > 0)
    if coords.size == 0 or crop_size <= 0:
        return label, prob, {
            'start': [0, 0, 0],
            'end': [int(value) for value in label.shape[1:]],
            'shape': [int(value) for value in label.shape[1:]],
        }

    spatial_shape = np.asarray(label_np.shape)
    center = np.rint(coords.mean(axis=0)).astype(int)
    start = center - crop_size // 2
    max_start = np.maximum(spatial_shape - crop_size, 0)
    start = np.clip(start, 0, max_start)
    end = np.minimum(start + crop_size, spatial_shape)
    start = np.maximum(end - crop_size, 0)

    slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(start, end))
    cropped_label = label[(slice(None),) + slices].contiguous()
    cropped_prob = prob[(slice(None),) + slices].contiguous()
    bbox = {
        'start': [int(value) for value in start.tolist()],
        'end': [int(value) for value in end.tolist()],
        'shape': [int(value) for value in cropped_label.shape[1:]],
    }
    return cropped_label, cropped_prob, bbox


def build_synth(args: argparse.Namespace) -> ControlledContrastAxonImage.XForm:
    synth = ControlledContrastAxonImage.XForm(
        background=args.background,
        fibers_lower_range=(args.fibers_lower_lo, args.fibers_lower_hi),
        background_upper_range=(args.bg_upper_lo, args.bg_upper_hi),
    )
    if args.geometry_mode == 'clean':
        synth.noisylabel = nn.Identity()
    return synth


def main() -> None:
    args = parse_args()
    t0 = time.monotonic()
    if args.debug_progress:
        os.environ['AXON_DATASET_PROGRESS'] = '1'
    progress(args.debug_progress, t0, f'parsed args source_mode={args.source_mode}')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress(args.debug_progress, t0, f'output_dir={output_dir}')

    saved = []
    if args.source_mode == 'dataset':
        progress(args.debug_progress, t0, 'creating val DataLoader')
        val_loader = create_dataloader(
            label_dir=args.label_dir,
            batch_size=1,
            num_workers=args.num_workers,
            split='val',
            generate_images=True,
            shuffle=False,
            drop_last=False,
            persistent_workers=(args.num_workers > 0),
            val_fraction=args.val_fraction,
            num_samples_per_volume=args.num_samples_per_volume,
            max_volumes=args.max_volumes,
            segmentation_mode='three_class_shell_interior',
            background=args.background,
            fibers_lower_range=(args.fibers_lower_lo, args.fibers_lower_hi),
            background_upper_range=(args.bg_upper_lo, args.bg_upper_hi),
        )
        progress(args.debug_progress, t0, 'val DataLoader created; entering iteration')

        for sample_idx, batch in enumerate(val_loader):
            progress(args.debug_progress, t0, f'batch_received sample_idx={sample_idx}')
            if sample_idx < args.sample_start:
                progress(args.debug_progress, t0, f'skipping sample_idx={sample_idx}')
                continue
            if len(saved) >= args.n_samples:
                progress(args.debug_progress, t0, 'requested sample count reached')
                break

            image = batch['image'][0]
            seg = batch['seg'][0]
            crop_bbox = {
                'start': [0, 0, 0],
                'end': [int(value) for value in image.shape[1:]],
                'shape': [int(value) for value in image.shape[1:]],
            }
            image_np = image.squeeze(0).cpu().numpy().astype(np.float32)
            gt_class_np = seg.squeeze(0).cpu().numpy().astype(np.uint8)
            gt_np = (gt_class_np > 0).astype(np.uint8)

            prefix = output_dir / f'sample_{len(saved):03d}'
            progress(args.debug_progress, t0, f'saving {prefix.name}')
            save_nii(image_np, Path(f'{prefix}_image.nii.gz'))
            save_nii(gt_np, Path(f'{prefix}_gt.nii.gz'))
            save_nii(gt_class_np, Path(f'{prefix}_gt_class.nii.gz'))

            saved.append({
                'export_index': len(saved),
                'sample_idx': sample_idx,
                'source_mode': args.source_mode,
                'geometry_mode': args.geometry_mode,
                'crop_bbox': crop_bbox,
                'subset_info': None,
                'density_config': None,
            })
            print(f'[ok] saved sample_{len(saved)-1:03d}_*.nii.gz from sample_idx={sample_idx}')
    else:
        progress(args.debug_progress, t0, 'creating manual-source dataset')
        dataset = AxonSubsetDataset(
            label_dir=args.label_dir,
            split='val',
            val_fraction=args.val_fraction,
            generate_images=False,
            num_samples_per_volume=args.num_samples_per_volume,
            max_volumes=args.max_volumes,
            segmentation_mode='three_class_shell_interior',
            background=args.background,
            fibers_lower_range=(args.fibers_lower_lo, args.fibers_lower_hi),
            background_upper_range=(args.bg_upper_lo, args.bg_upper_hi),
        )

        synth = build_synth(args)
        progress(args.debug_progress, t0, 'manual synth created; entering sample loop')
        sample_idx = args.sample_start

        while len(saved) < args.n_samples and sample_idx < len(dataset):
            set_seed(args.seed + sample_idx)
            progress(args.debug_progress, t0, f'loading manual sample_idx={sample_idx}')
            sample = dataset[sample_idx]
            label = sample['label']
            prob = sample['prob']
            label, prob, crop_bbox = crop_around_foreground(label, prob, args.crop_size)
            progress(args.debug_progress, t0, f'cropped sample_idx={sample_idx} shape={tuple(label.shape)}')

            try:
                progress(args.debug_progress, t0, f'synth_start sample_idx={sample_idx}')
                with torch.no_grad():
                    image, transformed_prob, transformed_label = synth(label, prob, label)
                progress(args.debug_progress, t0, f'synth_done sample_idx={sample_idx}')
            except Exception as exc:
                print(f'[warn] sample_idx={sample_idx}: synth failed: {exc}')
                progress(args.debug_progress, t0, f'synth_error sample_idx={sample_idx} exc={exc!r}')
                sample_idx += 1
                continue

            clean_label = transformed_label.clone()
            clean_label[transformed_prob <= 0] = 0

            image_np = image.squeeze(0).cpu().numpy().astype(np.float32)
            gt_class_np = build_shell_interior_target(
                clean_label.squeeze(0).cpu().numpy().astype(np.int64)
            ).astype(np.uint8)
            gt_np = (gt_class_np > 0).astype(np.uint8)

            prefix = output_dir / f'sample_{len(saved):03d}'
            progress(args.debug_progress, t0, f'saving {prefix.name}')
            save_nii(image_np, Path(f'{prefix}_image.nii.gz'))
            save_nii(gt_np, Path(f'{prefix}_gt.nii.gz'))
            save_nii(gt_class_np, Path(f'{prefix}_gt_class.nii.gz'))

            saved.append({
                'export_index': len(saved),
                'sample_idx': sample_idx,
                'source_mode': args.source_mode,
                'geometry_mode': args.geometry_mode,
                'crop_bbox': crop_bbox,
                'subset_info': sample.get('subset_info'),
                'density_config': sample.get('density_config'),
            })
            print(f'[ok] saved sample_{len(saved)-1:03d}_*.nii.gz from sample_idx={sample_idx}')
            sample_idx += 1

    summary = {
        'source_mode': args.source_mode,
        'geometry_mode': args.geometry_mode,
        'n_requested': args.n_samples,
        'n_saved': len(saved),
        'seed': args.seed,
        'samples': saved,
    }
    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2)

    progress(args.debug_progress, t0, f'writing summary to {output_dir / "summary.json"}')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()