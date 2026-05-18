"""Export a timed smoke comparison between current and hollowing 3-class targets.

This script synthesizes one clean/aligned base sample once, then derives two
3-class targets from the same geometry:

* current   : 6-neighbor shell/interior rule
* hollowing : mentor-style hollow-tube wall/interior rule

It writes one image/gt/gt_class triplet per method and stores per-method timing
plus shared synthesis timing in summary.json.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import cornucopia as cc
import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datagen.axon_image_controlled_contrast import ControlledContrastAxonImage
from datagen.axon_subset_dataset import AxonSubsetDataset, build_shell_interior_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Export one current-method sample and one hollowing-method sample with timings.'
    )
    parser.add_argument('--label_dir', required=True, help='Directory with *_label.nii.gz volumes')
    parser.add_argument('--output_dir', required=True, help='Where to save NIfTI outputs')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Validation sample index to reproduce before method comparison')
    parser.add_argument('--val_fraction', type=float, default=0.2,
                        help='Validation fraction used for the deterministic val split')
    parser.add_argument('--num_samples_per_volume', type=int, default=8,
                        help='Dataset num_samples_per_volume used for deterministic sampling')
    parser.add_argument('--max_volumes', type=int, default=1,
                        help='Optional cap on source volumes for faster smoke runs')
    parser.add_argument('--crop_size', type=int, default=128,
                        help='Crop around foreground before synthesis; <= 0 uses the full sample')
    parser.add_argument('--seed', type=int, default=0,
                        help='Base random seed for deterministic synthesis and hollowing')
    parser.add_argument('--wall_mode', choices=['thin', 'thick'], default='thin',
                        help='Hollowing-rule wall thickness preset')
    parser.add_argument('--background', type=float, default=0.5)
    parser.add_argument('--fibers_lower_lo', type=float, default=0.3)
    parser.add_argument('--fibers_lower_hi', type=float, default=0.5)
    parser.add_argument('--bg_upper_lo', type=float, default=0.2)
    parser.add_argument('--bg_upper_hi', type=float, default=0.4)
    parser.add_argument('--debug_progress', action='store_true',
                        help='Print progress markers and enable synth-stage instrumentation')
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def progress(enabled: bool, t0: float, message: str) -> None:
    if enabled:
        print(f'[smoke-v7 +{time.monotonic() - t0:7.2f}s] {message}', flush=True)


def save_nii(array: np.ndarray, path: Path) -> None:
    nib.save(nib.Nifti1Image(array, affine=np.eye(4)), str(path))


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


def build_hollowing_target(labels: torch.Tensor, wall_mode: str, seed: int) -> torch.Tensor:
    set_seed(seed)
    shape = int(max(labels.shape[-3:]))
    thin_wall = wall_mode == 'thin'
    shallow = cc.RandomSmoothShallowLabelTransform(
        labels=0.95,
        shape=shape,
        max_width=cc.RandInt(1, 2) if thin_wall else cc.RandInt(2, 3),
        min_width=cc.RandInt(1, 1) if thin_wall else cc.RandInt(1, 2),
        background_labels=0,
    )

    source = labels.clone()
    wall = shallow(source)
    for _ in range(10):
        if wall.any():
            break
        wall = shallow(source)
    else:
        wall = source

    target = torch.zeros_like(source, dtype=torch.long)
    wall_mask = wall > 0
    interior_mask = (source > 0) & ~wall_mask
    target[wall_mask] = 1
    target[interior_mask] = 2
    return target


def summarize_target(target: np.ndarray) -> dict:
    foreground_voxels = int(np.sum(target > 0))
    shell_voxels = int(np.sum(target == 1))
    interior_voxels = int(np.sum(target == 2))
    return {
        'foreground_voxels': foreground_voxels,
        'shell_voxels': shell_voxels,
        'interior_voxels': interior_voxels,
        'shell_fraction_of_foreground': float(shell_voxels / foreground_voxels) if foreground_voxels else 0.0,
        'interior_fraction_of_foreground': float(interior_voxels / foreground_voxels) if foreground_voxels else 0.0,
    }


def export_method_sample(
    prefix: Path,
    image_np: np.ndarray,
    gt_class_np: np.ndarray,
) -> float:
    t0 = time.monotonic()
    gt_np = (gt_class_np > 0).astype(np.uint8)
    save_nii(image_np, Path(f'{prefix}_image.nii.gz'))
    save_nii(gt_np, Path(f'{prefix}_gt.nii.gz'))
    save_nii(gt_class_np.astype(np.uint8), Path(f'{prefix}_gt_class.nii.gz'))
    return time.monotonic() - t0


def main() -> None:
    args = parse_args()
    t0 = time.monotonic()
    if args.debug_progress:
        os.environ['AXON_DATASET_PROGRESS'] = '1'
        os.environ['AXON_SYNTH_PROGRESS'] = '1'

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress(args.debug_progress, t0, f'output_dir={output_dir}')

    set_seed(args.seed)
    progress(args.debug_progress, t0, 'loading raw validation sample')
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
    sample = dataset[args.sample_idx]
    label = sample['label']
    prob = sample['prob']
    label, prob, crop_bbox = crop_around_foreground(label, prob, args.crop_size)
    progress(args.debug_progress, t0, f'cropped sample_idx={args.sample_idx} shape={tuple(label.shape)}')

    synth = ControlledContrastAxonImage.XForm(
        background=args.background,
        fibers_lower_range=(args.fibers_lower_lo, args.fibers_lower_hi),
        background_upper_range=(args.bg_upper_lo, args.bg_upper_hi),
        clean_target_lab=True,
    )

    progress(args.debug_progress, t0, 'shared synthesis start')
    synth_t0 = time.monotonic()
    with torch.no_grad():
        image, transformed_prob, transformed_label = synth(label, prob, label)
    shared_synth_elapsed_s = time.monotonic() - synth_t0
    progress(args.debug_progress, t0, f'shared synthesis done elapsed={shared_synth_elapsed_s:.2f}s')

    clean_label = transformed_label.clone()
    clean_label[transformed_prob <= 0] = 0
    clean_label_np = clean_label.squeeze(0).cpu().numpy().astype(np.int64)
    image_np = image.squeeze(0).cpu().numpy().astype(np.float32)

    progress(args.debug_progress, t0, 'building current target')
    current_t0 = time.monotonic()
    current_target_np = build_shell_interior_target(clean_label_np).astype(np.uint8)
    current_method_elapsed_s = time.monotonic() - current_t0

    progress(args.debug_progress, t0, 'building hollowing target')
    hollowing_t0 = time.monotonic()
    hollowing_target = build_hollowing_target(clean_label, wall_mode=args.wall_mode, seed=args.seed)
    hollowing_target_np = hollowing_target.squeeze(0).cpu().numpy().astype(np.uint8)
    hollowing_method_elapsed_s = time.monotonic() - hollowing_t0

    current_prefix = output_dir / 'sample_000_current'
    hollowing_prefix = output_dir / 'sample_001_hollowing'

    progress(args.debug_progress, t0, 'saving current-method sample')
    current_save_elapsed_s = export_method_sample(current_prefix, image_np, current_target_np)
    progress(args.debug_progress, t0, 'saving hollowing-method sample')
    hollowing_save_elapsed_s = export_method_sample(hollowing_prefix, image_np, hollowing_target_np)

    summary = {
        'sample_idx': args.sample_idx,
        'seed': args.seed,
        'wall_mode': args.wall_mode,
        'crop_bbox': crop_bbox,
        'source_mode': 'shared_manual_synthesis',
        'shared_synthesis_elapsed_s': shared_synth_elapsed_s,
        'exports': [
            {
                'prefix': current_prefix.name,
                'method': 'current_shell_interior',
                'method_elapsed_s': current_method_elapsed_s,
                'save_elapsed_s': current_save_elapsed_s,
                'total_incremental_elapsed_s': current_method_elapsed_s + current_save_elapsed_s,
                'total_including_shared_synthesis_elapsed_s': (
                    shared_synth_elapsed_s + current_method_elapsed_s + current_save_elapsed_s
                ),
                'target_summary': summarize_target(current_target_np),
            },
            {
                'prefix': hollowing_prefix.name,
                'method': 'hollowing',
                'method_elapsed_s': hollowing_method_elapsed_s,
                'save_elapsed_s': hollowing_save_elapsed_s,
                'total_incremental_elapsed_s': hollowing_method_elapsed_s + hollowing_save_elapsed_s,
                'total_including_shared_synthesis_elapsed_s': (
                    shared_synth_elapsed_s + hollowing_method_elapsed_s + hollowing_save_elapsed_s
                ),
                'target_summary': summarize_target(hollowing_target_np),
            },
        ],
    }

    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2)

    progress(args.debug_progress, t0, f'done total_elapsed={time.monotonic() - t0:.2f}s')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()