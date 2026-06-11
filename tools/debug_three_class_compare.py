from pathlib import Path
import argparse
import json
import random
import sys
import time

import cornucopia as cc
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datagen.axon_image_controlled_contrast import ControlledContrastAxonImage
from datagen.axon_subset_dataset import AxonSubsetDataset, build_shell_interior_target


def log_progress(message: str, start_time: float) -> None:
    elapsed = time.monotonic() - start_time
    print(f'[{elapsed:7.2f}s] {message}', flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compare current and hollowing-rule three-class targets on the same transformed label map.'
    )
    parser.add_argument('--output_dir', required=True, help='Where to save comparison outputs')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for deterministic synthesis and hollowing rule')
    parser.add_argument('--input_mode', choices=['direct', 'dataset'], default='dataset',
                        help='Load from raw label/prob files or reproduce a dataset sample.')
    parser.add_argument('--label', default=None, help='Path to *_label.nii.gz volume when --input_mode=direct')
    parser.add_argument('--prob', default=None, help='Path to *_prob.nii.gz volume when --input_mode=direct')
    parser.add_argument('--label_dir', default=None,
                        help='Directory with *_label.nii.gz volumes when --input_mode=dataset')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Dataset sample index to reproduce when --input_mode=dataset')
    parser.add_argument('--num_samples_per_volume', type=int, default=8,
                        help='Dataset num_samples_per_volume used when --input_mode=dataset')
    parser.add_argument('--val_fraction', type=float, default=0.2,
                        help='Validation fraction used when --input_mode=dataset')
    parser.add_argument('--max_volumes', type=int, default=None,
                        help='Optional cap on dataset volumes for faster debug runs')
    parser.add_argument('--keep_ids', nargs='+', type=int, default=None,
                        help='Explicit axon IDs to keep before synthesis')
    parser.add_argument('--max_axons', type=int, default=2,
                        help='Maximum number of axon IDs to keep when --keep_ids is not set')
    parser.add_argument('--selection_mode', choices=['largest', 'random'], default='largest',
                        help='How to choose IDs when --keep_ids is not set')
    parser.add_argument('--crop_size', type=int, default=64,
                        help='Isotropic crop size centered on the kept axons; 0 uses bbox+padding')
    parser.add_argument('--bbox_pad', type=int, default=8,
                        help='Padding around the kept-axon bounding box when crop_size=0')
    parser.add_argument('--wall_mode', choices=['thin', 'thick'], default='thin',
                        help='Hollowing-rule wall thickness preset')
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_direct_sample(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    if not args.label or not args.prob:
        raise ValueError('--label and --prob are required when --input_mode=direct')
    label = np.asarray(nib.load(str(args.label)).get_fdata(), dtype=np.int64)
    prob = np.asarray(nib.load(str(args.prob)).get_fdata(), dtype=np.float32)
    return torch.from_numpy(label).unsqueeze(0).long(), torch.from_numpy(prob).unsqueeze(0).float()


def load_dataset_sample(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    if not args.label_dir:
        raise ValueError('--label_dir is required when --input_mode=dataset')
    dataset = AxonSubsetDataset(
        label_dir=args.label_dir,
        split='val',
        val_fraction=args.val_fraction,
        generate_images=False,
        num_samples_per_volume=args.num_samples_per_volume,
        max_volumes=args.max_volumes,
        segmentation_mode='three_class_shell_interior',
        background=0.5,
        fibers_lower_range=(0.3, 0.5),
        background_upper_range=(0.2, 0.4),
    )
    sample = dataset[args.sample_idx]
    return sample['label'], sample['prob']


def choose_ids(label: torch.Tensor, keep_ids: list[int] | None, max_axons: int, selection_mode: str) -> list[int]:
    label_np = label.squeeze(0).cpu().numpy()
    counts = np.bincount(label_np.ravel())
    available_ids = np.flatnonzero(counts > 0)
    available_ids = available_ids[available_ids > 0]
    if available_ids.size == 0:
        raise ValueError('No foreground axons found in the selected sample')

    if keep_ids:
        chosen = [axon_id for axon_id in keep_ids if axon_id in available_ids]
        if not chosen:
            raise ValueError('None of the requested --keep_ids exist in the selected sample')
        return chosen

    if selection_mode == 'largest':
        order = available_ids[np.argsort(counts[available_ids])[::-1]]
    else:
        order = np.random.permutation(available_ids)
    return [int(axon_id) for axon_id in order[:max_axons]]


def keep_selected_ids(label: torch.Tensor, prob: torch.Tensor, chosen_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    label_np = label.squeeze(0).cpu().numpy()
    prob_np = prob.squeeze(0).cpu().numpy()
    keep_mask = np.isin(label_np, np.asarray(chosen_ids, dtype=label_np.dtype))
    kept_label = np.where(keep_mask, label_np, 0).astype(np.int64)
    kept_prob = np.where(keep_mask, prob_np, 0).astype(np.float32)
    return torch.from_numpy(kept_label).unsqueeze(0).long(), torch.from_numpy(kept_prob).unsqueeze(0).float()


def crop_around_foreground(
    label: torch.Tensor,
    prob: torch.Tensor,
    crop_size: int,
    bbox_pad: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    label_np = label.squeeze(0).cpu().numpy()
    coords = np.argwhere(label_np > 0)
    if coords.size == 0:
        raise ValueError('Cannot crop because the selected sample has no foreground voxels')

    lower = coords.min(axis=0)
    upper = coords.max(axis=0) + 1
    spatial_shape = np.asarray(label_np.shape)

    if crop_size > 0:
        center = (lower + upper) // 2
        start = center - crop_size // 2
        max_start = np.maximum(spatial_shape - crop_size, 0)
        start = np.clip(start, 0, max_start)
        end = np.minimum(start + crop_size, spatial_shape)
        start = np.maximum(end - crop_size, 0)
    else:
        start = np.maximum(lower - bbox_pad, 0)
        end = np.minimum(upper + bbox_pad, spatial_shape)

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


def save_nifti(array: np.ndarray, path: Path) -> None:
    nib.save(nib.Nifti1Image(array, np.eye(4)), str(path))


def extract_center_views(volume: np.ndarray) -> list[np.ndarray]:
    return [
        volume[volume.shape[0] // 2, :, :],
        volume[:, volume.shape[1] // 2, :],
        volume[:, :, volume.shape[2] // 2],
    ]


def plot_views(ax_row, volume: np.ndarray, title: str, cmap: str, vmin=None, vmax=None) -> None:
    views = extract_center_views(volume)
    for ax, slc, view_name in zip(ax_row, views, ['sagittal', 'coronal', 'axial']):
        ax.imshow(slc.T, cmap=cmap, origin='lower', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.set_title(f'{title}: {view_name}')
        ax.axis('off')


def save_comparison_figure(
    image: np.ndarray,
    transformed_labels: np.ndarray,
    current_target: np.ndarray,
    hollowing_target: np.ndarray,
    disagreement: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(5, 3, figsize=(10, 13))
    plot_views(axes[0], image, 'image', 'gray')
    plot_views(axes[1], transformed_labels, 'transformed_label', 'tab20', vmin=0)
    plot_views(axes[2], current_target, 'current_target', 'viridis', vmin=0, vmax=2)
    plot_views(axes[3], hollowing_target, 'hollowing_target', 'viridis', vmin=0, vmax=2)
    plot_views(axes[4], disagreement, 'disagreement', 'magma', vmin=0, vmax=1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_geometry_comparison_figure(
    label_geometry: np.ndarray,
    current_target: np.ndarray,
    hollowing_target: np.ndarray,
    disagreement: np.ndarray,
    output_path: Path,
    title_prefix: str,
) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(10, 10))
    plot_views(axes[0], label_geometry, f'{title_prefix}_label', 'tab20', vmin=0)
    plot_views(axes[1], current_target, f'{title_prefix}_current', 'viridis', vmin=0, vmax=2)
    plot_views(axes[2], hollowing_target, f'{title_prefix}_hollowing', 'viridis', vmin=0, vmax=2)
    plot_views(axes[3], disagreement, f'{title_prefix}_disagreement', 'magma', vmin=0, vmax=1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def summarize_target(target: np.ndarray, transformed_labels: np.ndarray) -> dict:
    foreground = transformed_labels > 0
    axon_ids = [int(axon_id) for axon_id in np.unique(transformed_labels[foreground])]
    axons_without_interior = 0
    for axon_id in axon_ids:
        if not np.any((transformed_labels == axon_id) & (target == 2)):
            axons_without_interior += 1

    shell_voxels = int(np.sum(target == 1))
    interior_voxels = int(np.sum(target == 2))
    foreground_voxels = int(np.sum(foreground))
    return {
        'foreground_voxels': foreground_voxels,
        'shell_voxels': shell_voxels,
        'interior_voxels': interior_voxels,
        'shell_fraction_of_foreground': float(shell_voxels / foreground_voxels) if foreground_voxels else 0.0,
        'interior_fraction_of_foreground': float(interior_voxels / foreground_voxels) if foreground_voxels else 0.0,
        'axon_count': len(axon_ids),
        'axons_without_interior': axons_without_interior,
    }


def main() -> None:
    start_time = time.monotonic()
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    log_progress('loading sample', start_time)
    if args.input_mode == 'direct':
        label, prob = load_direct_sample(args)
    else:
        label, prob = load_dataset_sample(args)

    chosen_ids = choose_ids(label, args.keep_ids, args.max_axons, args.selection_mode)
    log_progress(f'keeping axon IDs {chosen_ids}', start_time)
    label, prob = keep_selected_ids(label, prob, chosen_ids)
    label, prob, crop_bbox = crop_around_foreground(label, prob, args.crop_size, args.bbox_pad)
    log_progress(f'cropped sample to {tuple(label.shape[1:])}', start_time)

    synth = ControlledContrastAxonImage.XForm(
        background=0.5,
        fibers_lower_range=(0.3, 0.5),
        background_upper_range=(0.2, 0.4),
    )
    log_progress('running XForm synthesis', start_time)
    with torch.no_grad():
        image, transformed_prob, transformed_label = synth(label, prob, label)
    log_progress('synthesis complete', start_time)

    masked_label = transformed_label.clone()
    masked_label[transformed_prob <= 0] = 0

    input_label_np = label.squeeze(0).cpu().numpy().astype(np.int64)
    image_np = image.squeeze(0).cpu().numpy().astype(np.float32)
    prob_np = transformed_prob.squeeze(0).cpu().numpy().astype(np.float32)
    transformed_label_np = masked_label.squeeze(0).cpu().numpy().astype(np.int64)

    input_current_target_np = build_shell_interior_target(input_label_np)
    input_hollowing_target_np = build_hollowing_target(label, args.wall_mode, args.seed).squeeze(0).cpu().numpy().astype(np.int64)
    input_disagreement_np = (input_current_target_np != input_hollowing_target_np).astype(np.uint8)

    current_target_np = build_shell_interior_target(transformed_label_np)
    hollowing_target_np = build_hollowing_target(masked_label, args.wall_mode, args.seed).squeeze(0).cpu().numpy().astype(np.int64)
    disagreement_np = (current_target_np != hollowing_target_np).astype(np.uint8)

    save_nifti(input_label_np.astype(np.int32), output_dir / 'selected_label_input.nii.gz')
    save_nifti(prob.squeeze(0).cpu().numpy().astype(np.float32), output_dir / 'selected_prob_input.nii.gz')
    save_nifti(image_np, output_dir / 'image.nii.gz')
    save_nifti(prob_np, output_dir / 'transformed_prob.nii.gz')
    save_nifti(transformed_label_np.astype(np.int32), output_dir / 'transformed_label.nii.gz')
    save_nifti(input_current_target_np.astype(np.int32), output_dir / 'input_current_target.nii.gz')
    save_nifti(input_hollowing_target_np.astype(np.int32), output_dir / 'input_hollowing_target.nii.gz')
    save_nifti(input_disagreement_np, output_dir / 'input_target_disagreement.nii.gz')
    save_nifti(current_target_np.astype(np.int32), output_dir / 'current_target.nii.gz')
    save_nifti(hollowing_target_np.astype(np.int32), output_dir / 'hollowing_target.nii.gz')
    save_nifti(disagreement_np, output_dir / 'target_disagreement.nii.gz')
    save_geometry_comparison_figure(
        input_label_np,
        input_current_target_np,
        input_hollowing_target_np,
        input_disagreement_np,
        output_dir / 'comparison_input_geometry.png',
        'input',
    )
    save_comparison_figure(
        image_np,
        transformed_label_np,
        current_target_np,
        hollowing_target_np,
        disagreement_np,
        output_dir / 'comparison_live_geometry.png',
    )
    save_comparison_figure(
        image_np,
        transformed_label_np,
        current_target_np,
        hollowing_target_np,
        disagreement_np,
        output_dir / 'comparison.png',
    )

    summary = {
        'seed': args.seed,
        'input_mode': args.input_mode,
        'selected_ids': chosen_ids,
        'crop_bbox': crop_bbox,
        'wall_mode': args.wall_mode,
        'input_geometry': {
            'current': summarize_target(input_current_target_np, input_label_np),
            'hollowing': summarize_target(input_hollowing_target_np, input_label_np),
            'disagreement_voxels': int(input_disagreement_np.sum()),
        },
        'live_geometry': {
            'current': summarize_target(current_target_np, transformed_label_np),
            'hollowing': summarize_target(hollowing_target_np, transformed_label_np),
            'disagreement_voxels': int(disagreement_np.sum()),
        },
    }
    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2)

    print(json.dumps(summary, indent=2))
    log_progress(f'saved comparison outputs to {output_dir}', start_time)


if __name__ == '__main__':
    main()