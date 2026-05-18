"""Benchmark the dedicated GPU cache-builder path on a small sample set."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datagen.axon_subset_dataset import AxonSubsetDataset
from datagen.gpu_cache_builder import build_gpu_tensor_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Benchmark the dedicated GPU cache-builder path.')
    parser.add_argument('--label_dir', required=True, help='Directory with *_label.nii.gz volumes')
    parser.add_argument('--split', choices=['train', 'val'], default='val')
    parser.add_argument('--segmentation_mode', choices=['binary', 'three_class_shell_interior'],
                        default='three_class_shell_interior')
    parser.add_argument('--n_samples', type=int, default=1, help='Number of cached samples to build')
    parser.add_argument('--num_samples_per_volume', type=int, default=8)
    parser.add_argument('--max_volumes', type=int, default=1)
    parser.add_argument('--val_fraction', type=float, default=0.2)
    parser.add_argument('--subset_fraction_lo', type=float, default=0.3)
    parser.add_argument('--subset_fraction_hi', type=float, default=0.9)
    parser.add_argument('--gpu_label_block_size', type=int, default=8)
    parser.add_argument('--output_json', default=None, help='Optional summary JSON path')
    parser.add_argument('--output_dir', default=None,
                        help='Optional directory where image/gt/gt_class NIfTI files are saved')
    return parser.parse_args()


def save_nii(array: np.ndarray, path: Path) -> None:
    nib.save(nib.Nifti1Image(array, affine=np.eye(4)), str(path))


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    log = logging.getLogger('gpu-cache-bench')

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for benchmark_gpu_cache_builder.py')
    device = torch.device('cuda')

    dataset = AxonSubsetDataset(
        label_dir=args.label_dir,
        subset_fraction=(args.subset_fraction_lo, args.subset_fraction_hi),
        apply_density_curve=True,
        generate_images=False,
        num_samples_per_volume=args.num_samples_per_volume,
        max_volumes=args.max_volumes,
        segmentation_mode=args.segmentation_mode,
        split=args.split,
        val_fraction=args.val_fraction,
    )

    images, segs, metrics = build_gpu_tensor_cache(
        dataset,
        split=args.split,
        device=device,
        log=log,
        max_samples=args.n_samples,
        gpu_label_block_size=args.gpu_label_block_size,
    )

    saved_samples = []
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for sample_index in range(int(images.shape[0])):
            image_np = images[sample_index, 0].cpu().numpy().astype(np.float32)
            gt_class_np = segs[sample_index, 0].cpu().numpy().astype(np.uint8)
            gt_np = (gt_class_np > 0).astype(np.uint8)
            prefix = output_dir / f'sample_{sample_index:03d}'
            save_nii(image_np, Path(f'{prefix}_image.nii.gz'))
            save_nii(gt_np, Path(f'{prefix}_gt.nii.gz'))
            save_nii(gt_class_np, Path(f'{prefix}_gt_class.nii.gz'))
            saved_samples.append({
                'sample_index': sample_index,
                'image_path': str(Path(f'{prefix}_image.nii.gz')),
                'gt_path': str(Path(f'{prefix}_gt.nii.gz')),
                'gt_class_path': str(Path(f'{prefix}_gt_class.nii.gz')),
            })

    summary = {
        'label_dir': args.label_dir,
        'split': args.split,
        'segmentation_mode': args.segmentation_mode,
        'n_samples': int(images.shape[0]),
        'image_shape': list(images.shape),
        'seg_shape': list(segs.shape),
        'metrics': metrics,
        'output_dir': args.output_dir,
        'saved_samples': saved_samples,
    }
    print(json.dumps(summary, indent=2))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()