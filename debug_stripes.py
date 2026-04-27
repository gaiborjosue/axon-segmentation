from pathlib import Path
import argparse
import random
import time

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn

from datagen.axon_subset_dataset import AxonSubsetDataset
from datagen.axon_image_controlled_contrast import ControlledContrastAxonImage


class Identity(nn.Module):
    def forward(self, x):
        return x


def log_progress(message: str, start_time: float) -> None:
    elapsed = time.monotonic() - start_time
    print(f'[{elapsed:7.2f}s] {message}', flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate before/after stripe debug samples.')
    parser.add_argument('--label', default=None, help='Path to *_label.nii.gz volume')
    parser.add_argument('--prob', default=None, help='Path to *_prob.nii.gz volume')
    parser.add_argument('--output_dir', required=True, help='Where to save debug outputs')
    parser.add_argument('--crop_size', type=int, default=64, help='Isotropic crop size from the volume corner')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for deterministic synthesis')
    parser.add_argument('--input_mode', choices=['direct', 'dataset'], default='direct',
                        help='Use a raw label/prob volume crop or reproduce a deterministic dataset sample.')
    parser.add_argument('--label_dir', default=None,
                        help='Required when --input_mode=dataset; directory with *_label.nii.gz volumes.')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Dataset sample index to reproduce when --input_mode=dataset.')
    parser.add_argument('--num_samples_per_volume', type=int, default=8,
                        help='Dataset num_samples_per_volume used when --input_mode=dataset.')
    parser.add_argument('--val_fraction', type=float, default=0.2,
                        help='Validation fraction used when --input_mode=dataset.')
    parser.add_argument('--dataset_crop_size', type=int, default=0,
                        help='Optional isotropic crop applied after loading a dataset sample; 0 keeps the full volume.')
    parser.add_argument('--backend', choices=['xform', 'wrapper'], default='xform',
                        help='Whether to synthesize with the inner XForm directly or the AutoBatchTransform wrapper.')
    parser.add_argument(
        '--cases',
        nargs='+',
        choices=['current', 'no_add', 'no_mul', 'no_bias', 'no_bias_rescale'],
        default=['current', 'no_add', 'no_mul', 'no_bias', 'no_bias_rescale'],
        help='Which debug variants to synthesize.',
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_crop(label_path: Path, prob_path: Path, crop_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    label = np.asarray(nib.load(str(label_path)).get_fdata(), dtype=np.int64)
    prob = np.asarray(nib.load(str(prob_path)).get_fdata(), dtype=np.float32)
    label = torch.from_numpy(label[:crop_size, :crop_size, :crop_size]).unsqueeze(0).long()
    prob = torch.from_numpy(prob[:crop_size, :crop_size, :crop_size]).unsqueeze(0).float()
    return label, prob


def load_dataset_sample(args) -> tuple[torch.Tensor, torch.Tensor]:
    if not args.label_dir:
        raise ValueError('--label_dir is required when --input_mode=dataset')
    start_time = time.monotonic()
    log_progress('building AxonSubsetDataset', start_time)
    dataset = AxonSubsetDataset(
        label_dir=args.label_dir,
        split='val',
        val_fraction=args.val_fraction,
        generate_images=False,
        num_samples_per_volume=args.num_samples_per_volume,
        segmentation_mode='three_class_shell_interior',
        background=0.5,
        fibers_lower_range=(0.3, 0.5),
        background_upper_range=(0.2, 0.4),
    )
    log_progress(f'dataset ready with {len(dataset)} samples', start_time)
    sample = dataset[args.sample_idx]
    if args.dataset_crop_size and args.dataset_crop_size > 0:
        crop_size = args.dataset_crop_size
        sample['label'] = sample['label'][:, :crop_size, :crop_size, :crop_size]
        sample['prob'] = sample['prob'][:, :crop_size, :crop_size, :crop_size]
        log_progress(f'cropped dataset sample to {crop_size}^3', start_time)
    log_progress(f'loaded dataset sample {args.sample_idx}', start_time)
    return sample['label'], sample['prob']


def save_views(image_np: np.ndarray, png_path: Path, title: str) -> None:
    sagittal = image_np[image_np.shape[0] // 2, :, :]
    coronal = image_np[:, image_np.shape[1] // 2, :]
    axial = image_np[:, :, image_np.shape[2] // 2]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, slc, view_name in zip(axes, [sagittal, coronal, axial], ['sagittal', 'coronal', 'axial']):
        ax.imshow(slc.T, cmap='gray', origin='lower')
        ax.set_title(view_name)
        ax.axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def run_case(
    name: str,
    *,
    label: torch.Tensor,
    prob: torch.Tensor,
    output_dir: Path,
    seed: int,
    backend: str,
    disable_add: bool = False,
    disable_mul: bool = False,
    disable_rescale: bool = False,
) -> None:
    start_time = time.monotonic()
    log_progress(f'start case {name} with backend={backend}', start_time)
    set_seed(seed)
    if backend == 'wrapper':
        synth = ControlledContrastAxonImage(background=0.5)
    else:
        synth = ControlledContrastAxonImage.XForm(background=0.5)
    if disable_add:
        synth.addbias = Identity()
    if disable_mul:
        synth.mulbias = Identity()
    if disable_rescale:
        synth.rescale = Identity()

    log_progress(f'configured synth for {name}', start_time)
    with torch.no_grad():
        image, _ = synth(label, prob)
    log_progress(f'synthesized image for {name}', start_time)

    image_np = image[0].cpu().numpy().astype(np.float32)
    nib.save(nib.Nifti1Image(image_np, np.eye(4)), str(output_dir / f'{name}.nii.gz'))
    log_progress(f'saved NIfTI for {name}', start_time)
    save_views(image_np, output_dir / f'{name}.png', name)
    log_progress(f'saved preview for {name}', start_time)

    column_profile = image_np.mean(axis=(0, 2))
    print(
        name,
        'stats',
        float(image_np.min()),
        float(image_np.max()),
        float(image_np.mean()),
        'profile_std',
        float(column_profile.std()),
    )


def main() -> None:
    start_time = time.monotonic()
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_progress(f'writing outputs to {output_dir}', start_time)
    if args.input_mode == 'direct':
        if not args.label or not args.prob:
            raise ValueError('--label and --prob are required when --input_mode=direct')
        label, prob = load_crop(Path(args.label), Path(args.prob), args.crop_size)
        log_progress('loaded direct crop inputs', start_time)
    else:
        label, prob = load_dataset_sample(args)
        log_progress('loaded dataset-mode inputs', start_time)

    case_options = {
        'current': dict(),
        'no_add': dict(disable_add=True),
        'no_mul': dict(disable_mul=True),
        'no_bias': dict(disable_add=True, disable_mul=True),
        'no_bias_rescale': dict(disable_add=True, disable_mul=True, disable_rescale=True),
    }

    for case_name in args.cases:
        run_case(
            case_name,
            label=label,
            prob=prob,
            output_dir=output_dir,
            seed=args.seed,
            backend=args.backend,
            **case_options[case_name],
        )

    log_progress('all requested cases completed', start_time)
    print('saved_to', output_dir)


if __name__ == '__main__':
    main()