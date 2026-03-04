"""
Axon Subset Dataset

PyTorch Dataset that loads pre-generated dense label volumes and
generates random axon subsets with varying spatial density distributions
on-the-fly, enabling unlimited training variation from a fixed set of
expensive label volumes.

Spatial density distributions supported:
    linear, sigmoid, gaussian, radial, uniform

Typical usage
-------------
    from datagen import AxonSubsetDataset, create_dataloader

    loader = create_dataloader(
        label_dir='/path/to/dense_labels',
        batch_size=4,
        num_workers=4,
        apply_density_curve=True,
        generate_images=True,
    )
"""
import random as pyrandom
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Worker-local synthesizer
#
# Each DataLoader worker is a separate process, so CUDA models cannot be
# shared from the parent.  We lazily create one CPU ControlledContrastAxonImage
# per worker process on first use and reuse it for all subsequent __getitem__
# calls in that worker.
# ---------------------------------------------------------------------------
_worker_synth: Dict[frozenset, object] = {}


def _get_or_create_synth(synth_kwargs: dict):
    """Return the process-local synthesizer for the given kwargs, creating it on first call.

    Keyed by synth kwargs so multiple datasets with different synthesis params
    can coexist in the same worker process without overwriting each other.
    """
    key = frozenset(synth_kwargs.items())
    if key not in _worker_synth:
        from datagen.axon_image_controlled_contrast import ControlledContrastAxonImage
        _worker_synth[key] = ControlledContrastAxonImage(**synth_kwargs)
    return _worker_synth[key]


def worker_init_fn(worker_id: int) -> None:
    """Seed per-worker randomness so every worker produces independent augmentations.

    PyTorch forks workers from the same parent state, so without explicit
    seeding all workers generate identical random subsets and density fields.
    Pass this to DataLoader as ``worker_init_fn=worker_init_fn``.
    """
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(seed)
    pyrandom.seed(seed)


def collate_fn(batch: list) -> dict:
    """Batch only tensor values; non-tensor metadata is dropped.

    PyTorch's default collator cannot handle dicts containing strings,
    nested dicts, or mixed types.  This collator stacks only the tensor
    entries that are present in every sample.
    """
    elem = batch[0]
    return {
        key: torch.stack([sample[key] for sample in batch])
        for key in elem
        if torch.is_tensor(elem[key])
    }


class DensityDistribution:
    """Factory for 3-D spatial keep-probability fields."""

    @staticmethod
    def linear(
        shape: Tuple[int, ...],
        axis: int = 2,
        low: float = 0.1,
        high: float = 1.0,
        ascending: bool = True,
    ) -> np.ndarray:
        """Linear gradient along one axis."""
        values = np.linspace(low, high, shape[axis])
        if not ascending:
            values = values[::-1]
        slices = [np.newaxis, np.newaxis, np.newaxis]
        slices[axis] = slice(None)
        return np.broadcast_to(values[tuple(slices)], shape).copy()

    @staticmethod
    def sigmoid(
        shape: Tuple[int, ...],
        axis: int = 2,
        low: float = 0.1,
        high: float = 1.0,
        center: float = 0.5,
        steepness: float = 10.0,
        ascending: bool = True,
    ) -> np.ndarray:
        """Sigmoid transition along one axis."""
        t = np.linspace(0, 1, shape[axis])
        values = low + (high - low) / (1 + np.exp(-steepness * (t - center)))
        if not ascending:
            values = values[::-1]
        slices = [np.newaxis, np.newaxis, np.newaxis]
        slices[axis] = slice(None)
        return np.broadcast_to(values[tuple(slices)], shape).copy()

    @staticmethod
    def gaussian(
        shape: Tuple[int, ...],
        axis: int = 2,
        low: float = 0.1,
        high: float = 1.0,
        center: float = 0.5,
        sigma: float = 0.2,
    ) -> np.ndarray:
        """Gaussian peak along one axis."""
        t = np.linspace(0, 1, shape[axis])
        values = low + (high - low) * np.exp(-((t - center) ** 2) / (2 * sigma ** 2))
        slices = [np.newaxis, np.newaxis, np.newaxis]
        slices[axis] = slice(None)
        return np.broadcast_to(values[tuple(slices)], shape).copy()

    @staticmethod
    def radial(
        shape: Tuple[int, ...],
        low: float = 0.1,
        high: float = 1.0,
        center_frac: Tuple[float, ...] = (0.5, 0.5, 0.5),
        invert: bool = False,
    ) -> np.ndarray:
        """Radial gradient from a centre point."""
        center = np.array([c * s for c, s in zip(center_frac, shape)])
        coords = np.stack(
            np.meshgrid(
                np.arange(shape[0]),
                np.arange(shape[1]),
                np.arange(shape[2]),
                indexing='ij',
            ),
            axis=-1,
        ).astype(float)
        dist     = np.linalg.norm(coords - center, axis=-1)
        max_dist = np.linalg.norm(np.array(shape) / 2)
        if invert:
            field = low + (high - low) * (dist / max_dist)
        else:
            field = high - (high - low) * (dist / max_dist)
        return field.clip(low, high)

    @staticmethod
    def uniform(shape: Tuple[int, ...], value: float = 1.0) -> np.ndarray:
        """Constant keep-probability."""
        return np.full(shape, value, dtype=np.float32)

    @classmethod
    def random(cls, shape: Tuple[int, ...]) -> Tuple[np.ndarray, dict]:
        """Sample a random distribution type and parameters.

        Returns
        -------
        field  : (D, H, W) float32 array of per-voxel keep probabilities
        config : dict describing the chosen distribution (for logging)
        """
        kind      = pyrandom.choice(['linear', 'sigmoid', 'gaussian', 'radial', 'uniform'])
        axis      = pyrandom.randint(0, 2)
        low       = pyrandom.uniform(0.05, 0.4)
        high      = pyrandom.uniform(0.6, 1.0)
        ascending = pyrandom.choice([True, False])
        config    = dict(type=kind, axis=axis, low=low, high=high, ascending=ascending)

        if kind == 'linear':
            field = cls.linear(shape, axis, low, high, ascending)
        elif kind == 'sigmoid':
            center     = pyrandom.uniform(0.3, 0.7)
            steepness  = pyrandom.uniform(5, 20)
            config.update(center=center, steepness=steepness)
            field = cls.sigmoid(shape, axis, low, high, center, steepness, ascending)
        elif kind == 'gaussian':
            center = pyrandom.uniform(0.2, 0.8)
            sigma  = pyrandom.uniform(0.1, 0.4)
            config.update(center=center, sigma=sigma)
            field = cls.gaussian(shape, axis, low, high, center, sigma)
        elif kind == 'radial':
            center_frac = tuple(pyrandom.uniform(0.3, 0.7) for _ in range(3))
            invert      = pyrandom.choice([True, False])
            config.update(center_frac=center_frac, invert=invert)
            field = cls.radial(shape, low, high, center_frac, invert)
        else:   # uniform
            value = pyrandom.uniform(0.3, 1.0)
            config['value'] = value
            field = cls.uniform(shape, value)

        return field, config


class AxonSubsetDataset(Dataset):
    """On-the-fly axon subset dataset for 3-D UNet training.

    All dense label volumes are loaded into memory at construction time
    so that ``__getitem__`` never touches disk.  Image synthesis is
    performed in each DataLoader worker using a process-local CPU instance
    of ``ControlledContrastAxonImage``.

    Output tensors have shape ``(1, D, H, W)`` (channel-first, single
    channel) matching MONAI convention.  After batching the DataLoader
    returns ``(B, 1, D, H, W)`` tensors.

    Parameters
    ----------
    label_dir : str or Path
        Directory with ``*_label.nii.gz`` / ``*_prob.nii.gz`` pairs.
    subset_fraction : (float, float)
        Uniform range for the axon keep-fraction when
        ``apply_density_curve`` is False.
    apply_density_curve : bool
        Apply a random spatial keep-probability field instead of a
        flat per-volume fraction.
    generate_images : bool
        Synthesise images on-the-fly.  Requires synthspline + cornucopia.
    transform : callable, optional
        Additional transform applied to the output dict after synthesis.
    num_samples_per_volume : int
        Number of random subsets drawn from each source volume per epoch.
    fibers_lower_range : (float, float)
        Uniform range for the axon intensity floor (passed to synthesizer).
    background_upper_range : (float, float)
        Uniform range for the background intensity ceiling.
    background : float
        Probability of adding background structures (passed to synthesizer).
    """

    def __init__(
        self,
        label_dir: Union[str, Path],
        subset_fraction: Tuple[float, float] = (0.3, 0.8),
        apply_density_curve: bool = True,
        generate_images: bool = True,
        transform: Optional[Callable] = None,
        num_samples_per_volume: int = 100,
        fibers_lower_range: Tuple[float, float] = (0.3, 0.5),
        background_upper_range: Tuple[float, float] = (0.2, 0.4),
        background: float = 0.5,
        split: str = 'train',
        val_fraction: float = 0.2,
    ):
        self.label_dir              = Path(label_dir)
        self.subset_fraction        = subset_fraction
        self.apply_density_curve    = apply_density_curve
        self.generate_images        = generate_images
        self.transform              = transform
        self.num_samples_per_volume = num_samples_per_volume

        # Synthesizer kwargs — model is created lazily per worker process.
        self._synth_kwargs = dict(
            background=background,
            fibers_lower_range=fibers_lower_range,
            background_upper_range=background_upper_range,
        )

        all_label_files: List[Path] = sorted(self.label_dir.glob('*_label.nii.gz'))
        if not all_label_files:
            raise ValueError(f'No *_label.nii.gz files found in {label_dir}')

        # --- Deterministic train / val split (fix #5) ---
        n_val = max(1, int(len(all_label_files) * val_fraction))
        if split == 'val':
            label_files = all_label_files[:n_val]
        elif split == 'train':
            label_files = all_label_files[n_val:]
        else:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        if not label_files:
            raise ValueError(
                f"Split '{split}' has no volumes "
                f"(total={len(all_label_files)}, n_val={n_val})"
            )

        self.split = split

        # --- Cache all volumes in memory (fix #1) ---
        print(f'[{split}] Loading {len(label_files)} label volumes into memory...')
        self._volumes: List[Tuple[np.ndarray, np.ndarray]] = []
        for lf in label_files:
            pf = lf.parent / lf.name.replace('_label', '_prob')
            labels = nib.load(lf).get_fdata().astype(np.int32)
            prob   = nib.load(pf).get_fdata().astype(np.float32)
            self._volumes.append((labels, prob))
        print(f'Cached {len(self._volumes)} volumes | '
              f'{num_samples_per_volume} samples/vol | '
              f'total={len(self)}')

    def __len__(self) -> int:
        return len(self._volumes) * self.num_samples_per_volume

    def _apply_subset(
        self,
        labels: np.ndarray,
        prob: np.ndarray,
        keep_prob_field: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Vectorized axon subset selection."""
        unique_axons = np.unique(labels)
        unique_axons = unique_axons[unique_axons > 0]
        n_total = len(unique_axons)

        if n_total == 0:
            return labels.copy(), prob.copy(), dict(n_total=0, n_kept=0, fraction=0.0)

        if keep_prob_field is not None:
            # Per-axon probability = mean field value over axon voxels
            per_axon_p = np.array([
                keep_prob_field[labels == aid].mean() for aid in unique_axons
            ])
            keep_mask = np.random.random(n_total) < per_axon_p
        else:
            frac      = pyrandom.uniform(*self.subset_fraction)
            n_keep    = max(1, int(n_total * frac))
            perm      = np.random.permutation(n_total)
            keep_mask = np.zeros(n_total, dtype=bool)
            keep_mask[perm[:n_keep]] = True

        # Vectorized removal via lookup table (no Python loop over voxels)
        label_max = int(labels.max()) + 1
        keep_lut  = np.zeros(label_max, dtype=bool)
        keep_lut[0] = True                          # background always kept
        keep_lut[unique_axons[keep_mask]] = True    # kept axons

        keep_voxels   = keep_lut[labels]
        subset_labels = np.where(keep_voxels, labels, 0).astype(np.int32)
        subset_prob   = np.where(keep_voxels, prob, 0.0).astype(np.float32)

        n_kept = int(keep_mask.sum())
        return subset_labels, subset_prob, dict(
            n_total=n_total,
            n_kept=n_kept,
            fraction=n_kept / n_total,
        )

    def __getitem__(self, idx: int) -> dict:
        vol_idx      = idx // self.num_samples_per_volume
        labels, prob = self._volumes[vol_idx]          # from in-memory cache

        # Deterministic validation: same idx always produces identical sample (fix #6)
        if self.split == 'val':
            np.random.seed(idx % (2 ** 31))
            pyrandom.seed(idx)

        keep_prob_field = density_config = None
        if self.apply_density_curve:
            keep_prob_field, density_config = DensityDistribution.random(labels.shape)

        subset_labels, subset_prob, subset_info = self._apply_subset(
            labels, prob, keep_prob_field
        )

        # (1, D, H, W)  —  channel-first, MONAI convention
        label_t = torch.from_numpy(subset_labels).unsqueeze(0).long()
        prob_t  = torch.from_numpy(subset_prob).unsqueeze(0).float()

        result: dict = dict(
            # metadata — not collated into batches, useful for debugging
            density_config=density_config,
            subset_info=subset_info,
        )

        if self.generate_images:
            # Worker-local CPU synthesizer (fix #2)
            synth = _get_or_create_synth(self._synth_kwargs)
            with torch.no_grad():
                image, out_prob = synth(label_t, prob_t)
            # 'image': network input  |  'seg': segmentation target
            result['image'] = image      # (1, D, H, W)
            result['seg']   = out_prob   # (1, D, H, W)
        else:
            result['label'] = label_t
            result['prob']  = prob_t

        if self.transform is not None:
            result = self.transform(result)

        return result


def create_dataloader(
    label_dir: Union[str, Path],
    batch_size: int = 2,
    num_workers: int = 10,
    pin_memory: bool = True,
    **dataset_kwargs,
) -> torch.utils.data.DataLoader:
    """Convenience factory: AxonSubsetDataset → DataLoader.

    Defaults are tuned for a single-GPU training node with::

        #SBATCH --gres=gpu:1
        #SBATCH --cpus-per-task=12   # 10 workers + 1 main + 1 spare

    ``batch_size=2`` is chosen for 128³ volumes which are memory-heavy on GPU.
    Raise ``num_workers`` (and ``--cpus-per-task``) if ``nvidia-smi dmon``
    shows GPU utilisation dropping below ~85% between batches.

    Applies:
    - ``worker_init_fn``     — independent randomness per worker (fix #4)
    - ``collate_fn``         — tensor-only batching, drops metadata (fix #3)
    - ``persistent_workers`` — workers stay alive between epochs, preserving
                               the in-memory volume cache
    """
    dataset = AxonSubsetDataset(label_dir, **dataset_kwargs)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(dataset.split == 'train'),
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        persistent_workers=(num_workers > 0),
        drop_last=True,
        prefetch_factor=(4 if num_workers > 0 else None),
    )
