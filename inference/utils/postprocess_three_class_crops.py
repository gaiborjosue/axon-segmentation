"""Post-process 3-class crop outputs with component filtering and confidence anchoring.

The pipeline keeps only low-threshold foreground components that are anchored by
filtered high-confidence cores. This suppresses isolated low-confidence speckle
while preserving large structures supported by the 0.8 confidence core.
"""

import argparse
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


VALID_CLASS_LABELS = {0, 1, 2}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-process 3-class crop outputs using connected components and confidence anchoring"
    )
    parser.add_argument("--input_dir", required=True, help="Directory containing existing crop NIfTI files")
    parser.add_argument("--output_dir", required=True, help="Directory for post-processed crop outputs")
    parser.add_argument(
        "--low_threshold",
        type=float,
        default=0.5,
        help="Low threshold used to define candidate foreground components",
    )
    parser.add_argument(
        "--high_threshold",
        type=float,
        default=0.8,
        help="High threshold used to define high-confidence anchor components",
    )
    parser.add_argument(
        "--min_low_component_size",
        type=int,
        default=200,
        help="Minimum voxel count for a low-threshold component to be retained",
    )
    parser.add_argument(
        "--min_high_component_size",
        type=int,
        default=200,
        help="Minimum voxel count for a high-threshold component to anchor retained foreground",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="3D connected-component connectivity (1=6-neighbor, 2=18-neighbor, 3=26-neighbor)",
    )
    parser.add_argument(
        "--max_band_distance",
        type=float,
        default=None,
        help=(
            "Maximum Euclidean distance from the retained high-confidence core "
            "for non-core band voxels. Default: keep the full retained band."
        ),
    )
    parser.add_argument(
        "--min_band_mean_probability",
        type=float,
        default=0.0,
        help=(
            "Minimum mean probability of the non-core band voxels in a retained "
            "low-threshold component."
        ),
    )
    parser.add_argument(
        "--min_core_overlap_fraction",
        type=float,
        default=0.0,
        help=(
            "Minimum fraction of a low-threshold component that must be occupied "
            "by the retained high-confidence core."
        ),
    )
    return parser.parse_args()


def infer_tags(input_dir: Path) -> list[str]:
    suffix = "_pred_prob.nii.gz"
    tags = sorted(
        path.name[: -len(suffix)]
        for path in input_dir.glob(f"*{suffix}")
        if path.is_file()
    )
    if not tags:
        raise FileNotFoundError(f"No *_pred_prob.nii.gz files found in {input_dir}")
    return tags


def save_like(reference_img: nib.Nifti1Image, array: np.ndarray, output_path: Path):
    image = nib.Nifti1Image(array, reference_img.affine, header=reference_img.header.copy())
    image.set_data_dtype(array.dtype)
    nib.save(image, str(output_path))


def connected_components(mask: np.ndarray, connectivity: int):
    structure = ndi.generate_binary_structure(mask.ndim, connectivity)
    labels, count = ndi.label(mask, structure=structure)
    return labels, count


def filter_small_components(mask: np.ndarray, min_size: int, connectivity: int):
    labels, count = connected_components(mask, connectivity)
    if count == 0:
        return np.zeros_like(mask, dtype=bool), {
            "component_count": 0,
            "kept_component_count": 0,
            "component_sizes": [],
            "kept_component_sizes": [],
        }

    sizes = np.bincount(labels.ravel())[1:]
    keep_ids = np.flatnonzero(sizes >= min_size) + 1
    kept = np.isin(labels, keep_ids)
    return kept, {
        "component_count": int(count),
        "kept_component_count": int(len(keep_ids)),
        "component_sizes": sizes.astype(int).tolist(),
        "kept_component_sizes": sizes[keep_ids - 1].astype(int).tolist(),
    }


def anchored_foreground(
    probability: np.ndarray,
    low_threshold: float,
    high_threshold: float,
    min_low_component_size: int,
    min_high_component_size: int,
    connectivity: int,
    max_band_distance: float | None,
    min_band_mean_probability: float,
    min_core_overlap_fraction: float,
):
    low_mask = probability >= low_threshold
    high_mask = probability >= high_threshold

    high_filtered, high_stats = filter_small_components(
        high_mask,
        min_high_component_size,
        connectivity,
    )

    low_labels, low_count = connected_components(low_mask, connectivity)
    low_sizes = np.bincount(low_labels.ravel())[1:] if low_count else np.array([], dtype=int)
    distance_to_core = None
    if np.any(high_filtered):
        distance_to_core = ndi.distance_transform_edt(~high_filtered)

    kept_low_ids = []
    retained_band = np.zeros_like(low_mask, dtype=bool)
    low_component_band_mean_probabilities = []
    low_component_core_overlap_fractions = []
    low_component_band_voxel_counts = []
    low_component_retained_band_voxel_counts = []
    low_component_core_voxel_counts = []
    for component_id in range(1, low_count + 1):
        component_size = int(low_sizes[component_id - 1])
        component_mask = low_labels == component_id
        component_core = component_mask & high_filtered
        component_band = component_mask & ~high_filtered
        component_core_voxels = int(component_core.sum())
        component_band_voxels = int(component_band.sum())
        if component_band_voxels:
            component_band_mean_probability = float(probability[component_band].mean())
        else:
            component_band_mean_probability = 1.0
        component_core_overlap_fraction = (
            float(component_core_voxels) / float(component_size)
            if component_size
            else 0.0
        )
        if max_band_distance is None:
            component_retained_band = component_band
        else:
            if distance_to_core is None:
                component_retained_band = np.zeros_like(component_band, dtype=bool)
            else:
                component_retained_band = component_band & (distance_to_core <= max_band_distance)

        low_component_band_mean_probabilities.append(component_band_mean_probability)
        low_component_core_overlap_fractions.append(component_core_overlap_fraction)
        low_component_band_voxel_counts.append(component_band_voxels)
        low_component_retained_band_voxel_counts.append(int(component_retained_band.sum()))
        low_component_core_voxel_counts.append(component_core_voxels)

        if component_size < min_low_component_size:
            continue
        if component_core_voxels == 0:
            continue
        if component_band_mean_probability < min_band_mean_probability:
            continue
        if component_core_overlap_fraction < min_core_overlap_fraction:
            continue
        kept_low_ids.append(component_id)
        retained_band |= component_retained_band

    foreground = high_filtered | retained_band
    stats = {
        "low_component_count": int(low_count),
        "low_component_sizes": low_sizes.astype(int).tolist(),
        "kept_low_component_count": int(len(kept_low_ids)),
        "kept_low_component_sizes": low_sizes[np.array(kept_low_ids) - 1].astype(int).tolist()
        if kept_low_ids
        else [],
        "low_component_band_mean_probabilities": low_component_band_mean_probabilities,
        "low_component_core_overlap_fractions": low_component_core_overlap_fractions,
        "low_component_band_voxel_counts": low_component_band_voxel_counts,
        "low_component_retained_band_voxel_counts": low_component_retained_band_voxel_counts,
        "low_component_core_voxel_counts": low_component_core_voxel_counts,
        "high": high_stats,
        "low_threshold": float(low_threshold),
        "high_threshold": float(high_threshold),
        "min_low_component_size": int(min_low_component_size),
        "min_high_component_size": int(min_high_component_size),
        "max_band_distance": None if max_band_distance is None else float(max_band_distance),
        "min_band_mean_probability": float(min_band_mean_probability),
        "min_core_overlap_fraction": float(min_core_overlap_fraction),
    }
    return foreground, high_filtered, stats


def validate_class_labels(pred_class: np.ndarray) -> list[int]:
    unique_classes = [int(value) for value in np.unique(pred_class).tolist()]
    unexpected = sorted(set(unique_classes) - VALID_CLASS_LABELS)
    if unexpected:
        raise ValueError(
            f"Unexpected class labels {unexpected}; expected only {sorted(VALID_CLASS_LABELS)}"
        )
    return unique_classes


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tags = infer_tags(input_dir)
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "low_threshold": args.low_threshold,
        "high_threshold": args.high_threshold,
        "min_low_component_size": args.min_low_component_size,
        "min_high_component_size": args.min_high_component_size,
        "connectivity": args.connectivity,
        "max_band_distance": args.max_band_distance,
        "min_band_mean_probability": args.min_band_mean_probability,
        "min_core_overlap_fraction": args.min_core_overlap_fraction,
        "crops": [],
    }

    for tag in tags:
        input_path = input_dir / f"{tag}_input.nii.gz"
        pred_prob_path = input_dir / f"{tag}_pred_prob.nii.gz"
        pred_path = input_dir / f"{tag}_pred.nii.gz"
        pred_class_path = input_dir / f"{tag}_pred_class.nii.gz"

        if not pred_prob_path.exists() or not pred_path.exists() or not pred_class_path.exists():
            raise FileNotFoundError(f"Missing expected crop files for tag {tag}")

        pred_prob_img = nib.load(str(pred_prob_path))
        pred_img = nib.load(str(pred_path))
        pred_class_img = nib.load(str(pred_class_path))

        pred_prob = np.asarray(pred_prob_img.dataobj, dtype=np.float32)
        pred = np.asarray(pred_img.dataobj, dtype=np.uint8)
        pred_class = np.asarray(pred_class_img.dataobj, dtype=np.uint8)

        original_classes = validate_class_labels(pred_class)
        original_pred_voxels = int(pred.sum())
        low_support_voxels = int((pred_prob >= args.low_threshold).sum())
        foreground, high_core, stats = anchored_foreground(
            pred_prob,
            args.low_threshold,
            args.high_threshold,
            args.min_low_component_size,
            args.min_high_component_size,
            args.connectivity,
            args.max_band_distance,
            args.min_band_mean_probability,
            args.min_core_overlap_fraction,
        )

        pred_post = foreground.astype(np.uint8)
        pred_class_post = np.where(foreground, pred_class, 0).astype(np.uint8)
        post_classes = validate_class_labels(pred_class_post)
        shell_post = (pred_class_post == 1).astype(np.uint8)
        interior_post = (pred_class_post == 2).astype(np.uint8)
        band_post = (foreground & ~high_core).astype(np.uint8)
        postprocessed_foreground_voxels = int(pred_post.sum())

        if input_path.exists():
            shutil.copy2(input_path, output_dir / input_path.name)
        shutil.copy2(pred_prob_path, output_dir / pred_prob_path.name)
        save_like(pred_img, pred_post, output_dir / pred_path.name)
        save_like(pred_class_img, pred_class_post, output_dir / pred_class_path.name)
        save_like(pred_class_img, shell_post, output_dir / f"{tag}_pred_shell.nii.gz")
        save_like(pred_class_img, interior_post, output_dir / f"{tag}_pred_interior.nii.gz")
        save_like(pred_img, high_core.astype(np.uint8), output_dir / f"{tag}_pred_core.nii.gz")
        save_like(pred_img, band_post, output_dir / f"{tag}_pred_band.nii.gz")

        crop_summary = {
            "tag": tag,
            "original_pred_voxels": original_pred_voxels,
            "low_support_voxels": low_support_voxels,
            "postprocessed_foreground_voxels": postprocessed_foreground_voxels,
            "voxels_removed_from_low_support": int(low_support_voxels - postprocessed_foreground_voxels),
            "delta_vs_original_pred": int(postprocessed_foreground_voxels - original_pred_voxels),
            "original_classes_present": original_classes,
            "postprocessed_classes_present": post_classes,
            "high_core_voxels": int(high_core.sum()),
            "band_voxels": int(band_post.sum()),
            "component_stats": stats,
        }
        summary["crops"].append(crop_summary)

        print(
            f"{tag}: support {low_support_voxels} -> {postprocessed_foreground_voxels} "
            f"(delta vs saved pred {crop_summary['delta_vs_original_pred']:+d}) "
            f"classes={post_classes}"
        )

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved post-processed crops to {output_dir}")


if __name__ == "__main__":
    main()