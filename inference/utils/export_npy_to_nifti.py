"""Export one or more NumPy volumes to NIfTI for external viewers.

Example
-------
    python export_npy_to_nifti.py \
        --inputs /scratch/experiment/webknossos/macaque_NEFH_WM_binary.npy \
                 /scratch/experiment/webknossos/macaque_NEFH_WM_valid_mask.npy \
        --output_dir /scratch/experiment/webknossos/nifti_for_niivue \
        --voxel_size 1.0
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Export NumPy volumes to NIfTI")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input .npy files to export",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where .nii files will be written",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=1.0,
        help="Isotropic voxel size in um for the output affine",
    )
    return parser.parse_args()


def save_nifti(array: np.ndarray, output_path: Path, voxel_size: float):
    affine = np.diag([voxel_size, voxel_size, voxel_size, 1.0])
    nib.save(nib.Nifti1Image(array, affine=affine), str(output_path))


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_name in args.inputs:
        input_path = Path(input_name)
        array = np.load(input_path)
        output_path = output_dir / f"{input_path.stem}.nii"
        save_nifti(array, output_path, args.voxel_size)
        print(
            f"Saved {output_path} shape={array.shape} dtype={array.dtype} "
            f"range=[{array.min()}, {array.max()}]"
        )


if __name__ == "__main__":
    main()