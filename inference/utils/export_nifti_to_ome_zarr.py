"""Export top-level HiP-CT NIfTI outputs to OME-Zarr assets.

Example
-------
    python export_nifti_to_ome_zarr.py \
        --input_dir /scratch/experiment/hipct/inference/patch_I74_IC_zoom01/three_class_full \
        --output_dir /scratch/experiment/hipct/inference/patch_I74_IC_zoom01/three_class_full_ome
"""

import argparse
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import zarr
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image


DEFAULT_INPUTS = (
    "hipct_input.nii.gz",
    "hipct_pred.nii.gz",
    "hipct_pred_class.nii.gz",
    "hipct_pred_prob.nii.gz",
)


def parse_chunk_shape(value: str) -> tuple[int, int, int]:
    parts = tuple(int(part.strip()) for part in value.split(","))
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise argparse.ArgumentTypeError(
            "--chunk_shape must be three positive integers, e.g. 64,64,64"
        )
    return parts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert top-level HiP-CT NIfTI outputs into OME-Zarr assets"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing the top-level .nii.gz outputs",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="New directory where .ome.zarr outputs will be written",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=list(DEFAULT_INPUTS),
        help="Top-level NIfTI files to convert",
    )
    parser.add_argument(
        "--chunk_shape",
        type=parse_chunk_shape,
        default=(64, 64, 64),
        help="OME-Zarr chunk shape in z,y,x order (default: 64,64,64)",
    )
    return parser.parse_args()


def output_name_for(input_path: Path) -> str:
    if input_path.name.endswith(".nii.gz"):
        return f"{input_path.name[:-7]}.ome.zarr"
    if input_path.suffix == ".nii":
        return f"{input_path.stem}.ome.zarr"
    raise ValueError(f"Unsupported NIfTI extension: {input_path.name}")


def prepare_output_dir(output_dir: Path):
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory already exists and is not empty: {output_dir}"
            )
        return
    output_dir.mkdir(parents=True)


def load_nifti_as_zyx(input_path: Path) -> tuple[np.ndarray, list[float]]:
    image = nib.load(str(input_path))
    array_xyz = np.asanyarray(image.dataobj)
    if array_xyz.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape={array_xyz.shape}")

    # The inference outputs are saved as x,y,z NIfTI volumes. OME-Zarr expects z,y,x.
    array_zyx = np.transpose(array_xyz, (2, 1, 0))
    zooms_xyz = tuple(float(zoom) for zoom in image.header.get_zooms()[:3])
    scale_zyx = [zooms_xyz[2], zooms_xyz[1], zooms_xyz[0]]
    return array_zyx, scale_zyx


def write_single_scale_ome_zarr(
    input_path: Path,
    output_path: Path,
    chunk_shape: tuple[int, int, int],
):
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing path: {output_path}")

    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    if temp_path.exists():
        shutil.rmtree(temp_path)

    array_zyx, scale_zyx = load_nifti_as_zyx(input_path)

    try:
        location = parse_url(str(temp_path), mode="w")
        if location is None:
            raise RuntimeError(f"Could not open OME-Zarr store for writing: {temp_path}")
        root = zarr.group(store=location.store)
        write_image(
            array_zyx,
            group=root,
            scaler=None,
            axes=["z", "y", "x"],
            coordinate_transformations=[[{"type": "scale", "scale": scale_zyx}]],
            storage_options={
                "chunks": chunk_shape,
                "dimension_separator": "/",
            },
        )
        temp_path.rename(output_path)
    except Exception:
        if temp_path.exists():
            shutil.rmtree(temp_path)
        raise

    print(
        f"Wrote {output_path} from {input_path.name} "
        f"shape={array_zyx.shape} dtype={array_zyx.dtype} scale_zyx={scale_zyx}"
    )


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    prepare_output_dir(output_dir)

    missing_inputs = [name for name in args.inputs if not (input_dir / name).is_file()]
    if missing_inputs:
        missing = ", ".join(missing_inputs)
        raise FileNotFoundError(f"Missing input files: {missing}")

    for input_name in args.inputs:
        input_path = input_dir / input_name
        output_path = output_dir / output_name_for(input_path)
        write_single_scale_ome_zarr(input_path, output_path, args.chunk_shape)


if __name__ == "__main__":
    main()