"""
Download a bounding-box patch from either:

1. a legacy LINCBrain zarr asset exposed via the authenticated DANDI API, or
2. a public OME-Zarr root (for example the HiP-CT VOI assets hosted on GCS).

Usage
-----
Legacy authenticated mode:

    export LINCBRAIN_API_KEY=<your-lincbrain-api-key>
    python download_patch.py \
        --zarr_id   b8418d95-409f-4b87-89be-93a9ba240f5f \
        --level     0 \
        --bbox      1461,210,3295,3323,1180,4889 \
        --output    /scratch/experiment/hipct/patch_I74_IC_zoom01.raw

Public OME-Zarr mode:

    python download_patch.py \
        --source_url zarr://gs://ucl-hip-ct-35a68e99feaae8932b1d44da0358940b/I74/brain-internalCapsule/0.857um_VOI-1_bm05.ome.zarr/ \
        --level      0 \
        --bbox       2740,1612,1200,3289,4154,3915 \
        --output     /scratch/experiment/hipct/patch_I74_IC_VOI1.raw

The bounding box is specified as x0,y0,z0,x1,y1,z1 (voxel indices at the
chosen resolution level, exclusive end).
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import requests


API_BASE = "https://api.lincbrain.org/api"


def parse_args():
    p = argparse.ArgumentParser(
        description="Download a bounding-box patch from a legacy LINCBrain zarr asset or a public OME-Zarr root"
    )
    p.add_argument("--zarr_id", help="Legacy authenticated zarr blob UUID (from asset contentUrl)")
    p.add_argument(
        "--source_url",
        help=(
            "Public OME-Zarr root URL. Supports zarr://gs://bucket/path, gs://bucket/path, "
            "https://storage.googleapis.com/bucket/path, or any direct http(s) OME-Zarr root"
        ),
    )
    p.add_argument("--level", type=int, default=0, help="Resolution pyramid level (0 = full res)")
    p.add_argument("--bbox", required=True,
                   help="Bounding box as x0,y0,z0,x1,y1,z1 (voxel coords, exclusive end)")
    p.add_argument("--output", required=True, help="Output path (.raw)")
    p.add_argument("--api_key", default=None,
                   help="Legacy authenticated mode only: LINCBrain API key (default: $LINCBRAIN_API_KEY, fallback: $DANDI_API_KEY)")
    p.add_argument(
        "--block_depth",
        type=int,
        default=None,
        help="Public OME-Zarr mode only: number of z-slices to read per block (default: array chunk depth)",
    )
    return p.parse_args()


def parse_bbox(text):
    try:
        bbox = list(map(int, text.split(",")))
    except ValueError as exc:
        raise SystemExit(
            "Error: --bbox must have 6 comma-separated ints: x0,y0,z0,x1,y1,z1"
        ) from exc

    if len(bbox) != 6:
        sys.exit("Error: --bbox must have 6 comma-separated ints: x0,y0,z0,x1,y1,z1")

    return bbox


def validate_bbox(bounds, shape):
    starts = bounds[:3]
    ends = bounds[3:]
    for i, (lo, hi, size) in enumerate(zip(starts, ends, shape)):
        if lo < 0 or hi > size or lo >= hi:
            sys.exit(f"Error: bbox dim {i} [{lo}:{hi}] out of range [0:{size}]")


def allocate_output(output_path, dtype, patch_shape):
    tmp_path = Path(str(output_path) + ".tmp")
    patch = np.memmap(str(tmp_path), dtype=dtype, mode="w+", shape=patch_shape)
    print(f"Allocated memmap: {tmp_path}  ({patch.nbytes / 1e9:.1f} GB on disk)")
    return tmp_path, patch


def finalize_output(tmp_path, output_path, patch_shape, dtype, extra_meta=None):
    meta = {"shape": list(patch_shape), "dtype": str(dtype)}
    if extra_meta:
        meta.update(extra_meta)

    meta_path = Path(str(output_path) + ".json")
    meta_path.write_text(json.dumps(meta))

    tmp_path.rename(output_path)
    size_mb = output_path.stat().st_size / 1e6
    print(f"\nSaved {output_path}  ({size_mb:.0f} MB)")
    print(f"Shape: {patch_shape}, dtype: {dtype}")
    print(f"Load with: np.memmap(path, dtype='{dtype}', mode='r', shape={patch_shape})")


def get_zarr_json(session, zarr_id, path):
    """Fetch a zarr metadata file (e.g. .zarray) via presigned redirect."""
    url = f"{API_BASE}/zarr/{zarr_id}/files/"
    r = session.get(url, params={"prefix": path, "download": "true"},
                    allow_redirects=True, timeout=30)
    r.raise_for_status()
    return r.json()


def get_zarr_bytes(session, zarr_id, path):
    """Fetch raw bytes of a zarr chunk via presigned redirect."""
    url = f"{API_BASE}/zarr/{zarr_id}/files/"
    r = session.get(url, params={"prefix": path, "download": "true"},
                    allow_redirects=True, timeout=60)
    r.raise_for_status()
    return r.content


def normalize_public_source_url(source_url):
    url = source_url.strip()
    if url.startswith("zarr://"):
        url = url[len("zarr://"):]

    if url.startswith("gs://"):
        bucket_and_path = url[len("gs://"):].strip("/")
        bucket, sep, path = bucket_and_path.partition("/")
        if not bucket or not sep or not path:
            sys.exit("Error: gs:// source must include both bucket and object path")
        return f"https://storage.googleapis.com/{bucket}/{path.rstrip('/')}"

    if url.startswith("https://storage.googleapis.com/"):
        return url.rstrip("/")

    if url.startswith("https://") or url.startswith("http://"):
        return url.rstrip("/")

    sys.exit(
        "Error: --source_url must be zarr://gs://..., gs://..., https://storage.googleapis.com/..., or a direct http(s) OME-Zarr root"
    )


def get_public_json(url):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def download_public_patch(args, bbox, output_path):
    try:
        import tensorstore as ts
    except ImportError as exc:
        raise SystemExit(
            "Error: tensorstore is required for --source_url mode. Install it with `pip install tensorstore`."
        ) from exc

    public_root = normalize_public_source_url(args.source_url)
    try:
        root_zarr_json = get_public_json(f"{public_root}/zarr.json")
    except Exception:
        root_zarr_json = {}

    multiscales = ((root_zarr_json.get("attributes") or {}).get("multiscales") or [])
    if multiscales:
        datasets = multiscales[0].get("datasets") or []
        if args.level < 0 or args.level >= len(datasets):
            sys.exit(f"Error: level {args.level} is out of range for source with {len(datasets)} levels")
        dataset_path = datasets[args.level]["path"]
        axis_names = [
            axis["name"] if isinstance(axis, dict) else str(axis)
            for axis in (multiscales[0].get("axes") or [])
        ]
    else:
        dataset_path = str(args.level)
        axis_names = ["x", "y", "z"]

    if len(axis_names) != 3 or set(axis_names) != {"x", "y", "z"}:
        sys.exit(f"Error: public source axes must be a permutation of x,y,z, got {axis_names}")

    level_base = f"{public_root}/{dataset_path}/"
    zarr_json = get_public_json(f"{level_base}zarr.json")

    store_shape = tuple(zarr_json["shape"])
    store_chunk_shape = tuple(zarr_json["chunk_grid"]["configuration"]["chunk_shape"])
    dtype = np.dtype(zarr_json["data_type"])

    store_axis_to_shape = dict(zip(axis_names, store_shape))
    store_axis_to_chunk = dict(zip(axis_names, store_chunk_shape))
    xyz_shape = tuple(store_axis_to_shape[axis] for axis in ("x", "y", "z"))
    xyz_chunk_shape = tuple(store_axis_to_chunk[axis] for axis in ("x", "y", "z"))

    x0, y0, z0, x1, y1, z1 = bbox

    print(f"Public source: {public_root}")
    print(
        f"Level {args.level} (path {dataset_path}): "
        f"store_shape={store_shape}, store_chunks={store_chunk_shape}, dtype={dtype}"
    )
    print(f"Axes: {axis_names} (bbox interpreted as x,y,z)")
    print(f"Bounding box xyz: ({x0},{y0},{z0}) → ({x1},{y1},{z1})")
    print(
        f"Patch size xyz: {x1-x0} × {y1-y0} × {z1-z0} = "
        f"{(x1-x0)*(y1-y0)*(z1-z0):,} voxels"
    )

    validate_bbox(bbox, xyz_shape)

    spec = {
        "driver": "zarr3",
        "kvstore": {
            "driver": "http",
            "base_url": level_base,
        },
    }
    array = ts.open(spec).result()

    patch_shape = (x1 - x0, y1 - y0, z1 - z0)
    tmp_path, patch = allocate_output(output_path, dtype, patch_shape)

    chunk_depth = int(xyz_chunk_shape[2])
    block_depth = args.block_depth or chunk_depth
    if block_depth <= 0:
        sys.exit("Error: --block_depth must be positive")

    n_blocks = math.ceil((z1 - z0) / block_depth)
    print(f"Reading {n_blocks} z-blocks with block_depth={block_depth}")

    transpose_order = tuple(axis_names.index(axis) for axis in ("x", "y", "z"))

    for block_index, bz0 in enumerate(range(z0, z1, block_depth), start=1):
        bz1 = min(bz0 + block_depth, z1)
        store_slices = tuple(
            {
                "x": slice(x0, x1),
                "y": slice(y0, y1),
                "z": slice(bz0, bz1),
            }[axis]
            for axis in axis_names
        )
        slab = array[store_slices].read().result()
        if transpose_order != (0, 1, 2):
            slab = np.transpose(slab, transpose_order)
        patch[:, :, bz0 - z0 : bz1 - z0] = slab
        patch.flush()
        print(f"  [{block_index}/{n_blocks}] z[{bz0}:{bz1}] downloaded")

    patch.flush()
    del patch

    finalize_output(
        tmp_path,
        output_path,
        patch_shape,
        dtype,
        extra_meta={
            "level": args.level,
            "source_url": public_root,
            "axes": axis_names,
            "bbox_order": ["x", "y", "z"],
        },
    )


def download_legacy_api_patch(args, bbox, output_path):
    import blosc

    x0, y0, z0, x1, y1, z1 = bbox
    api_key = args.api_key or os.environ.get("LINCBRAIN_API_KEY") or os.environ.get("DANDI_API_KEY")
    if not api_key:
        sys.exit("Error: provide --api_key or set LINCBRAIN_API_KEY (or legacy DANDI_API_KEY)")

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {api_key}"})

    zarray = get_zarr_json(session, args.zarr_id, f"{args.level}/.zarray")
    shape = zarray["shape"]
    chunks = zarray["chunks"]
    dtype = np.dtype(zarray["dtype"])
    print(f"Level {args.level}: shape={shape}, chunks={chunks}, dtype={dtype}")
    print(f"Bounding box: ({x0},{y0},{z0}) → ({x1},{y1},{z1})")
    print(f"Patch size: {x1-x0} × {y1-y0} × {z1-z0} = {(x1-x0)*(y1-y0)*(z1-z0):,} voxels")

    validate_bbox(bbox, shape)

    cx, cy, cz = chunks
    ci0, ci1 = x0 // cx, math.ceil(x1 / cx)
    cj0, cj1 = y0 // cy, math.ceil(y1 / cy)
    ck0, ck1 = z0 // cz, math.ceil(z1 / cz)
    n_chunks = (ci1 - ci0) * (cj1 - cj0) * (ck1 - ck0)
    print(f"Chunk grid: x[{ci0}:{ci1}] y[{cj0}:{cj1}] z[{ck0}:{ck1}]  ({n_chunks} chunks to fetch)")

    patch_shape = (x1 - x0, y1 - y0, z1 - z0)
    tmp_path, patch = allocate_output(output_path, dtype, patch_shape)

    fill_value = zarray.get("fill_value", 0)
    order = zarray.get("order", "C")

    done = 0
    for ci in range(ci0, ci1):
        for cj in range(cj0, cj1):
            for ck in range(ck0, ck1):
                done += 1
                chunk_path = f"{args.level}/{ci}/{cj}/{ck}"

                gx0, gx1 = ci * cx, min((ci + 1) * cx, shape[0])
                gy0, gy1 = cj * cy, min((cj + 1) * cy, shape[1])
                gz0, gz1 = ck * cz, min((ck + 1) * cz, shape[2])

                ix0 = max(gx0, x0)
                ix1 = min(gx1, x1)
                iy0 = max(gy0, y0)
                iy1 = min(gy1, y1)
                iz0 = max(gz0, z0)
                iz1 = min(gz1, z1)

                try:
                    raw = get_zarr_bytes(session, args.zarr_id, chunk_path)
                    decompressed = blosc.decompress(raw)
                    chunk_shape = (gx1 - gx0, gy1 - gy0, gz1 - gz0)
                    arr = np.frombuffer(decompressed, dtype=dtype).reshape(
                        chunk_shape if order == "C" else chunk_shape[::-1]
                    )
                    if order == "F":
                        arr = arr.T
                except Exception as exc:
                    print(f"  [{done}/{n_chunks}] {chunk_path}: error ({exc}), filling with {fill_value}")
                    arr = np.full((gx1 - gx0, gy1 - gy0, gz1 - gz0), fill_value, dtype=dtype)

                patch[
                    ix0 - x0 : ix1 - x0,
                    iy0 - y0 : iy1 - y0,
                    iz0 - z0 : iz1 - z0,
                ] = arr[
                    ix0 - gx0 : ix1 - gx0,
                    iy0 - gy0 : iy1 - gy0,
                    iz0 - gz0 : iz1 - gz0,
                ]

                if done % 50 == 0 or done == n_chunks:
                    patch.flush()
                    print(f"  [{done}/{n_chunks}] chunks downloaded")

    patch.flush()
    del patch

    finalize_output(
        tmp_path,
        output_path,
        patch_shape,
        dtype,
        extra_meta={
            "level": args.level,
            "zarr_id": args.zarr_id,
        },
    )


def main():
    args = parse_args()
    bbox = parse_bbox(args.bbox)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if bool(args.zarr_id) == bool(args.source_url):
        sys.exit("Error: provide exactly one of --zarr_id or --source_url")

    if args.source_url:
        download_public_patch(args, bbox, output_path)
        return

    download_legacy_api_patch(args, bbox, output_path)


if __name__ == "__main__":
    main()
