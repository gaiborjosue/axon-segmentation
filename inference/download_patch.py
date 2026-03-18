"""
Download a bounding-box patch from a LINCBrain OME-Zarr asset via the DANDI API.

The S3 bucket requires authentication; this script uses the DANDI API's
presigned-URL mechanism (/zarr/{id}/files/?prefix=...&download=true)
to read individual zarr chunks and assemble the requested sub-volume.

Usage
-----
    export DANDI_API_KEY=<your-lincbrain-api-key>
    python download_patch.py \
        --zarr_id   b8418d95-409f-4b87-89be-93a9ba240f5f \
        --level     0 \
        --bbox      1461,210,3295,3323,1180,4889 \
        --output    /scratch/experiment/hipct/patch_I74_IC_zoom01.npy

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
    p = argparse.ArgumentParser(description="Download OME-Zarr bounding-box patch from LINCBrain")
    p.add_argument("--zarr_id", required=True, help="Zarr blob UUID (from asset contentUrl)")
    p.add_argument("--level", type=int, default=0, help="Resolution pyramid level (0 = full res)")
    p.add_argument("--bbox", required=True,
                   help="Bounding box as x0,y0,z0,x1,y1,z1 (voxel coords, exclusive end)")
    p.add_argument("--output", required=True, help="Output path (.npy)")
    p.add_argument("--api_key", default=None,
                   help="LINCBrain API key (default: $DANDI_API_KEY env var)")
    return p.parse_args()


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


def main():
    args = parse_args()
    api_key = args.api_key or os.environ.get("DANDI_API_KEY")
    if not api_key:
        sys.exit("Error: provide --api_key or set DANDI_API_KEY env var")

    bbox = list(map(int, args.bbox.split(",")))
    if len(bbox) != 6:
        sys.exit("Error: --bbox must have 6 comma-separated ints: x0,y0,z0,x1,y1,z1")
    x0, y0, z0, x1, y1, z1 = bbox

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Authenticated session
    session = requests.Session()
    session.headers.update({"Authorization": f"token {api_key}"})

    # --- Read array metadata ---
    zarray = get_zarr_json(session, args.zarr_id, f"{args.level}/.zarray")
    shape = zarray["shape"]       # [X, Y, Z] for this dataset
    chunks = zarray["chunks"]     # [cx, cy, cz]
    dtype = np.dtype(zarray["dtype"])
    print(f"Level {args.level}: shape={shape}, chunks={chunks}, dtype={dtype}")
    print(f"Bounding box: ({x0},{y0},{z0}) → ({x1},{y1},{z1})")
    print(f"Patch size: {x1-x0} × {y1-y0} × {z1-z0} = "
          f"{(x1-x0)*(y1-y0)*(z1-z0):,} voxels")

    # Validate bounds
    for i, (lo, hi, s) in enumerate(zip([x0, y0, z0], [x1, y1, z1], shape)):
        if lo < 0 or hi > s or lo >= hi:
            sys.exit(f"Error: bbox dim {i} [{lo}:{hi}] out of range [0:{s}]")

    # --- Determine which chunks we need ---
    cx, cy, cz = chunks
    ci0, ci1 = x0 // cx, math.ceil(x1 / cx)
    cj0, cj1 = y0 // cy, math.ceil(y1 / cy)
    ck0, ck1 = z0 // cz, math.ceil(z1 / cz)
    n_chunks = (ci1 - ci0) * (cj1 - cj0) * (ck1 - ck0)
    print(f"Chunk grid: x[{ci0}:{ci1}] y[{cj0}:{cj1}] z[{ck0}:{ck1}]  "
          f"({n_chunks} chunks to fetch)")

    # --- Allocate output as a disk-backed memmap (avoids OOM on login nodes) ---
    tmp_path = Path(str(output_path) + ".tmp")
    patch_shape = (x1 - x0, y1 - y0, z1 - z0)
    patch = np.memmap(str(tmp_path), dtype=dtype, mode="w+", shape=patch_shape)
    print(f"Allocated memmap: {tmp_path}  "
          f"({patch.nbytes / 1e9:.1f} GB on disk)")

    # Blosc decompressor (matches the zarr compressor)
    import blosc
    fill_value = zarray.get("fill_value", 0)
    order = zarray.get("order", "C")

    done = 0
    for ci in range(ci0, ci1):
        for cj in range(cj0, cj1):
            for ck in range(ck0, ck1):
                done += 1
                chunk_path = f"{args.level}/{ci}/{cj}/{ck}"

                # Global voxel range covered by this chunk
                gx0, gx1 = ci * cx, min((ci + 1) * cx, shape[0])
                gy0, gy1 = cj * cy, min((cj + 1) * cy, shape[1])
                gz0, gz1 = ck * cz, min((ck + 1) * cz, shape[2])

                # Intersection with our bounding box
                ix0 = max(gx0, x0)
                ix1 = min(gx1, x1)
                iy0 = max(gy0, y0)
                iy1 = min(gy1, y1)
                iz0 = max(gz0, z0)
                iz1 = min(gz1, z1)

                # Fetch chunk
                try:
                    raw = get_zarr_bytes(session, args.zarr_id, chunk_path)
                    # Decompress with blosc
                    decompressed = blosc.decompress(raw)
                    chunk_shape = (gx1 - gx0, gy1 - gy0, gz1 - gz0)
                    arr = np.frombuffer(decompressed, dtype=dtype).reshape(
                        chunk_shape if order == "C" else chunk_shape[::-1]
                    )
                    if order == "F":
                        arr = arr.T
                except Exception as e:
                    print(f"  [{done}/{n_chunks}] {chunk_path}: error ({e}), filling with {fill_value}")
                    arr = np.full(
                        (gx1 - gx0, gy1 - gy0, gz1 - gz0),
                        fill_value, dtype=dtype
                    )

                # Slice into output
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

    # --- Save: rename memmap temp file to final output ---
    patch.flush()
    del patch  # close memmap

    # Save shape/dtype metadata alongside the raw file
    import json as _json
    meta = {"shape": list(patch_shape), "dtype": str(dtype)}
    meta_path = Path(str(output_path) + ".json")
    meta_path.write_text(_json.dumps(meta))

    tmp_path.rename(output_path)
    size_mb = output_path.stat().st_size / 1e6
    print(f"\nSaved {output_path}  ({size_mb:.0f} MB)")
    print(f"Shape: {patch_shape}, dtype: {dtype}")
    print(f"Load with: np.memmap(path, dtype='{dtype}', mode='r', shape={patch_shape})")


if __name__ == "__main__":
    main()
