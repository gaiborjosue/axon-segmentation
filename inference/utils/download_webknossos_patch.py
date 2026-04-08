#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import requests


API_VERSION = 9


ELEMENT_CLASS_TO_DTYPE = {
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "float32": np.float32,
    "float64": np.float64,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a WebKnossos dataset patch using an annotation summary or explicit dataset coordinates."
    )
    parser.add_argument("--annotation-summary", type=Path, default=None)
    parser.add_argument("--box-name", default=None)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--organization", default=None)
    parser.add_argument("--layer", default="0")
    parser.add_argument(
        "--bbox",
        default=None,
        help="Bounding box as x,y,z,width,height,depth",
    )
    parser.add_argument(
        "--mag",
        default="1-1-1",
        help="Magnification string accepted by the datastore endpoint, e.g. 1-1-1",
    )
    parser.add_argument("--url", default="https://webknossos.lincbrain.org")
    parser.add_argument("--token", default=None, help="WebKnossos API token")
    parser.add_argument(
        "--sharing-token",
        default=None,
        help="Optional dataset sharing token. If omitted, one is fetched via the API token.",
    )
    parser.add_argument("--output", required=True, help="Output path (.npy or .raw)")
    parser.add_argument(
        "--padding-value",
        type=float,
        default=0.0,
        help="Fill value for voxels that fall outside the dataset bounds",
    )
    return parser.parse_args()


def _parse_bbox_text(text: str) -> list[int]:
    values = [int(value.strip()) for value in text.split(",")]
    if len(values) != 6:
        raise ValueError("bbox must be x,y,z,width,height,depth")
    return values


def _load_annotation_summary(path: Path, box_name: str | None) -> dict[str, Any]:
    summary = json.loads(path.read_text())
    if box_name is None:
        raise ValueError("--box-name is required when using --annotation-summary")

    boxes = summary["archive"]["nml"]["user_bounding_boxes"]
    matches = [box for box in boxes if box["name"] == box_name]
    if not matches:
        raise ValueError(f"No user bounding box named {box_name!r} found in annotation summary")

    box = matches[0]
    return {
        "dataset_id": summary["dataset_id"],
        "dataset_name": summary["dataset_name"],
        "organization": summary["organization"],
        "bbox": [*box["top_left"], *box["size"]],
    }


def _api_headers(token: str | None) -> dict[str, str]:
    return {} if token is None else {"X-Auth-Token": token}


def _fetch_dataset_info(base_url: str, dataset_id: str, token: str) -> dict[str, Any]:
    response = requests.get(
        f"{base_url}/api/v{API_VERSION}/datasets/{dataset_id}",
        headers=_api_headers(token),
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _fetch_sharing_token(base_url: str, dataset_id: str, token: str) -> str:
    response = requests.get(
        f"{base_url}/api/v{API_VERSION}/datasets/{dataset_id}/sharingToken",
        headers=_api_headers(token),
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["sharingToken"]


def _compute_overlap(requested: list[int], dataset_box: dict[str, Any]) -> dict[str, Any]:
    req_x, req_y, req_z, req_w, req_h, req_d = requested
    req_x1 = req_x + req_w
    req_y1 = req_y + req_h
    req_z1 = req_z + req_d

    data_x, data_y, data_z = dataset_box["topLeft"]
    data_w = dataset_box["width"]
    data_h = dataset_box["height"]
    data_d = dataset_box["depth"]
    data_x1 = data_x + data_w
    data_y1 = data_y + data_h
    data_z1 = data_z + data_d

    clip_x0 = max(req_x, data_x)
    clip_y0 = max(req_y, data_y)
    clip_z0 = max(req_z, data_z)
    clip_x1 = min(req_x1, data_x1)
    clip_y1 = min(req_y1, data_y1)
    clip_z1 = min(req_z1, data_z1)

    if clip_x0 >= clip_x1 or clip_y0 >= clip_y1 or clip_z0 >= clip_z1:
        raise ValueError("Requested bounding box does not overlap the dataset bounds")

    return {
        "requested": {
            "top_left": [req_x, req_y, req_z],
            "size": [req_w, req_h, req_d],
        },
        "clipped": {
            "top_left": [clip_x0, clip_y0, clip_z0],
            "size": [clip_x1 - clip_x0, clip_y1 - clip_y0, clip_z1 - clip_z0],
        },
        "insert_offset": [clip_x0 - req_x, clip_y0 - req_y, clip_z0 - req_z],
    }


def _download_patch(
    base_url: str,
    organization: str,
    dataset_name: str,
    layer: str,
    mag: str,
    sharing_token: str,
    clipped_box: dict[str, Any],
    dtype: np.dtype,
) -> np.ndarray:
    x, y, z = clipped_box["top_left"]
    width, height, depth = clipped_box["size"]
    dataset_name_encoded = quote(dataset_name, safe="")
    url = f"{base_url}/data/datasets/{organization}/{dataset_name_encoded}/layers/{layer}/data"
    response = requests.get(
        url,
        params={
            "mag": mag,
            "x": x,
            "y": y,
            "z": z,
            "width": width,
            "height": height,
            "depth": depth,
            "token": sharing_token,
        },
        timeout=300,
    )
    response.raise_for_status()

    missing = response.headers.get("MISSING-BUCKETS")
    if missing not in (None, "[]"):
        raise RuntimeError(f"Datastore response reported missing buckets: {missing}")

    array = np.frombuffer(response.content, dtype=dtype).reshape(
        (1, width, height, depth), order="F"
    )[0]
    return array


def _save_output(output_path: Path, patch: np.ndarray, metadata: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".npy":
        np.save(output_path, patch)
    else:
        memmap = np.memmap(str(output_path), dtype=patch.dtype, mode="w+", shape=patch.shape)
        memmap[...] = patch
        memmap.flush()
        del memmap
        Path(str(output_path) + ".json").write_text(
            json.dumps({"shape": list(patch.shape), "dtype": str(patch.dtype)}) + "\n"
        )

    Path(str(output_path) + ".meta.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> int:
    args = parse_args()

    if args.annotation_summary is not None:
        summary_data = _load_annotation_summary(args.annotation_summary, args.box_name)
        dataset_id = summary_data["dataset_id"]
        dataset_name = summary_data["dataset_name"]
        organization = summary_data["organization"]
        bbox = summary_data["bbox"]
    else:
        if not all([args.dataset_id, args.dataset_name, args.organization, args.bbox]):
            print(
                "Provide either --annotation-summary with --box-name, or --dataset-id, --dataset-name, --organization, and --bbox.",
                file=sys.stderr,
            )
            return 2
        dataset_id = args.dataset_id
        dataset_name = args.dataset_name
        organization = args.organization
        bbox = _parse_bbox_text(args.bbox)

    token = args.token or __import__("os").environ.get("WEBKNOSSOS_TOKEN")
    if args.sharing_token is None and not token:
        print("Provide --sharing-token or a WebKnossos API token via --token/WEBKNOSSOS_TOKEN.", file=sys.stderr)
        return 2

    dataset_info = _fetch_dataset_info(args.url.rstrip("/"), dataset_id, token) if token else None
    if dataset_info is None:
        print("Fetching dataset metadata requires an API token.", file=sys.stderr)
        return 2

    data_layers = dataset_info["dataSource"]["dataLayers"]
    layer_matches = [layer for layer in data_layers if layer["name"] == args.layer]
    if not layer_matches:
        raise ValueError(f"Layer {args.layer!r} not found in dataset metadata")
    layer_info = layer_matches[0]

    dtype = np.dtype(ELEMENT_CLASS_TO_DTYPE[layer_info["elementClass"]])
    overlap = _compute_overlap(bbox, layer_info["boundingBox"])
    sharing_token = args.sharing_token or _fetch_sharing_token(args.url.rstrip("/"), dataset_id, token)

    clipped_patch = _download_patch(
        base_url=args.url.rstrip("/"),
        organization=organization,
        dataset_name=dataset_name,
        layer=args.layer,
        mag=args.mag,
        sharing_token=sharing_token,
        clipped_box=overlap["clipped"],
        dtype=dtype,
    )

    req_w, req_h, req_d = overlap["requested"]["size"]
    patch = np.full((req_w, req_h, req_d), args.padding_value, dtype=dtype)
    off_x, off_y, off_z = overlap["insert_offset"]
    clip_w, clip_h, clip_d = overlap["clipped"]["size"]
    patch[
        off_x : off_x + clip_w,
        off_y : off_y + clip_h,
        off_z : off_z + clip_d,
    ] = clipped_patch

    metadata = {
        "source": {
            "url": args.url,
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "organization": organization,
            "layer": args.layer,
            "mag": args.mag,
        },
        "bbox": overlap,
        "dtype": str(dtype),
        "shape": list(patch.shape),
        "padding_value": args.padding_value,
    }

    output_path = Path(args.output)
    _save_output(output_path, patch, metadata)

    print(f"Saved {output_path}")
    print(f"Shape: {patch.shape}, dtype: {patch.dtype}")
    print(f"Requested box: {overlap['requested']}")
    print(f"Clipped download: {overlap['clipped']}")
    print(f"Insert offset: {overlap['insert_offset']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())