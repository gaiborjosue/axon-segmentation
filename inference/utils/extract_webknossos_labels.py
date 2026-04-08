#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import wkw
import xml.etree.ElementTree as ET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract an aligned label volume and valid-data mask from a downloaded WebKnossos annotation archive."
    )
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--patch-meta", type=Path, required=True)
    parser.add_argument("--box-name", required=True)
    parser.add_argument(
        "--segment-id",
        type=int,
        default=None,
        help="Optional explicit segment id. If omitted, it is inferred from the NML by box name.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Output path prefix, e.g. /scratch/.../macaque_NEFH_WM",
    )
    return parser.parse_args()


def _load_patch_meta(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_nml(annotation_dir: Path) -> ET.Element:
    nml_files = sorted(annotation_dir.glob("extracted/*.nml"))
    if not nml_files:
        raise FileNotFoundError(f"No extracted NML file found under {annotation_dir}")
    return ET.fromstring(nml_files[0].read_text())


def _resolve_segment_id(root: ET.Element, box_name: str, segment_id: int | None) -> int:
    if segment_id is not None:
        return segment_id

    for segment in root.findall("./volume/segments/segment"):
        if segment.attrib.get("name") == box_name:
            return int(segment.attrib["id"])

    raise ValueError(f"No segment id found for box name {box_name!r}")


def _open_volume_dataset(annotation_dir: Path) -> wkw.Dataset:
    root = annotation_dir / "extracted" / "data_Volume" / "1"
    if not root.exists():
        raise FileNotFoundError(f"Expected extracted WKW dataset at {root}")
    return wkw.Dataset.open(str(root))


def _extract_labels(
    dataset: wkw.Dataset,
    clipped_top_left: list[int],
    clipped_size: list[int],
    requested_size: list[int],
    insert_offset: list[int],
    segment_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clipped = dataset.read(clipped_top_left, clipped_size)[0]

    labels_full = np.zeros(tuple(requested_size), dtype=clipped.dtype)
    valid_mask = np.zeros(tuple(requested_size), dtype=np.uint8)

    off_x, off_y, off_z = insert_offset
    size_x, size_y, size_z = clipped.shape

    labels_full[
        off_x : off_x + size_x,
        off_y : off_y + size_y,
        off_z : off_z + size_z,
    ] = clipped
    valid_mask[
        off_x : off_x + size_x,
        off_y : off_y + size_y,
        off_z : off_z + size_z,
    ] = 1

    binary = (labels_full == segment_id).astype(np.uint8)
    return labels_full, binary, valid_mask


def main() -> int:
    args = parse_args()

    patch_meta = _load_patch_meta(args.patch_meta)
    root = _load_nml(args.annotation_dir)
    segment_id = _resolve_segment_id(root, args.box_name, args.segment_id)
    dataset = _open_volume_dataset(args.annotation_dir)

    requested = patch_meta["bbox"]["requested"]
    clipped = patch_meta["bbox"]["clipped"]
    insert_offset = patch_meta["bbox"]["insert_offset"]

    labels_full, binary, valid_mask = _extract_labels(
        dataset=dataset,
        clipped_top_left=clipped["top_left"],
        clipped_size=clipped["size"],
        requested_size=requested["size"],
        insert_offset=insert_offset,
        segment_id=segment_id,
    )
    dataset.close()

    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    labels_path = Path(f"{output_prefix}_labels.npy")
    binary_path = Path(f"{output_prefix}_binary.npy")
    valid_path = Path(f"{output_prefix}_valid_mask.npy")
    meta_path = Path(f"{output_prefix}_labels.meta.json")

    np.save(labels_path, labels_full)
    np.save(binary_path, binary)
    np.save(valid_path, valid_mask)

    meta = {
        "box_name": args.box_name,
        "segment_id": segment_id,
        "requested_bbox": requested,
        "clipped_bbox": clipped,
        "insert_offset": insert_offset,
        "labels_path": str(labels_path),
        "binary_path": str(binary_path),
        "valid_mask_path": str(valid_path),
        "shape": list(labels_full.shape),
        "label_values": sorted(int(value) for value in np.unique(labels_full)),
        "positive_voxels": int(binary.sum()),
        "valid_voxels": int(valid_mask.sum()),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"Saved {labels_path}")
    print(f"Saved {binary_path}")
    print(f"Saved {valid_path}")
    print(f"Saved {meta_path}")
    print(f"Segment id: {segment_id}")
    print(f"Shape: {labels_full.shape}")
    print(f"Positive voxels: {int(binary.sum())}")
    print(f"Valid voxels: {int(valid_mask.sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())