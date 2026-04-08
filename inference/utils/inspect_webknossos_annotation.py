#!/usr/bin/env python3
import argparse
import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import requests
import xml.etree.ElementTree as ET


def _headers(token: str) -> dict[str, str]:
    return {"X-Auth-Token": token}


def _fetch_info(base_url: str, api_version: int, annotation_id: str, token: str) -> dict[str, Any]:
    url = f"{base_url}/api/v{api_version}/annotations/{annotation_id}/info"
    response = requests.get(url, headers=_headers(token), timeout=60)
    response.raise_for_status()
    return response.json()


def _fetch_archive(
    base_url: str,
    api_version: int,
    annotation_id: str,
    token: str,
    include_volume: bool,
    volume_format: str,
) -> bytes:
    url = f"{base_url}/api/v{api_version}/annotations/{annotation_id}/download"
    params = {
        "skipVolumeData": str(not include_volume).lower(),
        "volumeDataZipFormat": volume_format,
    }
    response = requests.get(url, headers=_headers(token), params=params, timeout=300)
    response.raise_for_status()
    return response.content


def _parse_nml(nml_text: str) -> dict[str, Any]:
    root = ET.fromstring(nml_text)
    params = root.find("parameters")
    if params is None:
        return {}

    experiment = params.find("experiment")
    scale = params.find("scale")
    edit_position = params.find("editPosition")
    zoom = params.find("zoomLevel")

    user_boxes = []
    for box in params.findall("userBoundingBox"):
        user_boxes.append(
            {
                "id": box.attrib.get("id"),
                "name": box.attrib.get("name"),
                "top_left": [
                    int(box.attrib["topLeftX"]),
                    int(box.attrib["topLeftY"]),
                    int(box.attrib["topLeftZ"]),
                ],
                "size": [
                    int(box.attrib["width"]),
                    int(box.attrib["height"]),
                    int(box.attrib["depth"]),
                ],
            }
        )

    segments = []
    for segment in root.findall("./volume/segments/segment"):
        segments.append(
            {
                "id": segment.attrib.get("id"),
                "name": segment.attrib.get("name"),
                "anchor_position": [
                    int(segment.attrib["anchorPositionX"]),
                    int(segment.attrib["anchorPositionY"]),
                    int(segment.attrib["anchorPositionZ"]),
                ],
            }
        )

    return {
        "experiment": dict(experiment.attrib) if experiment is not None else None,
        "scale": dict(scale.attrib) if scale is not None else None,
        "edit_position": dict(edit_position.attrib) if edit_position is not None else None,
        "zoom_level": dict(zoom.attrib) if zoom is not None else None,
        "user_bounding_boxes": user_boxes,
        "segments": segments,
    }


def _read_archive_metadata(archive_bytes: bytes) -> dict[str, Any]:
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as outer_zip:
        names = outer_zip.namelist()
        nml_name = next((name for name in names if name.endswith(".nml")), None)
        if nml_name is None:
            return {"archive_entries": names}

        nml_text = outer_zip.read(nml_name).decode("utf-8", errors="replace")
        metadata = {
            "archive_entries": names,
            "nml_name": nml_name,
            "nml": _parse_nml(nml_text),
        }

        if "data_Volume.zip" in names:
            with zipfile.ZipFile(io.BytesIO(outer_zip.read("data_Volume.zip"))) as inner_zip:
                inner_names = inner_zip.namelist()
                metadata["volume_archive_entries_preview"] = inner_names[:200]
                metadata["volume_archive_entry_count"] = len(inner_names)

        return metadata


def _build_summary(info: dict[str, Any], archive_metadata: dict[str, Any] | None) -> dict[str, Any]:
    summary = {
        "annotation_id": info.get("id"),
        "name": info.get("name"),
        "organization": info.get("organization"),
        "dataset_name": info.get("dataSetName"),
        "dataset_id": info.get("datasetId"),
        "annotation_layers": info.get("annotationLayers", []),
        "restrictions": info.get("restrictions", {}),
        "visibility": info.get("visibility"),
    }

    if archive_metadata is not None:
        summary["archive"] = archive_metadata

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a WebKnossos annotation through the server's v9 REST API.")
    parser.add_argument("annotation_id", help="WebKnossos annotation id")
    parser.add_argument(
        "--url",
        default="https://webknossos.lincbrain.org",
        help="Base WebKnossos URL",
    )
    parser.add_argument(
        "--api-version",
        type=int,
        default=9,
        help="Server API version to use. LINC currently supports up to v9.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="WebKnossos token. If omitted, WEBKNOSSOS_TOKEN is used.",
    )
    parser.add_argument(
        "--download",
        choices=["none", "metadata", "full"],
        default="metadata",
        help="Download no archive, the annotation without volume data, or the full archive.",
    )
    parser.add_argument(
        "--volume-format",
        choices=["wkw", "zarr3"],
        default="wkw",
        help="Requested volume archive format when --download=full.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to save the downloaded archive and summary JSON.",
    )
    args = parser.parse_args()

    token = args.token or __import__("os").environ.get("WEBKNOSSOS_TOKEN")
    if not token:
        print("Missing token. Pass --token or set WEBKNOSSOS_TOKEN.", file=sys.stderr)
        return 2

    info = _fetch_info(args.url.rstrip("/"), args.api_version, args.annotation_id, token)

    archive_metadata = None
    archive_bytes = None
    archive_name = None
    if args.download != "none":
        include_volume = args.download == "full"
        archive_bytes = _fetch_archive(
            args.url.rstrip("/"),
            args.api_version,
            args.annotation_id,
            token,
            include_volume=include_volume,
            volume_format=args.volume_format,
        )
        archive_metadata = _read_archive_metadata(archive_bytes)
        suffix = "full" if include_volume else "metadata"
        archive_name = f"{args.annotation_id}_{suffix}.zip"

    summary = _build_summary(info, archive_metadata)
    print(json.dumps(summary, indent=2))

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = args.output_dir / f"{args.annotation_id}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        if archive_bytes is not None and archive_name is not None:
            (args.output_dir / archive_name).write_bytes(archive_bytes)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())