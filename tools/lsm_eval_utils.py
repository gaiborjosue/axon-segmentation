#!/usr/bin/env python3
"""Shared helpers for annotated LSM evaluation scripts."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def patch_name_from_raw(raw_path: Path) -> str:
    name = raw_path.name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    else:
        name = raw_path.stem
    return name[:-4] if name.endswith("_raw") else name


def discover_lsm_patches(data_root: Path) -> dict[tuple[str, str], dict[str, Path]]:
    records: dict[tuple[str, str], dict[str, Path]] = {}
    for raw_path in sorted(data_root.glob("*/*_raw.nii.gz")):
        dataset = raw_path.parent.name
        patch = patch_name_from_raw(raw_path)
        target = raw_path.with_name(f"{patch}_gt.nii.gz")
        if not target.is_file():
            raise FileNotFoundError(f"Missing target for {raw_path}: {target}")
        records[(dataset, patch)] = {
            "raw": raw_path,
            "target": target,
        }
    return records


def patch_from_prediction_path(prediction: Path) -> str:
    return prediction.parent.parent.name


def dataset_dir_from_prediction_path(prediction: Path) -> Path:
    return prediction.parent.parent.parent.parent


def target_for_prediction_path(prediction: Path) -> Path:
    patch = patch_from_prediction_path(prediction)
    return dataset_dir_from_prediction_path(prediction) / f"{patch}_gt.nii.gz"


def run_masked_eval(
    *,
    prediction: Path,
    target: Path,
    output_json: Path,
    threshold: float | None = None,
    correct_neighbors: bool = True,
    correct_neighbor_rounds: int = 2,
    device: str = "cpu",
    skip_cldice: bool = True,
    topology_metrics: bool = True,
    topology_connectivity: int = 6,
    corrected_topology_metrics: bool = False,
    quiet: bool = False,
) -> None:
    script = Path("inference/utils/evaluate_masked_prediction.py")
    if not script.is_file():
        raise FileNotFoundError(f"Run from experiment repo root; missing {script}")

    cmd = [
        sys.executable,
        str(script),
        "--prediction",
        str(prediction),
        "--target",
        str(target),
        "--device",
        device,
        "--output-json",
        str(output_json),
    ]
    if threshold is not None:
        cmd.extend(["--threshold", str(threshold)])
    if correct_neighbors:
        cmd.extend(["--correct-neighbors", "--correct-neighbor-rounds", str(correct_neighbor_rounds)])
    if skip_cldice:
        cmd.append("--skip-cldice")
    if topology_metrics:
        cmd.extend(["--topology-metrics", "--topology-connectivity", str(topology_connectivity)])
        if corrected_topology_metrics:
            cmd.append("--corrected-topology-metrics")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    stdout = subprocess.DEVNULL if quiet else None
    subprocess.run(cmd, check=True, stdout=stdout)


def safe_divide(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def get_nested(data: dict[str, Any], keys: list[str], default=None):
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def add_metric_fields(
    row: dict[str, Any],
    prefix: str,
    metrics: dict[str, Any] | None,
    *,
    include_cldice: bool = False,
) -> None:
    if not metrics:
        return
    tp = int(metrics.get("tp", 0))
    fp = int(metrics.get("fp", 0))
    fn = int(metrics.get("fn", 0))
    tn = int(metrics.get("tn", 0))
    fields = {
        f"{prefix}_dice": metrics.get("dice"),
        f"{prefix}_iou": metrics.get("iou"),
        f"{prefix}_precision": metrics.get("precision"),
        f"{prefix}_tpr": metrics.get("recall"),
        f"{prefix}_fdr": safe_divide(fp, tp + fp),
        f"{prefix}_fpr": safe_divide(fp, fp + tn),
        f"{prefix}_tp": tp,
        f"{prefix}_fp": fp,
        f"{prefix}_fn": fn,
        f"{prefix}_tn": tn,
        f"{prefix}_target_positive_fraction": metrics.get("target_positive_fraction"),
        f"{prefix}_pred_positive_fraction": metrics.get("pred_positive_fraction"),
    }
    if include_cldice:
        fields[f"{prefix}_cldice"] = metrics.get("cldice")
    row.update(fields)


def add_topology_fields(
    row: dict[str, Any],
    topology: dict[str, Any] | None,
    *,
    include_connectivity: bool = False,
) -> None:
    if not topology:
        return
    for key in ["betti0", "betti1", "betti2", "euler_characteristic"]:
        row[f"topology_pred_{key}"] = get_nested(topology, ["prediction", key])
        row[f"topology_target_{key}"] = get_nested(topology, ["target", key])
        row[f"topology_abs_error_{key}"] = get_nested(topology, ["absolute_error", key])
    if include_connectivity:
        row["topology_foreground_connectivity"] = get_nested(
            topology, ["connectivity", "foreground"]
        )
        row["topology_background_connectivity"] = get_nested(
            topology, ["connectivity", "background"]
        )


def threshold_from_prediction_path(path: str | None) -> float | None:
    if not path:
        return None
    match = re.search(r"_pred_t(\d{3})\.nii(?:\.gz)?$", path)
    if not match:
        return None
    return int(match.group(1)) / 100.0


def best_threshold_from_sweep(sweep_path: Path) -> tuple[float | None, str | None]:
    if not sweep_path.is_file():
        return None, None
    summary = json.loads(sweep_path.read_text())
    if "best_by_corrected_dice" in summary:
        return float(summary["best_by_corrected_dice"]["threshold"]), "corrected_dice"
    if "best_by_dice" in summary:
        return float(summary["best_by_dice"]["threshold"]), "dice"
    return None, None


def otsu_threshold_from_metadata(model_dir: Path, patch: str) -> float | None:
    metadata_path = model_dir / f"{patch}_baseline_metadata.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text())
    threshold = metadata.get("otsu_threshold")
    return float(threshold) if threshold is not None else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n")
