#!/usr/bin/env python3
"""Collect complete metrics for annotated LSM patch inference runs."""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect eval_complete metrics from annotated LSM inference folders."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/egaibor/orcd/scratch/LSM_axonal_marker_annotated_patches"),
    )
    parser.add_argument("--eval-tag", default="eval_complete")
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Default: <data-root>/summary_<eval-tag>",
    )
    return parser.parse_args()


def safe_divide(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def get_nested(data: dict[str, Any], keys: list[str], default=None):
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def add_metric_fields(row: dict[str, Any], prefix: str, metrics: dict[str, Any] | None):
    if not metrics:
        return
    tp = int(metrics.get("tp", 0))
    fp = int(metrics.get("fp", 0))
    fn = int(metrics.get("fn", 0))
    tn = int(metrics.get("tn", 0))
    row.update(
        {
            f"{prefix}_dice": metrics.get("dice"),
            f"{prefix}_cldice": metrics.get("cldice"),
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
    )


def add_topology_fields(row: dict[str, Any], topology: dict[str, Any] | None):
    if not topology:
        return
    pred = topology.get("prediction", {})
    target = topology.get("target", {})
    error = topology.get("absolute_error", {})
    for key in ["betti0", "betti1", "betti2", "euler_characteristic"]:
        row[f"topology_pred_{key}"] = pred.get(key)
        row[f"topology_target_{key}"] = target.get(key)
        row[f"topology_abs_error_{key}"] = error.get(key)
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


def build_row(corrected_json: Path, data_root: Path, eval_tag: str) -> dict[str, Any]:
    data = json.loads(corrected_json.read_text())
    model_dir = corrected_json.parent
    model = model_dir.name
    patch = model_dir.parent.name
    dataset = model_dir.parent.parent.parent.name
    sweep_path = model_dir / f"{patch}_{eval_tag}_sweep.json"
    best_threshold, best_criterion = best_threshold_from_sweep(sweep_path)
    if best_threshold is None:
        best_threshold = threshold_from_prediction_path(data.get("prediction_path"))

    row: dict[str, Any] = {
        "dataset": dataset,
        "patch": patch,
        "model": model,
        "output_dir": str(model_dir),
        "eval_json": str(corrected_json),
        "sweep_json": str(sweep_path) if sweep_path.exists() else None,
        "prediction_path": data.get("prediction_path"),
        "target_path": data.get("target_path"),
        "valid_mask_path": data.get("valid_mask_path"),
        "best_threshold": best_threshold,
        "best_threshold_criterion": best_criterion,
        "shape": "x".join(str(value) for value in data.get("shape", [])),
    }
    try:
        row["relative_output_dir"] = str(model_dir.relative_to(data_root))
    except ValueError:
        row["relative_output_dir"] = str(model_dir)

    add_metric_fields(row, "raw", data.get("metrics"))
    add_metric_fields(row, "corrected", data.get("corrected_metrics"))
    add_topology_fields(row, data.get("topology"))
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n")


def main() -> int:
    args = parse_args()
    data_root = args.data_root
    pattern = f"*/inference/*/*/*_{args.eval_tag}_corrected.json"
    rows = [
        build_row(path, data_root, args.eval_tag)
        for path in sorted(data_root.glob(pattern))
    ]

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)

    for dataset, dataset_rows in sorted(by_dataset.items()):
        inference_dir = data_root / dataset / "inference"
        write_csv(inference_dir / f"summary_{args.eval_tag}.csv", dataset_rows)
        write_json(inference_dir / f"summary_{args.eval_tag}.json", dataset_rows)

    output_prefix = (
        Path(args.output_prefix)
        if args.output_prefix is not None
        else data_root / f"summary_{args.eval_tag}"
    )
    write_csv(output_prefix.with_suffix(".csv"), rows)
    write_json(output_prefix.with_suffix(".json"), rows)
    print(f"Collected {len(rows)} eval files from {data_root}")
    print(f"Wrote {output_prefix.with_suffix('.csv')}")
    print(f"Wrote {output_prefix.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
