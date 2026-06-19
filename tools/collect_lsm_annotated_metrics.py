#!/usr/bin/env python3
"""Collect complete metrics for annotated LSM patch inference and baseline runs."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from lsm_eval_utils import (
    add_metric_fields,
    add_topology_fields,
    best_threshold_from_sweep,
    otsu_threshold_from_metadata,
    threshold_from_prediction_path,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect eval_complete metrics from annotated LSM inference/baseline folders."
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


def build_row(corrected_json: Path, data_root: Path, eval_tag: str) -> dict[str, Any]:
    data = json.loads(corrected_json.read_text())
    model_dir = corrected_json.parent
    source = model_dir.parent.parent.name
    patch = model_dir.parent.name
    if source == "baseline":
        model = "otsu_baseline" if corrected_json.name.startswith(f"{patch}_otsu_") else "threshold_baseline"
    else:
        model = model_dir.name
    dataset = model_dir.parent.parent.parent.name
    sweep_path = model_dir / f"{patch}_{eval_tag}_sweep.json"
    best_threshold, best_criterion = best_threshold_from_sweep(sweep_path)
    if best_threshold is None:
        best_threshold = threshold_from_prediction_path(data.get("prediction_path"))
    if best_threshold is None and data.get("threshold") is not None:
        best_threshold = float(data["threshold"])
        best_criterion = "fixed_threshold"
    if best_threshold is None and model == "otsu_baseline":
        best_threshold = otsu_threshold_from_metadata(model_dir, patch)
        best_criterion = "otsu"

    row: dict[str, Any] = {
        "dataset": dataset,
        "patch": patch,
        "source": source,
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

    add_metric_fields(row, "raw", data.get("metrics"), include_cldice=True)
    add_metric_fields(row, "corrected", data.get("corrected_metrics"), include_cldice=True)
    add_topology_fields(row, data.get("topology"), include_connectivity=True)
    return row


def main() -> int:
    args = parse_args()
    data_root = args.data_root
    patterns = [
        f"*/inference/*/*/*_{args.eval_tag}_corrected.json",
        f"*/baseline/*/*/*_{args.eval_tag}_corrected.json",
    ]
    paths = []
    for pattern in patterns:
        paths.extend(data_root.glob(pattern))
    rows = [
        build_row(path, data_root, args.eval_tag)
        for path in sorted(paths)
    ]

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_dataset_source: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
        by_dataset_source[(row["dataset"], row["source"])].append(row)

    for dataset, dataset_rows in sorted(by_dataset.items()):
        dataset_dir = data_root / dataset
        write_csv(dataset_dir / f"summary_{args.eval_tag}.csv", dataset_rows)
        write_json(dataset_dir / f"summary_{args.eval_tag}.json", dataset_rows)

    for (dataset, source), source_rows in sorted(by_dataset_source.items()):
        source_dir = data_root / dataset / source
        write_csv(source_dir / f"summary_{args.eval_tag}.csv", source_rows)
        write_json(source_dir / f"summary_{args.eval_tag}.json", source_rows)

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
