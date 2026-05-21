#!/usr/bin/env python3
"""Save a thresholded binary mask from a sweep JSON and probability volume."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize the best thresholded prediction from a sweep JSON."
    )
    parser.add_argument("--probability", type=Path, required=True)
    parser.add_argument("--sweep-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument(
        "--criterion",
        choices=["auto", "corrected_dice", "dice"],
        default="auto",
        help="Which sweep summary field to use when picking the best threshold.",
    )
    return parser.parse_args()


def _resolve_best_threshold(summary: dict, criterion: str) -> float:
    if criterion == "auto":
        if "best_by_corrected_dice" in summary:
            return float(summary["best_by_corrected_dice"]["threshold"])
        return float(summary["best_by_dice"]["threshold"])
    if criterion == "corrected_dice":
        return float(summary["best_by_corrected_dice"]["threshold"])
    return float(summary["best_by_dice"]["threshold"])


def main() -> int:
    args = parse_args()
    summary = json.loads(args.sweep_json.read_text())
    threshold = _resolve_best_threshold(summary, args.criterion)

    nii = nib.load(str(args.probability))
    probability = np.asarray(nii.dataobj)
    prediction = (probability >= threshold).astype(np.uint8)

    threshold_tag = int(round(threshold * 100.0))
    output_path = args.output_dir / f"{args.output_prefix}_pred_t{threshold_tag:03d}.nii.gz"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(prediction, affine=nii.affine, header=nii.header), str(output_path))

    result = {
        "probability_path": str(args.probability),
        "sweep_json_path": str(args.sweep_json),
        "criterion": args.criterion,
        "threshold": threshold,
        "output_path": str(output_path),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())