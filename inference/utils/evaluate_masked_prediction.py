#!/usr/bin/env python3
"""Evaluate a predicted segmentation against a manual binary mask inside a valid region.

Example
-------
    python evaluate_masked_prediction.py \
        --prediction /scratch/experiment/webknossos/inference/macaque_NEFH_WM/higher_range_ep200_percentile/macaque_NEFH_WM_pred.nii.gz \
        --target /scratch/experiment/webknossos/macaque_NEFH_WM_binary.npy \
        --valid-mask /scratch/experiment/webknossos/macaque_NEFH_WM_valid_mask.npy \
        --output-json /scratch/experiment/webknossos/inference/macaque_NEFH_WM/higher_range_ep200_percentile/macaque_NEFH_WM_eval.json

    python evaluate_masked_prediction.py \
        --prediction /scratch/experiment/webknossos/inference/macaque_NEFH_WM/higher_range_ep200_percentile/macaque_NEFH_WM_pred_prob.nii.gz \
        --target /scratch/experiment/webknossos/macaque_NEFH_WM_binary.npy \
        --valid-mask /scratch/experiment/webknossos/macaque_NEFH_WM_valid_mask.npy \
        --sweep-start 0.05 \
        --sweep-stop 0.95 \
        --sweep-step 0.01 \
        --output-json /scratch/experiment/webknossos/inference/macaque_NEFH_WM/higher_range_ep200_percentile/macaque_NEFH_WM_eval_sweep.json
"""

import argparse
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a predicted segmentation inside a valid-data mask."
    )
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--valid-mask", type=Path, required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional threshold for probability-valued prediction volumes.",
    )
    parser.add_argument(
        "--sweep-start",
        type=float,
        default=None,
        help="Optional threshold-sweep start value.",
    )
    parser.add_argument(
        "--sweep-stop",
        type=float,
        default=None,
        help="Optional threshold-sweep stop value.",
    )
    parser.add_argument(
        "--sweep-step",
        type=float,
        default=None,
        help="Optional threshold-sweep step size.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _load_array(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".nii" or path.name.endswith(".nii.gz"):
        return np.asarray(nib.load(str(path)).dataobj)
    raise ValueError(f"Unsupported file type: {path}")


def _safe_divide(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _binarize_prediction(prediction: np.ndarray, threshold: float | None) -> np.ndarray:
    if threshold is not None:
        return prediction >= threshold

    unique_values = np.unique(prediction)
    if np.all(np.isin(unique_values, [0, 1])):
        return prediction.astype(bool)

    raise ValueError(
        "Prediction is not binary. Pass --threshold if evaluating a probability map."
    )


def _resolve_sweep_thresholds(args: argparse.Namespace) -> list[float] | None:
    sweep_values = [args.sweep_start, args.sweep_stop, args.sweep_step]
    if not any(value is not None for value in sweep_values):
        return None

    if not all(value is not None for value in sweep_values):
        raise ValueError("Provide --sweep-start, --sweep-stop, and --sweep-step together.")
    if args.threshold is not None:
        raise ValueError("Use either --threshold or --sweep-*, not both.")
    if args.sweep_step <= 0:
        raise ValueError("--sweep-step must be positive.")
    if args.sweep_stop < args.sweep_start:
        raise ValueError("--sweep-stop must be >= --sweep-start.")
    if args.sweep_start < 0.0 or args.sweep_stop > 1.0:
        raise ValueError("Sweep thresholds must lie within [0, 1].")

    thresholds = np.arange(
        args.sweep_start,
        args.sweep_stop + 0.5 * args.sweep_step,
        args.sweep_step,
        dtype=np.float64,
    )
    return [round(float(threshold), 6) for threshold in thresholds]


def _compute_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, Any]:
    valid = valid_mask.astype(bool)
    pred = prediction.astype(bool) & valid
    truth = target.astype(bool) & valid

    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, np.logical_not(truth) & valid).sum())
    fn = int(np.logical_and(np.logical_not(pred) & valid, truth).sum())
    tn = int(np.logical_and(np.logical_not(pred) & valid, np.logical_not(truth) & valid).sum())

    dice = _safe_divide(2 * tp, 2 * tp + fp + fn)
    iou = _safe_divide(tp, tp + fp + fn)
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    accuracy = _safe_divide(tp + tn, tp + tn + fp + fn)

    valid_voxels = int(valid.sum())
    target_positive = int(truth.sum())
    pred_positive = int(pred.sum())

    return {
        "valid_voxels": valid_voxels,
        "target_positive_voxels": target_positive,
        "pred_positive_voxels": pred_positive,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "target_positive_fraction": _safe_divide(target_positive, valid_voxels),
        "pred_positive_fraction": _safe_divide(pred_positive, valid_voxels),
    }


def main() -> int:
    args = parse_args()

    prediction_raw = _load_array(args.prediction)
    target_raw = _load_array(args.target)
    valid_mask_raw = _load_array(args.valid_mask)

    if prediction_raw.shape != target_raw.shape or prediction_raw.shape != valid_mask_raw.shape:
        raise ValueError(
            "Shape mismatch: "
            f"prediction={prediction_raw.shape}, target={target_raw.shape}, valid_mask={valid_mask_raw.shape}"
        )

    target = target_raw.astype(bool)
    valid_mask = valid_mask_raw.astype(bool)

    sweep_thresholds = _resolve_sweep_thresholds(args)
    if sweep_thresholds is not None:
        evaluations = []
        for threshold in sweep_thresholds:
            prediction = _binarize_prediction(prediction_raw, threshold)
            evaluations.append(
                {
                    "threshold": threshold,
                    "metrics": _compute_metrics(prediction, target, valid_mask),
                }
            )

        best_by_dice = max(
            evaluations,
            key=lambda item: (item["metrics"]["dice"], item["metrics"]["precision"]),
        )
        summary = {
            "prediction_path": str(args.prediction),
            "target_path": str(args.target),
            "valid_mask_path": str(args.valid_mask),
            "shape": list(prediction_raw.shape),
            "mode": "threshold_sweep",
            "sweep": {
                "start": args.sweep_start,
                "stop": args.sweep_stop,
                "step": args.sweep_step,
                "count": len(evaluations),
            },
            "best_by_dice": best_by_dice,
            "evaluations": evaluations,
        }
    else:
        prediction = _binarize_prediction(prediction_raw, args.threshold)
        metrics = _compute_metrics(prediction, target, valid_mask)
        summary = {
            "prediction_path": str(args.prediction),
            "target_path": str(args.target),
            "valid_mask_path": str(args.valid_mask),
            "shape": list(prediction_raw.shape),
            "mode": "single_threshold",
            "threshold": args.threshold,
            "metrics": metrics,
        }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Saved {args.output_json}")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())