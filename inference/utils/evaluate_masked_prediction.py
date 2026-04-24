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
import torch
import torch.nn.functional as F


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
    parser.add_argument(
        "--correct-neighbors",
        action="store_true",
        help=(
            "Apply lab-style neighbor correction before reporting corrected metrics. "
            "False positives/false negatives that are face-neighbors of a true positive "
            "can be relabeled over a small number of rounds."
        ),
    )
    parser.add_argument(
        "--correct-neighbor-rounds",
        type=int,
        default=2,
        help="Number of face-neighbor correction rounds to apply when --correct-neighbors is set.",
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
    cldice = _compute_cldice(pred, truth)

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
        "cldice": cldice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "target_positive_fraction": _safe_divide(target_positive, valid_voxels),
        "pred_positive_fraction": _safe_divide(pred_positive, valid_voxels),
    }


def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
    p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
    p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
    return torch.min(torch.min(p1, p2), p3)


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(img))


def _soft_skeletonize(img: torch.Tensor, num_iter: int = 10) -> torch.Tensor:
    opened = _soft_open(img)
    skeleton = F.relu(img - opened)

    for _ in range(num_iter):
        img = _soft_erode(img)
        opened = _soft_open(img)
        delta = F.relu(img - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)

    return skeleton


def _compute_cldice(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    num_iter: int = 10,
    smooth: float = 1.0,
) -> float:
    pred_tensor = torch.from_numpy(
        prediction.astype(np.float32, copy=False)
    ).unsqueeze(0).unsqueeze(0)
    target_tensor = torch.from_numpy(
        target.astype(np.float32, copy=False)
    ).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        skel_prediction = _soft_skeletonize(pred_tensor, num_iter=num_iter)
        skel_target = _soft_skeletonize(target_tensor, num_iter=num_iter)

        topology_precision = (
            torch.sum(skel_prediction * target_tensor) + smooth
        ) / (torch.sum(skel_prediction) + smooth)
        topology_sensitivity = (
            torch.sum(skel_target * pred_tensor) + smooth
        ) / (torch.sum(skel_target) + smooth)
        cldice = (
            2.0 * topology_precision * topology_sensitivity
        ) / (topology_precision + topology_sensitivity + smooth)

    return float(cldice.item())


def _has_face_neighbor(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool, copy=False)
    neighbors = np.zeros_like(mask, dtype=bool)

    neighbors[1:, :, :] |= mask[:-1, :, :]
    neighbors[:-1, :, :] |= mask[1:, :, :]
    neighbors[:, 1:, :] |= mask[:, :-1, :]
    neighbors[:, :-1, :] |= mask[:, 1:, :]
    neighbors[:, :, 1:] |= mask[:, :, :-1]
    neighbors[:, :, :-1] |= mask[:, :, 1:]

    return neighbors


def _apply_neighbor_correction(
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
    rounds: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    valid = valid_mask.astype(bool)
    truth = target.astype(bool) & valid
    corrected_prediction = prediction.astype(bool) & valid

    tp_seed = corrected_prediction & truth
    fn_remaining = truth & np.logical_not(corrected_prediction)
    fp_remaining = corrected_prediction & np.logical_not(truth)

    fn_seed = tp_seed.copy()
    fp_seed = tp_seed.copy()
    fn_promoted = np.zeros_like(truth, dtype=bool)
    fp_removed = np.zeros_like(truth, dtype=bool)
    rounds_applied = 0

    for _ in range(rounds):
        fn_fix = fn_remaining & _has_face_neighbor(fn_seed)
        fp_fix = fp_remaining & _has_face_neighbor(fp_seed)

        if not fn_fix.any() and not fp_fix.any():
            break

        corrected_prediction[fn_fix] = True
        corrected_prediction[fp_fix] = False

        fn_remaining &= np.logical_not(fn_fix)
        fp_remaining &= np.logical_not(fp_fix)
        fn_seed |= fn_fix
        fp_seed |= fp_fix
        fn_promoted |= fn_fix
        fp_removed |= fp_fix
        rounds_applied += 1

    correction = {
        "connectivity": "face-6",
        "rounds_requested": rounds,
        "rounds_applied": rounds_applied,
        "fn_promoted_to_tp_voxels": int(fn_promoted.sum()),
        "fp_removed_voxels": int(fp_removed.sum()),
    }
    return corrected_prediction, correction


def _evaluate_prediction(
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
    *,
    correct_neighbors: bool,
    correct_neighbor_rounds: int,
) -> dict[str, Any]:
    evaluation = {
        "metrics": _compute_metrics(prediction, target, valid_mask),
    }

    if correct_neighbors:
        corrected_prediction, correction = _apply_neighbor_correction(
            prediction,
            target,
            valid_mask,
            rounds=correct_neighbor_rounds,
        )
        evaluation["corrected_metrics"] = _compute_metrics(
            corrected_prediction,
            target,
            valid_mask,
        )
        evaluation["correction"] = correction

    return evaluation


def main() -> int:
    args = parse_args()

    if args.correct_neighbors and args.correct_neighbor_rounds < 1:
        raise ValueError("--correct-neighbor-rounds must be >= 1 when correction is enabled.")

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
            evaluation = _evaluate_prediction(
                prediction,
                target,
                valid_mask,
                correct_neighbors=args.correct_neighbors,
                correct_neighbor_rounds=args.correct_neighbor_rounds,
            )
            evaluations.append({"threshold": threshold, **evaluation})

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
        if args.correct_neighbors:
            summary["correction_method"] = {
                "enabled": True,
                "connectivity": "face-6",
                "neighbor_rounds": args.correct_neighbor_rounds,
            }
            summary["best_by_corrected_dice"] = max(
                evaluations,
                key=lambda item: (
                    item["corrected_metrics"]["dice"],
                    item["corrected_metrics"]["precision"],
                ),
            )
    else:
        prediction = _binarize_prediction(prediction_raw, args.threshold)
        evaluation = _evaluate_prediction(
            prediction,
            target,
            valid_mask,
            correct_neighbors=args.correct_neighbors,
            correct_neighbor_rounds=args.correct_neighbor_rounds,
        )
        summary = {
            "prediction_path": str(args.prediction),
            "target_path": str(args.target),
            "valid_mask_path": str(args.valid_mask),
            "shape": list(prediction_raw.shape),
            "mode": "single_threshold",
            "threshold": args.threshold,
            **evaluation,
        }
        if args.correct_neighbors:
            summary["correction_method"] = {
                "enabled": True,
                "connectivity": "face-6",
                "neighbor_rounds": args.correct_neighbor_rounds,
            }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Saved {args.output_json}")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())