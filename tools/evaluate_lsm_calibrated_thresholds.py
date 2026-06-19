#!/usr/bin/env python3
"""Calibrate fixed thresholds on two patches and evaluate held-out LSM patches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from lsm_eval_utils import (
    add_metric_fields,
    add_topology_fields,
    discover_lsm_patches,
    otsu_threshold_from_metadata,
    run_masked_eval,
    write_csv,
    write_json,
)


METHODS_TO_CALIBRATE = ["binary", "threeclass", "fixed_threshold_baseline"]
ALL_SUMMARY_METHODS = [*METHODS_TO_CALIBRATE, "otsu_baseline"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pick one threshold per method on calibration patches, then evaluate held-out "
            "annotated LSM patches without clDice."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/egaibor/orcd/scratch/LSM_axonal_marker_annotated_patches"),
    )
    parser.add_argument("--eval-tag", default="eval_calibrated_2patch")
    parser.add_argument(
        "--domain-specific",
        action="store_true",
        help=(
            "Calibrate separate thresholds by species/domain using the calibration "
            "patch from that domain, instead of one global threshold."
        ),
    )
    parser.add_argument(
        "--calibration-patch",
        action="append",
        default=None,
        help=(
            "Calibration patch as <dataset>/<patch>. Repeat twice. Default: "
            "human_NEFH/Human_NEFH_GM and macaque_PV/macaque_PV_WM_2."
        ),
    )
    parser.add_argument("--sweep-start", type=float, default=0.05)
    parser.add_argument("--sweep-stop", type=float, default=0.95)
    parser.add_argument("--sweep-step", type=float, default=0.01)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--correct-neighbor-rounds", type=int, default=2)
    parser.add_argument("--topology-connectivity", type=int, default=6, choices=[6, 26])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_calibration_keys(values: list[str] | None) -> list[tuple[str, str]]:
    values = values or [
        "human_NEFH/Human_NEFH_GM",
        "macaque_PV/macaque_PV_WM_2",
    ]
    if len(values) != 2:
        raise ValueError("Provide exactly two --calibration-patch values.")
    keys = []
    for value in values:
        parts = value.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Expected <dataset>/<patch>, got: {value}")
        keys.append((parts[0], parts[1]))
    return keys


def domain_from_dataset(dataset: str) -> str:
    return dataset.split("_", 1)[0]


def group_calibration_keys_by_domain(
    calibration_keys: list[tuple[str, str]],
) -> dict[str, list[tuple[str, str]]]:
    grouped: dict[str, list[tuple[str, str]]] = {}
    for key in calibration_keys:
        grouped.setdefault(domain_from_dataset(key[0]), []).append(key)
    return grouped


def prediction_path_for(data_root: Path, dataset: str, patch: str, method: str) -> Path:
    dataset_dir = data_root / dataset
    if method in {"binary", "threeclass"}:
        return dataset_dir / "inference" / patch / method / f"{patch}_pred_prob.nii.gz"
    if method == "fixed_threshold_baseline":
        return dataset_dir / "baseline" / patch / "threshold" / f"{patch}_pred_prob.nii.gz"
    if method == "otsu_baseline":
        return dataset_dir / "baseline" / patch / "threshold" / f"{patch}_pred_otsu.nii.gz"
    raise ValueError(f"Unknown method: {method}")


def output_json_for(data_root: Path, dataset: str, patch: str, method: str, eval_tag: str) -> Path:
    prediction = prediction_path_for(data_root, dataset, patch, method)
    if method == "fixed_threshold_baseline":
        return prediction.parent / f"{patch}_fixed_{eval_tag}_corrected.json"
    if method == "otsu_baseline":
        return prediction.parent / f"{patch}_otsu_eval_complete_corrected.json"
    return prediction.parent / f"{patch}_{eval_tag}_corrected.json"


def load_bool(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj).astype(bool)


def load_float(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)


def has_face_neighbor(mask: np.ndarray) -> np.ndarray:
    neighbors = np.zeros_like(mask, dtype=bool)
    neighbors[1:, :, :] |= mask[:-1, :, :]
    neighbors[:-1, :, :] |= mask[1:, :, :]
    neighbors[:, 1:, :] |= mask[:, :-1, :]
    neighbors[:, :-1, :] |= mask[:, 1:, :]
    neighbors[:, :, 1:] |= mask[:, :, :-1]
    neighbors[:, :, :-1] |= mask[:, :, 1:]
    return neighbors


def apply_neighbor_correction(prediction: np.ndarray, target: np.ndarray, rounds: int) -> np.ndarray:
    corrected = prediction.astype(bool).copy()
    truth = target.astype(bool)
    tp_seed = corrected & truth
    fn_remaining = truth & ~corrected
    fp_remaining = corrected & ~truth
    fn_seed = tp_seed.copy()
    fp_seed = tp_seed.copy()

    for _ in range(rounds):
        fn_fix = fn_remaining & has_face_neighbor(fn_seed)
        fp_fix = fp_remaining & has_face_neighbor(fp_seed)
        if not fn_fix.any() and not fp_fix.any():
            break
        corrected[fn_fix] = True
        corrected[fp_fix] = False
        fn_remaining &= ~fn_fix
        fp_remaining &= ~fp_fix
        fn_seed |= fn_fix
        fp_seed |= fp_fix
    return corrected


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = prediction.astype(bool)
    truth = target.astype(bool)
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    denom = 2 * tp + fp + fn
    return float(2 * tp) / float(denom) if denom else 0.0


def threshold_values(start: float, stop: float, step: float) -> list[float]:
    values = np.arange(start, stop + 0.5 * step, step, dtype=np.float64)
    return [round(float(value), 6) for value in values]


def calibrate_threshold(
    data_root: Path,
    records: dict[tuple[str, str], dict[str, Path]],
    calibration_keys: list[tuple[str, str]],
    method: str,
    thresholds: list[float],
    rounds: int,
) -> dict[str, Any]:
    calibration_data = []
    for dataset, patch in calibration_keys:
        prediction = prediction_path_for(data_root, dataset, patch, method)
        if not prediction.is_file():
            raise FileNotFoundError(f"Missing prediction for calibration: {prediction}")
        target = records[(dataset, patch)]["target"]
        calibration_data.append(
            {
                "dataset": dataset,
                "patch": patch,
                "prediction": prediction,
                "score": load_float(prediction),
                "target": load_bool(target),
            }
        )

    evaluations = []
    for threshold in thresholds:
        raw_scores = []
        corrected_scores = []
        for item in calibration_data:
            pred = item["score"] >= threshold
            raw_scores.append(dice(pred, item["target"]))
            corrected = apply_neighbor_correction(pred, item["target"], rounds=rounds)
            corrected_scores.append(dice(corrected, item["target"]))
        evaluations.append(
            {
                "threshold": threshold,
                "mean_dice": float(np.mean(raw_scores)),
                "mean_corrected_dice": float(np.mean(corrected_scores)),
                "per_patch_dice": raw_scores,
                "per_patch_corrected_dice": corrected_scores,
            }
        )

    best = max(
        evaluations,
        key=lambda item: (item["mean_corrected_dice"], item["mean_dice"]),
    )
    return {
        "method": method,
        "selected_threshold": float(best["threshold"]),
        "selection_metric": "mean_corrected_dice",
        "best_calibration_result": best,
        "evaluations": evaluations,
    }


def build_summary_row(
    data_root: Path,
    dataset: str,
    patch: str,
    method: str,
    eval_json: Path,
    threshold: float | None,
    calibration_keys: list[tuple[str, str]],
    eval_tag: str,
    calibration_domain: str,
) -> dict[str, Any]:
    data = json.loads(eval_json.read_text())
    row: dict[str, Any] = {
        "dataset": dataset,
        "patch": patch,
        "source": "baseline" if "baseline" in method else "inference",
        "model": method,
        "calibration_role": "heldout",
        "calibration_domain": calibration_domain,
        "calibration_patches": ";".join(f"{d}/{p}" for d, p in calibration_keys),
        "eval_tag": eval_tag,
        "eval_json": str(eval_json),
        "prediction_path": data.get("prediction_path"),
        "target_path": data.get("target_path"),
        "valid_mask_path": data.get("valid_mask_path"),
        "best_threshold": threshold,
        "best_threshold_criterion": "calibrated_corrected_dice" if method != "otsu_baseline" else "otsu",
        "shape": "x".join(str(value) for value in data.get("shape", [])),
    }
    add_metric_fields(row, "raw", data.get("metrics"))
    add_metric_fields(row, "corrected", data.get("corrected_metrics"))
    add_topology_fields(row, data.get("topology"))
    return row


def main() -> int:
    args = parse_args()
    records = discover_lsm_patches(args.data_root)
    calibration_keys = parse_calibration_keys(args.calibration_patch)
    missing_keys = [key for key in calibration_keys if key not in records]
    if missing_keys:
        raise KeyError(f"Calibration patches not found: {missing_keys}")

    thresholds = threshold_values(args.sweep_start, args.sweep_stop, args.sweep_step)
    if args.domain_specific:
        calibration_keys_by_domain = group_calibration_keys_by_domain(calibration_keys)
        heldout_domains = {domain_from_dataset(dataset) for dataset, _ in records}
        missing_domains = sorted(heldout_domains - set(calibration_keys_by_domain))
        if missing_domains:
            raise ValueError(
                "Domain-specific calibration requested, but no calibration patch was "
                f"provided for domains: {missing_domains}"
            )
        calibration_results = {
            domain: {
                method: calibrate_threshold(
                    args.data_root,
                    records,
                    domain_calibration_keys,
                    method,
                    thresholds,
                    args.correct_neighbor_rounds,
                )
                for method in METHODS_TO_CALIBRATE
            }
            for domain, domain_calibration_keys in calibration_keys_by_domain.items()
        }
    else:
        calibration_keys_by_domain = {"global": calibration_keys}
        calibration_results = {
            "global": {
                method: calibrate_threshold(
                    args.data_root,
                    records,
                    calibration_keys,
                    method,
                    thresholds,
                    args.correct_neighbor_rounds,
                )
                for method in METHODS_TO_CALIBRATE
            }
        }

    heldout_keys = [key for key in sorted(records) if key not in calibration_keys]
    rows = []
    for dataset, patch in heldout_keys:
        target = records[(dataset, patch)]["target"]
        calibration_domain = domain_from_dataset(dataset) if args.domain_specific else "global"
        active_calibration_keys = calibration_keys_by_domain[calibration_domain]
        for method in METHODS_TO_CALIBRATE:
            prediction = prediction_path_for(args.data_root, dataset, patch, method)
            output_json = output_json_for(args.data_root, dataset, patch, method, args.eval_tag)
            threshold = calibration_results[calibration_domain][method]["selected_threshold"]
            if args.overwrite or not output_json.is_file():
                run_masked_eval(
                    prediction=prediction,
                    target=target,
                    output_json=output_json,
                    threshold=threshold,
                    correct_neighbor_rounds=args.correct_neighbor_rounds,
                    device=args.device,
                    skip_cldice=True,
                    topology_metrics=True,
                    topology_connectivity=args.topology_connectivity,
                    quiet=True,
                )
            rows.append(
                build_summary_row(
                    args.data_root,
                    dataset,
                    patch,
                    method,
                    output_json,
                    threshold,
                    active_calibration_keys,
                    args.eval_tag,
                    calibration_domain,
                )
            )

        otsu_json = output_json_for(args.data_root, dataset, patch, "otsu_baseline", args.eval_tag)
        if not otsu_json.is_file():
            raise FileNotFoundError(
                f"Missing Otsu eval JSON for held-out patch {dataset}/{patch}: {otsu_json}"
            )
        rows.append(
            build_summary_row(
                args.data_root,
                dataset,
                patch,
                "otsu_baseline",
                otsu_json,
                otsu_threshold_from_metadata(otsu_json.parent, patch),
                active_calibration_keys,
                args.eval_tag,
                calibration_domain,
            )
        )

    output_prefix = args.data_root / f"summary_{args.eval_tag}"
    write_csv(output_prefix.with_suffix(".csv"), rows)
    write_json(output_prefix.with_suffix(".json"), rows)

    metadata = {
        "eval_tag": args.eval_tag,
        "data_root": str(args.data_root),
        "calibration_patches": [
            {"dataset": dataset, "patch": patch}
            for dataset, patch in calibration_keys
        ],
        "domain_specific": bool(args.domain_specific),
        "calibration_patches_by_domain": {
            domain: [
                {"dataset": dataset, "patch": patch}
                for dataset, patch in domain_calibration_keys
            ]
            for domain, domain_calibration_keys in calibration_keys_by_domain.items()
        },
        "heldout_patches": [
            {"dataset": dataset, "patch": patch}
            for dataset, patch in heldout_keys
        ],
        "threshold_grid": {
            "start": args.sweep_start,
            "stop": args.sweep_stop,
            "step": args.sweep_step,
        },
        "selection_metric": "mean_corrected_dice",
        "cldice_enabled": False,
        "calibration_results": calibration_results,
    }
    metadata_path = args.data_root / f"calibration_{args.eval_tag}.json"
    write_json(metadata_path, metadata)
    print(f"Calibration patches: {metadata['calibration_patches']}")
    print(f"Domain-specific: {args.domain_specific}")
    print("Selected thresholds:")
    for domain, domain_results in calibration_results.items():
        print(f"  [{domain}]")
        for method, result in domain_results.items():
            print(f"    {method}: {result['selected_threshold']:.3f}")
    print(f"Wrote {output_prefix.with_suffix('.csv')}")
    print(f"Wrote {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
