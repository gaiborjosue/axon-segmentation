#!/usr/bin/env python3
"""Evaluate saved Otsu baseline masks for annotated LSM patches."""

import argparse
from pathlib import Path

from lsm_eval_utils import run_masked_eval, target_for_prediction_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate *_pred_otsu.nii.gz files under annotated LSM baseline folders."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/egaibor/orcd/scratch/LSM_axonal_marker_annotated_patches"),
    )
    parser.add_argument("--eval-tag", default="eval_complete")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--correct-neighbor-rounds", type=int, default=2)
    parser.add_argument("--topology-connectivity", type=int, default=6, choices=[6, 26])
    parser.add_argument(
        "--compute-cldice",
        action="store_true",
        help="Compute clDice for Otsu eval. By default, Otsu eval skips clDice.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute Otsu eval JSONs even if they already exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    otsu_paths = sorted(args.data_root.glob("*/baseline/*/threshold/*_pred_otsu.nii.gz"))
    if not otsu_paths:
        raise FileNotFoundError(f"No *_pred_otsu.nii.gz files found under {args.data_root}")

    completed = 0
    skipped = 0
    for prediction in otsu_paths:
        patch = prediction.parent.parent.name
        target = target_for_prediction_path(prediction)
        output_json = prediction.parent / f"{patch}_otsu_{args.eval_tag}_corrected.json"

        if not target.is_file():
            raise FileNotFoundError(f"Missing target for {prediction}: {target}")
        if output_json.exists() and not args.overwrite:
            print(f"Skipping existing {output_json}")
            skipped += 1
            continue

        print(f"Evaluating Otsu baseline: {prediction}")
        run_masked_eval(
            prediction=prediction,
            target=target,
            output_json=output_json,
            threshold=None,
            correct_neighbor_rounds=args.correct_neighbor_rounds,
            device=args.device,
            skip_cldice=not args.compute_cldice,
            topology_metrics=True,
            topology_connectivity=args.topology_connectivity,
        )
        completed += 1

    print(f"Completed {completed} Otsu evals; skipped {skipped} existing evals.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
