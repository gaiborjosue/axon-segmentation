#!/usr/bin/env python3
"""Evaluate learned LSM inference outputs at a fixed probability threshold."""

import argparse
from pathlib import Path

from lsm_eval_utils import run_masked_eval, target_for_prediction_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate binary/threeclass annotated LSM predictions at one fixed threshold."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/egaibor/orcd/scratch/LSM_axonal_marker_annotated_patches"),
    )
    parser.add_argument("--eval-tag", default="eval_fixed_t050")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--correct-neighbor-rounds", type=int, default=2)
    parser.add_argument("--topology-connectivity", type=int, default=6, choices=[6, 26])
    parser.add_argument(
        "--models",
        nargs="+",
        default=["binary", "threeclass"],
        choices=["binary", "threeclass"],
        help="Inference model folders to evaluate.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute JSONs even if they already exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prediction_paths: list[Path] = []
    for model in args.models:
        prediction_paths.extend(
            sorted(args.data_root.glob(f"*/inference/*/{model}/*_pred_prob.nii.gz"))
        )
    if not prediction_paths:
        raise FileNotFoundError(f"No inference *_pred_prob.nii.gz files found under {args.data_root}")

    completed = 0
    skipped = 0
    for prediction in prediction_paths:
        patch = prediction.parent.parent.name
        target = target_for_prediction_path(prediction)
        output_json = prediction.parent / f"{patch}_{args.eval_tag}_corrected.json"

        if not target.is_file():
            raise FileNotFoundError(f"Missing target for {prediction}: {target}")
        if output_json.exists() and not args.overwrite:
            print(f"Skipping existing {output_json}")
            skipped += 1
            continue

        print(f"Evaluating fixed threshold {args.threshold:.3f}: {prediction}")
        run_masked_eval(
            prediction=prediction,
            target=target,
            output_json=output_json,
            threshold=args.threshold,
            correct_neighbor_rounds=args.correct_neighbor_rounds,
            device=args.device,
            skip_cldice=True,
            topology_metrics=True,
            topology_connectivity=args.topology_connectivity,
            quiet=True,
        )
        completed += 1

    print(f"Completed {completed} fixed-threshold evals; skipped {skipped} existing evals.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
