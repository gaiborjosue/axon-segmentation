# Run Launchers

`run/` is organized by how likely a launcher is to still matter:

- `production/`: current end-to-end jobs you are likely to submit again.
- `smoke/`: short validation jobs for integration checks.
- `debug/`: focused profiling and environment checks.
- `legacy/`: older experiments, diagnostics, and superseded helpers kept for reference.

If you only need the main workflow, start here:

- `run/production/train_production.sbatch`: binary production training.
- `run/production/train_three_class_production.sbatch`: 3-class production training.
- `run/production/infer_lsm.sbatch`: LSM/WebKnossos inference.
- `run/production/infer_lsm_eval.sbatch`: LSM inference plus sweep and corrected evaluation.
- `run/smoke/train_three_class_smoke.sbatch`: fast 3-class training smoke test.

Auxiliary but still useful:

- `run/production/export_dense_three_class_labels.sbatch`: export dense shell/interior targets for inspection.