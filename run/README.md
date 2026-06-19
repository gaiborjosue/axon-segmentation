# Run Launchers

`run/` is organized by how likely a launcher is to still matter:

- `production/`: current end-to-end jobs you are likely to submit again.
- `smoke/`: short validation jobs for integration checks.
- `debug/`: focused profiling and environment checks.
- `legacy/`: older experiments, diagnostics, and superseded helpers kept for reference.

If you only need the main workflow, start here:

- `run/production/train_production.sbatch`: binary production training.
- `run/production/train_three_class_production.sbatch`: 3-class production training.
- `run/production/infer_hipct.sbatch`: HiP-CT inference with env overrides for binary or 3-class checkpoints.
- `run/production/infer_lsm.sbatch`: LSM/WebKnossos inference.
- `run/production/infer_microct.sbatch`: microCT inference on raw+JSON patches downloaded from public OME-Zarr assets.
- `run/production/infer_lsm_eval.sbatch`: LSM inference plus sweep and corrected evaluation.
- `run/smoke/train_three_class_smoke.sbatch`: fast 3-class training smoke test.

Auxiliary but still useful:

- `run/production/export_dense_three_class_labels.sbatch`: export dense shell/interior targets for inspection.
- `run/production/save_hipct_crops.sbatch`: export review crops from a HiP-CT inference directory, including 3-class class/shell/interior volumes.
- `run/production/stage_microct_dandi_derivative.sbatch`: export top-level microCT inference volumes to OME-Zarr on a compute node and stage a short-name derivative branch under the `001769` DANDI root.