LATEST VERSION OF THIS REPO HAS BEEN MOVED TO: https://github.com/lincbrain/axonsynth

# Axon Segmentation Experiments

This repo contains the end-to-end pipeline for synthetic 3D axon segmentation:

- synthetic data generation from dense instance-label volumes,
- MONAI 3D U-Net training for binary and 3-class targets,
- inference on HiP-CT and LSM volumes,
- real-data evaluation on manually labeled WebKnossos patches,
- SLURM launchers for training, inference, exports, and benchmarks.

## Main entrypoints

- `train.py`: train binary or 3-class models.
- `inference/`: inference and evaluation scripts.
- `datagen/`: synthetic label/image generation and GPU cache building.
- `run/`: SBATCH launchers organized into `production/`, `smoke/`, `debug/`, and `legacy/`.
- `tools/`: one-off exports, benchmarks, and debug helpers.
- `fast_rasterizer/`: accelerated rasterization helpers.
- `environment.yml`: reproducible conda environment.

If you only need the core workflow, start with `train.py`, `inference/`, `datagen/`, and `run/`.
