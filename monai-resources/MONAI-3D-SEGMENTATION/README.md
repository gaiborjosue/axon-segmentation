3D segmentation
ignite examples

Training and evaluation examples of 3D segmentation based on UNet3D and synthetic dataset. The examples are PyTorch Ignite programs and have both dictionary-based and array-based transformations.


torch examples

Training, evaluation and inference examples of 3D segmentation based on UNet3D and synthetic dataset. The examples are standard PyTorch programs and have both dictionary-based and array-based versions.


brats_segmentation_3d

This tutorial shows how to construct a training workflow of multi-labels segmentation task based on MSD Brain Tumor dataset, and how to convert the pytorch model to an onnx model for inference and comparison.


spleen_segmentation_3d_aim

This notebook shows how MONAI may be used in conjunction with the aimhubio/aim.


spleen_segmentation_3d_lightning

This notebook shows how MONAI may be used in conjunction with the PyTorch Lightning framework.


spleen_segmentation_3d

This notebook is an end-to-end training and evaluation example of 3D segmentation based on MSD Spleen dataset. The example shows the flexibility of MONAI modules in a PyTorch-based program:

    Transforms for dictionary-based training data structure.
    Load NIfTI images with metadata.
    Scale medical image intensity with expected range.
    Crop out a batch of balanced image patch samples based on positive / negative label ratio.
    Cache IO and transforms to accelerate training and validation.
    3D UNet, Dice loss function, Mean Dice metric for 3D segmentation task.
    Sliding window inference.
    Deterministic training for reproducibility.


unet_segmentation_3d_ignite

This notebook is an end-to-end training & evaluation example of 3D segmentation based on synthetic dataset. The example is a PyTorch Ignite program and shows several key features of MONAI, especially with medical domain specific transforms and event handlers for profiling (logging, TensorBoard, MLFlow, etc.).


COVID 19-20 challenge baseline

This folder provides a simple baseline method for training, validation, and inference for COVID-19 LUNG CT LESION SEGMENTATION CHALLENGE - 2020 (a MICCAI Endorsed Event).


unetr_btcv_segmentation_3d

This notebook demonstrates how to construct a training workflow of UNETR on multi-organ segmentation task using the BTCV challenge dataset.


unetr_btcv_segmentation_3d_lightning

This tutorial demonstrates how MONAI can be used in conjunction with PyTorch Lightning framework to construct a training workflow of UNETR on multi-organ segmentation task using the BTCV challenge dataset.


vista3d

This tutorial showcases the process of fine-tuning VISTA3D on MSD Spleen dataset using MONAI. For an in-depth exploration, please visit the VISTA repository.