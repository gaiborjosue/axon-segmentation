"""
Axon Segmentation — 3D UNet Training Script

Uses synthetic label volumes from datagen.AxonSubsetDataset as training data.
Images and density-varying subsets are synthesised on-the-fly by DataLoader workers.

Usage
-----
    python train.py \
        --label_dir /path/to/dense_labels \
        --output_dir /path/to/output \
        --epochs 200 \
        --batch_size 2

Minimal draft test (3 volumes, 10 epochs):
    python train.py \
        --label_dir /scratch/experiment/draft/dense_labels \
        --output_dir /scratch/experiment/draft/training_out \
        --epochs 10 --batch_size 2 --num_workers 4 --val_fraction 0.5
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.utils import set_determinism
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    RandFlipd,
    RandRotate90d,
)
from monai.data import decollate_batch

sys.path.insert(0, str(Path(__file__).parent))
from datagen import create_dataloader


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Train 3D UNet on synthetic axon data')
    p.add_argument('--label_dir',   required=True,  help='Directory with *_label.nii.gz volumes')
    p.add_argument('--output_dir',  required=True,  help='Where to save checkpoints + TensorBoard logs')
    p.add_argument('--epochs',           type=int,   default=200)
    p.add_argument('--batch_size',       type=int,   default=2)
    p.add_argument('--lr',               type=float, default=1e-4)
    p.add_argument('--num_workers',      type=int,   default=10)
    p.add_argument('--val_fraction',     type=float, default=0.2,
                   help='Fraction of volumes reserved for validation (sorted order)')
    p.add_argument('--val_interval',     type=int,   default=5,
                   help='Run validation every N epochs')
    p.add_argument('--samples_per_vol',  type=int,   default=100,
                   help='Random subsets drawn per label volume per epoch')
    p.add_argument('--seed',             type=int,   default=42)
    p.add_argument('--roi_size',         type=int,   default=128,
                   help='Sliding window roi size (isotropic). Use 128 for 128³ volumes.')
    p.add_argument('--sw_batch_size',    type=int,   default=4,
                   help='Sliding window batch size during validation')
    # Synthesis params
    p.add_argument('--no_images',        action='store_true',
                   help='Skip image synthesis (use raw label/prob tensors). For debugging only.')
    p.add_argument('--background',       type=float, default=0.5)
    p.add_argument('--fibers_lower_lo',  type=float, default=0.3)
    p.add_argument('--fibers_lower_hi',  type=float, default=0.5)
    p.add_argument('--bg_upper_lo',      type=float, default=0.2)
    p.add_argument('--bg_upper_hi',      type=float, default=0.4)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- Setup ---
    set_determinism(seed=args.seed)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    log = logging.getLogger(__name__)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = output_dir / 'tensorboard'
    ckpt_dir = output_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Device: {device}')
    # Fixed input shape → cuDNN benchmarks fastest conv algorithm once then reuses it
    torch.backends.cudnn.benchmark = True
    monai.config.print_config()

    # --- DataLoaders ---
    synth_kwargs = dict(
        generate_images=not args.no_images,
        num_samples_per_volume=args.samples_per_vol,
        val_fraction=args.val_fraction,
        fibers_lower_range=(args.fibers_lower_lo, args.fibers_lower_hi),
        background_upper_range=(args.bg_upper_lo, args.bg_upper_hi),
        background=args.background,
    )
    train_loader = create_dataloader(
        label_dir=args.label_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split='train',
        **synth_kwargs,
    )
    val_loader = create_dataloader(
        label_dir=args.label_dir,
        batch_size=1,
        num_workers=max(2, args.num_workers // 4),
        split='val',
        **synth_kwargs,
    )

    log.info(f'Train batches/epoch: {len(train_loader)}')
    log.info(f'Val batches:         {len(val_loader)}')

    # --- Additional geometric augmentation applied post-batch ---
    # Our DataLoader already does heavy synthesis augmentation (cornucopia).
    # These are cheap free augmentations applied to the batch tensors directly.
    geo_aug = Compose([
        RandFlipd(keys=['image', 'seg'], prob=0.5, spatial_axis=0),
        RandFlipd(keys=['image', 'seg'], prob=0.5, spatial_axis=1),
        RandFlipd(keys=['image', 'seg'], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=['image', 'seg'], prob=0.5, spatial_axes=(0, 2)),
    ])

    # --- Model ---
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)
    log.info(f'Model params: {sum(p.numel() for p in model.parameters()):,}')

    # --- Loss, optimizer ---
    # DiceCELoss: Dice stabilises spatial overlap, CE stabilises class balance early in training.
    loss_fn   = DiceCELoss(sigmoid=True, lambda_dice=0.5, lambda_ce=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr / 100
    )

    # --- Post-processing for validation ---
    post_pred  = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    post_label = Compose([AsDiscrete(threshold=0.5)])

    dice_metric = DiceMetric(include_background=False, reduction='mean', get_not_nans=False)

    # --- Logging ---
    writer = SummaryWriter(log_dir=str(tb_dir))

    best_dice   = -1.0
    best_epoch  = -1
    roi_size    = (args.roi_size,) * 3
    use_amp     = device.type == 'cuda'
    scaler      = torch.amp.GradScaler('cuda', enabled=use_amp)

    log.info(f'Starting training: {args.epochs} epochs, lr={args.lr}, roi={roi_size}, AMP={use_amp}')

    for epoch in range(1, args.epochs + 1):
        # ------------------------------------------------------------------ #
        # Train
        # ------------------------------------------------------------------ #
        model.train()
        epoch_loss = 0.0
        step = 0
        t0 = time.time()

        for batch in train_loader:
            # Apply geometric augmentation per-sample then re-stack
            samples = [geo_aug(s) for s in decollate_batch(batch)]
            image = torch.stack([s['image'] for s in samples]).to(device)
            seg   = torch.stack([s['seg']   for s in samples]).to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                pred = model(image)
                loss = loss_fn(pred, seg)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            step += 1

        scheduler.step()

        epoch_loss /= step
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        log.info(f'Epoch {epoch:04d}/{args.epochs} | loss={epoch_loss:.4f} | '
                 f'lr={lr_now:.2e} | {elapsed:.0f}s')
        writer.add_scalar('train/loss',   epoch_loss, epoch)
        writer.add_scalar('train/lr',     lr_now,     epoch)

        # ------------------------------------------------------------------ #
        # Validate
        # ------------------------------------------------------------------ #
        if epoch % args.val_interval == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                for val_batch in val_loader:
                    val_image = val_batch['image'].to(device)
                    val_seg   = val_batch['seg'].to(device)

                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        val_pred = sliding_window_inference(
                            val_image, roi_size, args.sw_batch_size, model
                        )
                    val_pred_post  = [post_pred(p)  for p in decollate_batch(val_pred)]
                    val_label_post = [post_label(l) for l in decollate_batch(val_seg)]
                    dice_metric(y_pred=val_pred_post, y=val_label_post)

            mean_dice = dice_metric.aggregate().item()
            dice_metric.reset()

            log.info(f'  Val Dice: {mean_dice:.4f} (best={best_dice:.4f} @ epoch {best_epoch})')
            writer.add_scalar('val/dice', mean_dice, epoch)

            if mean_dice > best_dice:
                best_dice  = mean_dice
                best_epoch = epoch
                ckpt_path  = ckpt_dir / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'val_dice': best_dice,
                    'args': vars(args),
                }, str(ckpt_path))
                log.info(f'  Saved best checkpoint → {ckpt_path}')

        # Save periodic checkpoint every 50 epochs
        if epoch % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_dice': best_dice,
            }, str(ckpt_dir / f'epoch_{epoch:04d}.pt'))

    # --- Final summary ---
    log.info('='*60)
    log.info(f'Training complete.')
    log.info(f'Best Val Dice: {best_dice:.4f} at epoch {best_epoch}')
    log.info(f'Checkpoints:   {ckpt_dir}')
    log.info(f'TensorBoard:   tensorboard --logdir {tb_dir}')
    writer.close()


if __name__ == '__main__':
    main()
