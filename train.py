"""
Axon Segmentation — 3D UNet Training Script

Two-GPU pipeline: GPU 0 runs cornucopia synthesis in a background thread,
GPU 1 runs the UNet forward/backward pass. A queue decouples them so
synthesis and training overlap.

Workers perform fast subset selection on CPU; label/prob tensors are sent to
GPU 0 for synthesis via ControlledContrastAxonImage.

Usage
-----
    python train.py \
        --label_dir /path/to/dense_labels \
        --output_dir /path/to/output \
        --epochs 200 \
        --batch_size 2
"""

import argparse
import logging
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

import monai
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
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
)
from monai.data import decollate_batch

sys.path.insert(0, str(Path(__file__).parent))
from datagen import create_dataloader
from datagen.axon_image_controlled_contrast import ControlledContrastAxonImage


# ---------------------------------------------------------------------------
# Label grouping
# ---------------------------------------------------------------------------

def collapse_labels(label: torch.Tensor, n_groups: int = 8) -> torch.Tensor:
    """Randomly remap unique axon IDs → 1..n_groups, keeping background=0."""
    out = torch.zeros_like(label)
    unique = label.unique()
    unique = unique[unique > 0]
    if unique.numel() == 0:
        return out
    groups = torch.randint(1, n_groups + 1, (unique.numel(),),
                           device=label.device, dtype=label.dtype)
    max_id = int(unique.max().item()) + 1
    lut = torch.zeros(max_id, device=label.device, dtype=label.dtype)
    lut[unique] = groups
    out = lut[label.clamp(0, max_id - 1)]
    return out


# ---------------------------------------------------------------------------
# Two-GPU prefetcher: synthesis on GPU 0, training on GPU 1
# ---------------------------------------------------------------------------

class SynthPrefetcher:
    """Background-thread prefetcher: runs cornucopia synthesis on synth_device
    and pushes ready (image, seg) tensors on train_device into a queue.

    The main training loop simply iterates over this like a dataloader.
    Because synthesis and training run on different GPUs, they overlap.
    """

    def __init__(self, dataloader, synth, synth_device, train_device,
                 n_label_groups=8, queue_depth=4, timeout=30):
        self.dataloader = dataloader
        self.synth = synth
        self.synth_device = synth_device
        self.train_device = train_device
        self.n_label_groups = n_label_groups
        self.queue_depth = queue_depth
        self.timeout = timeout

    def _produce(self, q, stop_event):
        """Worker thread: synthesize batches and enqueue them."""
        for batch in self.dataloader:
            if stop_event.is_set():
                break
            label = batch['label'].to(self.synth_device, non_blocking=True)
            prob  = batch['prob'].to(self.synth_device, non_blocking=True)
            label_g = collapse_labels(label, n_groups=self.n_label_groups)
            try:
                with torch.no_grad():
                    image, seg = self.synth(label_g, prob)
                # Move to training GPU
                image = image.to(self.train_device, non_blocking=True)
                seg   = seg.to(self.train_device, non_blocking=True)
                q.put((image, seg))
            except Exception as e:
                # Synthesis failure (timeout, bad input, etc.) — skip batch
                logging.getLogger(__name__).warning(f'Synth failed, skipping batch: {e}')
                continue
        q.put(None)  # sentinel: epoch done

    def __iter__(self):
        q = queue.Queue(maxsize=self.queue_depth)
        stop_event = threading.Event()
        t = threading.Thread(target=self._produce, args=(q, stop_event), daemon=True)
        t.start()
        while True:
            item = q.get()
            if item is None:
                break
            yield item
        t.join(timeout=5)

    def __len__(self):
        return len(self.dataloader)


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
    p.add_argument('--max_volumes',      type=int,   default=None,
                   help='Cap number of label volumes loaded (None = all). Applied before train/val split.')
    p.add_argument('--seed',             type=int,   default=42)
    p.add_argument('--roi_size',         type=int,   default=128,
                   help='Sliding window roi size (isotropic). Use 128 for 128³ volumes.')
    p.add_argument('--sw_batch_size',    type=int,   default=4,
                   help='Sliding window batch size during validation')
    p.add_argument('--n_label_groups',  type=int,   default=8,
                   help='Collapse unique axon IDs to N groups before synthesis '
                        '(speeds up cornucopia morphological ops ~N_axons/N times)')
    # Synthesis params
    p.add_argument('--no_images',        action='store_true',
                   help='Skip image synthesis (use raw label/prob tensors). For debugging only.')
    p.add_argument('--background',       type=float, default=0.5)
    p.add_argument('--fibers_lower_lo',  type=float, default=0.3)
    p.add_argument('--fibers_lower_hi',  type=float, default=0.5)
    p.add_argument('--bg_upper_lo',      type=float, default=0.2)
    p.add_argument('--bg_upper_hi',      type=float, default=0.4)
    p.add_argument('--resume',           action='store_true',
                   help='Resume from latest checkpoint in output_dir/checkpoints/')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- Setup ---
    # Manual seeding (set_determinism also sets cudnn.deterministic=True
    # and benchmark=False, but we want benchmark=True for speed).
    torch.manual_seed(args.seed)
    import numpy as _np; _np.random.seed(args.seed)
    import random as _pyrandom; _pyrandom.seed(args.seed)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    log = logging.getLogger(__name__)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = output_dir / 'tensorboard'
    ckpt_dir = output_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Train device: {device}')
    # Fixed input shape → cuDNN benchmarks fastest conv algorithm once then reuses it
    torch.backends.cudnn.benchmark = True
    monai.config.print_config()

    # --- DataLoaders ---
    # Workers run cornucopia CPU synthesis in parallel (generate_images=True).
    loader_kwargs = dict(
        generate_images=(not args.no_images),
        num_samples_per_volume=args.samples_per_vol,
        val_fraction=args.val_fraction,
        max_volumes=args.max_volumes,
        n_label_groups=args.n_label_groups,
        fibers_lower_range=(args.fibers_lower_lo, args.fibers_lower_hi),
        background_upper_range=(args.bg_upper_lo, args.bg_upper_hi),
        background=args.background,
    )
    train_loader = create_dataloader(
        label_dir=args.label_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split='train',
        **loader_kwargs,
    )
    val_loader = create_dataloader(
        label_dir=args.label_dir,
        batch_size=1,
        num_workers=max(2, args.num_workers // 4),
        split='val',
        **loader_kwargs,
    )

    log.info(f'Workers handle CPU synthesis (generate_images={not args.no_images})')
    log.info(f'Train batches/epoch: {len(train_loader)}')
    log.info(f'Val batches:         {len(val_loader)}')

    # --- Post-batch augmentation ---
    # Geometric: applied to both image and seg.
    # Intensity: applied to image only — closes the domain gap between
    #            synthetic contrast and real microscopy.
    geo_aug = Compose([
        RandFlipd(keys=['image', 'seg'], prob=0.5, spatial_axis=0),
        RandFlipd(keys=['image', 'seg'], prob=0.5, spatial_axis=1),
        RandFlipd(keys=['image', 'seg'], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=['image', 'seg'], prob=0.5, spatial_axes=(0, 2)),
        # Intensity augmentation (image only)
        RandScaleIntensityd(keys=['image'], factors=0.1, prob=1.0),
        RandShiftIntensityd(keys=['image'], offsets=0.1, prob=1.0),
        RandGaussianNoised(keys=['image'], prob=0.15, mean=0.0, std=0.05),
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
        dropout=0.1,
    ).to(device)
    log.info(f'Model params: {sum(p.numel() for p in model.parameters()):,}')

    # --- Loss, optimizer, scheduler ---
    # DiceCELoss: Dice stabilises spatial overlap, CE stabilises class balance early in training.
    loss_fn   = DiceCELoss(sigmoid=True, lambda_dice=0.5, lambda_ce=0.5)
    # AdamW: decoupled weight decay regularises all weights equally regardless
    # of gradient magnitude (Loshchilov & Hutter 2019).
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    # 5-epoch linear warmup → cosine decay: lets batch-norm stats and Adam moment
    # estimates warm up before full learning rate kicks in.
    warmup_epochs = 5
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, total_iters=warmup_epochs)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs, eta_min=args.lr / 100)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_epochs])

    # --- Post-processing for validation ---
    post_pred  = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    post_label = Compose([AsDiscrete(threshold=0.5)])

    dice_metric = DiceMetric(include_background=False, reduction='mean', get_not_nans=False)

    # --- Logging ---
    writer = SummaryWriter(log_dir=str(tb_dir))

    best_dice   = -1.0
    best_epoch  = -1
    start_epoch = 1
    roi_size    = (args.roi_size,) * 3
    use_amp     = device.type == 'cuda'
    scaler      = torch.amp.GradScaler('cuda', enabled=use_amp)

    # --- Resume from checkpoint ---
    if args.resume:
        ckpts = sorted(ckpt_dir.glob('epoch_*.pt'))
        resume_ckpt = ckpts[-1] if ckpts else (ckpt_dir / 'best_model.pt' if (ckpt_dir / 'best_model.pt').exists() else None)
        if resume_ckpt and resume_ckpt.exists():
            log.info(f'Resuming from {resume_ckpt}')
            ckpt = torch.load(str(resume_ckpt), map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if 'scaler_state_dict' in ckpt:
                scaler.load_state_dict(ckpt['scaler_state_dict'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_dice   = ckpt.get('val_dice', -1.0)
            best_epoch  = ckpt.get('best_epoch', -1)
            log.info(f'Resumed at epoch {start_epoch}, best_dice={best_dice:.4f} @ epoch {best_epoch}')
        else:
            log.info('--resume set but no checkpoint found — starting from scratch')

    # --- Preemption handler: save checkpoint on SIGTERM so job can be resumed ---
    def _save_preemption_ckpt(signum, frame):
        log.info('SIGTERM received — saving preemption checkpoint...')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'val_dice': best_dice,
            'best_epoch': best_epoch,
        }, str(ckpt_dir / f'epoch_{epoch:04d}.pt'))
        log.info(f'Preemption checkpoint saved at epoch {epoch}. Resubmit with --resume.')
        sys.exit(0)
    signal.signal(signal.SIGTERM, _save_preemption_ckpt)

    epoch = 0  # guard: defined before loop in case SIGTERM fires during setup
    log.info(f'Starting training: {args.epochs} epochs, lr={args.lr}, roi={roi_size}, AMP={use_amp}')

    for epoch in range(start_epoch, args.epochs + 1):
        # ------------------------------------------------------------------ #
        # Train
        # ------------------------------------------------------------------ #
        model.train()
        epoch_loss = 0.0
        step = 0
        t0 = time.time()
        t_batch_end = time.time()  # for measuring data-wait time
        _total_wait = 0.0
        _total_train = 0.0

        for batch in train_loader:
            t_wait = time.time() - t_batch_end  # time waiting for DataLoader
            t_step = time.time()

            image = batch['image'].to(device)
            seg   = batch['seg'].to(device)

            # Geometric augmentation per-sample then re-stack
            samples = [geo_aug(s) for s in decollate_batch({'image': image, 'seg': seg})]
            image = torch.stack([s['image'] for s in samples])
            seg   = torch.stack([s['seg']   for s in samples])

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                pred = model(image)
                loss = loss_fn(pred, seg)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            step += 1
            t_train = time.time() - t_step
            _total_wait += t_wait
            _total_train += t_train
            t_batch_end = time.time()

            # Log every batch during epoch 1, then every 50th batch
            if epoch == start_epoch or step % 50 == 0 or step <= 5:
                log.info(f'  batch {step:3d}/{len(train_loader)} | '
                         f'wait={t_wait:.2f}s train={t_train:.3f}s loss={loss.item():.4f}')

        scheduler.step()

        epoch_loss /= step
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        log.info(f'Epoch {epoch:04d}/{args.epochs} | loss={epoch_loss:.4f} | '
                 f'lr={lr_now:.2e} | {elapsed:.0f}s | '
                 f'data_wait={_total_wait:.0f}s train={_total_train:.1f}s '
                 f'overhead={elapsed - _total_wait - _total_train:.1f}s')
        writer.add_scalar('train/loss',   epoch_loss, epoch)
        writer.add_scalar('train/lr',     lr_now,     epoch)

        # ------------------------------------------------------------------ #
        # Validate
        # ------------------------------------------------------------------ #
        if epoch % args.val_interval == 0 or epoch == args.epochs:
            model.eval()
            log_images_this_epoch = True   # capture first val batch for TensorBoard
            with torch.no_grad():
                for val_batch in val_loader:
                    val_image = val_batch['image'].to(device)
                    val_seg   = val_batch['seg'].to(device)

                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        val_pred = sliding_window_inference(
                            val_image, roi_size, args.sw_batch_size, model,
                            overlap=0.5,
                        )
                    val_pred_post  = [post_pred(p)  for p in decollate_batch(val_pred)]
                    val_label_post = [post_label(l) for l in decollate_batch(val_seg)]
                    dice_metric(y_pred=val_pred_post, y=val_label_post)

                    # Log center-slice images from first batch only
                    if log_images_this_epoch:
                        log_images_this_epoch = False
                        z = val_image.shape[-1] // 2  # center axial slice
                        # Normalise image slice to [0,1] for display
                        img_slice  = val_image[0, 0, :, :, z]
                        img_slice  = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
                        seg_slice  = val_seg[0, 0, :, :, z]
                        pred_slice = torch.sigmoid(val_pred[0, 0, :, :, z])
                        # Stack side-by-side: image | ground truth | prediction
                        grid = torch.stack([img_slice, seg_slice, pred_slice], dim=0).unsqueeze(1)  # (3,1,H,W)
                        writer.add_images('val/image_gt_pred', grid, epoch, dataformats='NCHW')

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
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'val_dice': best_dice,
                    'best_epoch': best_epoch,
                    'args': vars(args),
                }, str(ckpt_path))
                log.info(f'  Saved best checkpoint → {ckpt_path}')

        # Save periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_dice': best_dice,            'best_epoch': best_epoch,            }, str(ckpt_dir / f'epoch_{epoch:04d}.pt'))

    # --- Final summary ---
    log.info('='*60)
    log.info(f'Training complete.')
    log.info(f'Best Val Dice: {best_dice:.4f} at epoch {best_epoch}')
    log.info(f'Checkpoints:   {ckpt_dir}')
    log.info(f'TensorBoard:   tensorboard --logdir {tb_dir}')
    writer.close()


if __name__ == '__main__':
    main()
