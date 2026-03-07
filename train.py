"""
Axon Segmentation — 3D UNet Training Script

Training samples are synthesized from dense label volumes on demand, cached in
RAM for a configurable number of epochs, and refreshed from the dense source
volumes between cache cycles. Validation is synthesized once at startup and
kept fixed for the entire run so checkpoint selection is stable.
"""

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
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


class CachedTensorDataset(Dataset):
    """Small dict-style dataset backed by in-memory tensors."""

    def __init__(self, images: torch.Tensor, segs: torch.Tensor):
        if images.shape[0] != segs.shape[0]:
            raise ValueError(
                f'Cached tensors disagree on batch dimension: '
                f'images={tuple(images.shape)} segs={tuple(segs.shape)}'
            )
        self.images = images.contiguous()
        self.segs = segs.contiguous()

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, idx: int) -> dict:
        return {
            'image': self.images[idx],
            'seg': self.segs[idx],
        }


def build_tensor_cache(loader, split: str, log: logging.Logger) -> CachedTensorDataset:
    """Materialize one full pass of a source loader into RAM."""
    t0 = time.time()
    image_batches = []
    seg_batches = []

    for batch_idx, batch in enumerate(loader, start=1):
        if 'image' not in batch or 'seg' not in batch:
            raise KeyError(
                f"Source {split} batch is missing 'image'/'seg' keys. "
                f'Available keys: {sorted(batch.keys())}'
            )
        image_batches.append(batch['image'].detach().cpu().clone().float())
        seg_batches.append(batch['seg'].detach().cpu().clone().float())
        if batch_idx == 1 or batch_idx % 50 == 0 or batch_idx == len(loader):
            log.info(f'  caching {split}: batch {batch_idx:3d}/{len(loader)}')

    if not image_batches:
        raise RuntimeError(f'Failed to build {split} cache: source loader yielded no batches')

    images = torch.cat(image_batches, dim=0).contiguous()
    segs = torch.cat(seg_batches, dim=0).contiguous()
    n_bytes = images.numel() * images.element_size() + segs.numel() * segs.element_size()
    elapsed = time.time() - t0
    log.info(
        f'Built {split} cache: {images.shape[0]} samples | '
        f'{n_bytes / (1024 ** 3):.2f} GiB | {elapsed:.1f}s'
    )
    return CachedTensorDataset(images, segs)


def create_cached_loader(
    dataset: CachedTensorDataset,
    batch_size: int,
    *,
    shuffle: bool,
    drop_last: bool,
    pin_memory: bool = True,
) -> DataLoader:
    """Serve cached tensors with lightweight loader settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


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
    p.add_argument('--cache_epochs',     type=int,   default=1,
                   help='Reuse each synthesized train cache for N epochs before refreshing')
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
    if args.cache_epochs < 1:
        raise ValueError(f'--cache_epochs must be >= 1, got {args.cache_epochs}')

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

    # --- Source DataLoaders ---
    # Workers run cornucopia CPU synthesis in parallel to build RAM caches.
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
    train_source_loader = create_dataloader(
        label_dir=args.label_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split='train',
        shuffle=True,
        drop_last=True,
        persistent_workers=True,
        **loader_kwargs,
    )
    val_source_loader = create_dataloader(
        label_dir=args.label_dir,
        batch_size=1,
        num_workers=max(2, args.num_workers // 4),
        split='val',
        shuffle=False,
        drop_last=False,
        persistent_workers=False,
        **loader_kwargs,
    )

    log.info(f'Workers handle CPU synthesis (generate_images={not args.no_images})')
    log.info(f'Source train batches/cache build: {len(train_source_loader)}')
    log.info(f'Source val batches/cache build:   {len(val_source_loader)}')

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

    log.info('Building fixed validation cache...')
    val_cache = build_tensor_cache(val_source_loader, split='val', log=log)
    val_loader = create_cached_loader(
        val_cache,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    log.info(f'Validation cache ready: {len(val_cache)} samples, {len(val_loader)} batches')

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

    train_cache = None
    train_loader = None
    train_cache_start_epoch = None
    total_train_epochs = max(0, args.epochs - start_epoch + 1)
    total_cache_cycles = (
        (total_train_epochs + args.cache_epochs - 1) // args.cache_epochs
        if total_train_epochs > 0 else 0
    )

    def _checkpoint_state(epoch_num: int) -> dict:
        return {
            'epoch': epoch_num,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'val_dice': best_dice,
            'best_epoch': best_epoch,
            'args': vars(args),
            'train_cache_start_epoch': train_cache_start_epoch,
        }

    # --- Preemption handler: save checkpoint on SIGTERM so job can be resumed ---
    def _save_preemption_ckpt(signum, frame):
        log.info('SIGTERM received — saving preemption checkpoint...')
        torch.save(_checkpoint_state(epoch), str(ckpt_dir / f'epoch_{epoch:04d}.pt'))
        log.info(f'Preemption checkpoint saved at epoch {epoch}. Resubmit with --resume.')
        sys.exit(0)
    signal.signal(signal.SIGTERM, _save_preemption_ckpt)

    epoch = 0  # guard: defined before loop in case SIGTERM fires during setup
    log.info(
        f'Starting training: {args.epochs} epochs, lr={args.lr}, roi={roi_size}, '
        f'AMP={use_amp}, cache_epochs={args.cache_epochs}'
    )

    for epoch in range(start_epoch, args.epochs + 1):
        # ------------------------------------------------------------------ #
        # Train
        # ------------------------------------------------------------------ #
        if train_loader is None or epoch >= train_cache_start_epoch + args.cache_epochs:
            train_cache_start_epoch = epoch
            cache_cycle_idx = ((epoch - start_epoch) // args.cache_epochs) + 1
            cache_cycle_len = min(args.cache_epochs, args.epochs - epoch + 1)
            log.info(
                f'Building train cache cycle {cache_cycle_idx}/{total_cache_cycles} '
                f'for epochs {epoch}-{epoch + cache_cycle_len - 1}...'
            )
            train_cache = build_tensor_cache(train_source_loader, split='train', log=log)
            train_loader = create_cached_loader(
                train_cache,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
            )
            log.info(f'Train cache ready: {len(train_cache)} samples, {len(train_loader)} batches')
        else:
            cache_cycle_len = min(args.cache_epochs, args.epochs - train_cache_start_epoch + 1)
            cache_reuse_idx = epoch - train_cache_start_epoch + 1
            log.info(
                f'Reusing train cache from epoch {train_cache_start_epoch} '
                f'({cache_reuse_idx}/{cache_cycle_len})'
            )

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
                torch.save(_checkpoint_state(epoch), str(ckpt_path))
                log.info(f'  Saved best checkpoint → {ckpt_path}')

        # Save periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save(_checkpoint_state(epoch), str(ckpt_dir / f'epoch_{epoch:04d}.pt'))

    # --- Final summary ---
    log.info('='*60)
    log.info(f'Training complete.')
    log.info(f'Best Val Dice: {best_dice:.4f} at epoch {best_epoch}')
    log.info(f'Checkpoints:   {ckpt_dir}')
    log.info(f'TensorBoard:   tensorboard --logdir {tb_dir}')
    writer.close()


if __name__ == '__main__':
    main()
