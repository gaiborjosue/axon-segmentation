"""
IoU Sanity Check: fast_rasterize vs synthspline
================================================

Generates ONE volume of B-spline curves (same seed, same curves),
rasterizes it with BOTH methods, thresholds at prob > 0.5, and
computes binary mask quality metrics.

This is the ground truth quality check — if IoU > 0.99 on the
binary masks, the training targets are effectively identical.

Usage:
    python iou_sanity_check.py [--n_curves 2500] [--shape 128]

Outputs
-------
  - Console: IoU, Dice, per-class breakdown, timing comparison
  - iou_results.txt  (same directory)
  - iou_slices.png   (mid-slice visual comparison, if matplotlib available)
"""

import argparse
import math
import sys
import time
import pathlib

import torch

# ── Path setup ────────────────────────────────────────────────────────────────
TEST_DIR  = pathlib.Path(__file__).resolve().parent
EXP_DIR   = TEST_DIR.parent.parent          # experiment/
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_curves",   type=int,   default=2500,
                   help="Synthetic curve count (default: 2500, close to real density)")
    p.add_argument("--shape",      type=int,   default=128)
    p.add_argument("--seg_length", type=float, default=1.5)
    p.add_argument("--batch_size", type=int,   default=2048)
    p.add_argument("--threshold",  type=float, default=0.5,
                   help="prob threshold for binary mask (default: 0.5)")
    p.add_argument("--device",     type=str,   default="cuda")
    p.add_argument("--save_slices", action="store_true",
                   help="Save mid-slice PNG comparison (requires matplotlib)")
    return p.parse_args()


# ── Curve builder ─────────────────────────────────────────────────────────────
def make_curves(n_curves, vol_size, device):
    from synthspline.curves import BSplineCurves, BSplineCurve
    torch.manual_seed(0)
    curve_list = []
    for _ in range(n_curves):
        n_wp   = torch.randint(4, 9, (1,)).item()
        wp     = torch.rand(n_wp, 3) * (vol_size - 1)
        radius = torch.rand(1).item() * 2.0 + 1.0   # 1–3 voxels
        c = BSplineCurve(waypoints=wp, radius=radius).to(device)
        curve_list.append(c)
    return BSplineCurves(curve_list)


# ── Metrics ───────────────────────────────────────────────────────────────────
def binary_metrics(mask_ref, mask_new, name="mask"):
    """Compute IoU, Dice, FP rate, FN rate between two binary masks."""
    ref = mask_ref.bool()
    new = mask_new.bool()

    tp = (ref &  new).sum().item()
    fp = (~ref &  new).sum().item()
    fn = (ref & ~new).sum().item()
    tn = (~ref & ~new).sum().item()

    iou  = tp / (tp + fp + fn + 1e-9)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-9)
    fp_rate = fp / (fp + tn + 1e-9)   # false axon voxels
    fn_rate = fn / (tp + fn + 1e-9)   # missed axon voxels

    total = ref.numel()
    pct_axon_ref = ref.sum().item() / total * 100

    print(f"\n  [{name}]")
    print(f"    Axon voxels (synthspline) : {ref.sum().item():>8,}  ({pct_axon_ref:.1f}%)")
    print(f"    Axon voxels (fast)        : {new.sum().item():>8,}  "
          f"({new.sum().item()/total*100:.1f}%)")
    print(f"    TP: {tp:>8,}  FP: {fp:>6,}  FN: {fn:>6,}")
    print(f"    IoU  : {iou:.5f}")
    print(f"    Dice : {dice:.5f}")
    print(f"    FP rate (false axon voxels) : {fp_rate*100:.3f}%")
    print(f"    FN rate (missed axon voxels): {fn_rate*100:.3f}%")

    return dict(iou=iou, dice=dice, fp_rate=fp_rate, fn_rate=fn_rate,
                n_axon_ref=tp+fn, n_axon_new=tp+fp)


# ── Slice visualizer ──────────────────────────────────────────────────────────
def save_slice_comparison(prob_ref, prob_new, mask_ref, mask_new, shape, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available, skipping slice plot)")
        return

    mid = shape // 2
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f"Mid-slice z={mid}: synthspline vs fast_rasterize", fontsize=13)

    axes[0, 0].imshow(prob_ref[:, :, mid].cpu(), vmin=0, vmax=1, cmap='hot')
    axes[0, 0].set_title("synthspline  prob")
    axes[0, 1].imshow(prob_new[:, :, mid].cpu(), vmin=0, vmax=1, cmap='hot')
    axes[0, 1].set_title("fast_rasterize  prob")
    diff = (prob_ref - prob_new).abs()
    axes[0, 2].imshow(diff[:, :, mid].cpu(), vmin=0, vmax=0.5, cmap='RdBu')
    axes[0, 2].set_title("|Δprob|  (max=0.5)")

    axes[1, 0].imshow(mask_ref[:, :, mid].cpu(), vmin=0, vmax=1, cmap='gray')
    axes[1, 0].set_title("synthspline  mask (>0.5)")
    axes[1, 1].imshow(mask_new[:, :, mid].cpu(), vmin=0, vmax=1, cmap='gray')
    axes[1, 1].set_title("fast_rasterize  mask (>0.5)")

    # Disagreement map: green=FP, red=FN
    disagree = torch.zeros(*mask_ref.shape[:2], 3)
    fp_map = (~mask_ref[:, :, mid].bool() &  mask_new[:, :, mid].bool()).cpu()
    fn_map = ( mask_ref[:, :, mid].bool() & ~mask_new[:, :, mid].bool()).cpu()
    agree  = ( mask_ref[:, :, mid].bool() &  mask_new[:, :, mid].bool()).cpu()
    disagree[agree,  1] = 0.6                       # agreed axon: green
    disagree[fp_map, 0] = 1.0                       # FP: red
    disagree[fn_map, 2] = 1.0                       # FN: blue
    axes[1, 2].imshow(disagree.numpy())
    axes[1, 2].set_title("Disagree: red=FP  blue=FN  green=agree")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Slice plot saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    shape  = (args.shape, args.shape, args.shape)
    thresh = args.threshold

    print("=" * 60)
    print("  IoU Sanity Check: fast_rasterize vs synthspline")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"  GPU:      {torch.cuda.get_device_name(device)}")
    print(f"  n_curves: {args.n_curves}")
    print(f"  shape:    {shape}")
    print(f"  threshold:{thresh}")
    print()

    # ── Build identical curves for both methods ──────────────────────────
    print("  Building curves (seed=0, same for both) …")
    curves = make_curves(args.n_curves, args.shape, device)

    # ── synthspline reference ────────────────────────────────────────────
    print("\n  Running synthspline rasterize …")
    import synthspline
    synthspline.backend.jitfields = True

    t0 = time.perf_counter()
    prob_ref, label_ref, dist_ref = curves.rasterize(shape, mode='cosine')
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_ref = time.perf_counter() - t0
    print(f"  synthspline: {t_ref:.2f}s")

    # ── fast_rasterize ───────────────────────────────────────────────────
    print("\n  Running fast_rasterize …")
    from fast_rasterizer import fast_rasterize

    t0 = time.perf_counter()
    prob_new, label_new, dist_new = fast_rasterize(
        curves, shape, mode='cosine',
        seg_length=args.seg_length,
        batch_size=args.batch_size,
    )
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_fast = time.perf_counter() - t0
    print(f"  fast_rasterize: {t_fast:.2f}s  (speedup: {t_ref/t_fast:.1f}×)")

    # ── Binary masks ─────────────────────────────────────────────────────
    mask_ref = prob_ref >= thresh
    mask_new = prob_new >= thresh

    # ── Metrics ──────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  BINARY MASK QUALITY  (prob > {})".format(thresh))
    print("─" * 60)
    metrics = binary_metrics(mask_ref, mask_new, name="prob > 0.5")

    print("\n" + "─" * 60)
    print("  CONTINUOUS PROB STATS")
    print("─" * 60)
    diff = (prob_ref - prob_new).abs()
    print(f"    max  |Δprob| : {diff.max().item():.4f}")
    print(f"    mean |Δprob| : {diff.mean().item():.6f}")
    print(f"    p95  |Δprob| : {diff.flatten().kthvalue(int(0.95 * diff.numel())).values.item():.4f}")

    valid = ~dist_ref.isinf() & ~dist_new.isinf()
    if valid.any():
        ddist = (dist_ref[valid] - dist_new[valid]).abs()
        print(f"    mean |Δdist| : {ddist.mean().item():.6f} voxels")

    # ── Save results ─────────────────────────────────────────────────────
    out_path = TEST_DIR / "iou_results.txt"
    with open(out_path, "w") as f:
        f.write("IoU Sanity Check Results\n")
        f.write("=" * 40 + "\n")
        if torch.cuda.is_available():
            f.write(f"GPU:               {torch.cuda.get_device_name(device)}\n")
        f.write(f"n_curves:          {args.n_curves}\n")
        f.write(f"shape:             {shape}\n")
        f.write(f"threshold:         {thresh}\n")
        f.write(f"synthspline_time:  {t_ref:.2f}s\n")
        f.write(f"fast_time:         {t_fast:.2f}s\n")
        f.write(f"speedup:           {t_ref/t_fast:.1f}x\n")
        f.write("\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write(f"\nprob_max_diff:  {diff.max().item():.4f}\n")
        f.write(f"prob_mean_diff: {diff.mean().item():.6f}\n")
    print(f"\n  Results saved: {out_path}")

    # ── Optional slice plot ───────────────────────────────────────────────
    if args.save_slices:
        save_slice_comparison(
            prob_ref, prob_new, mask_ref, mask_new,
            args.shape, TEST_DIR / "iou_slices.png"
        )

    # ── Final verdict ─────────────────────────────────────────────────────
    iou = metrics["iou"]
    print()
    print("=" * 60)
    if iou >= 0.99:
        print(f"  PASS  IoU = {iou:.4f} ≥ 0.99  — fast_rasterize is safe to use")
    elif iou >= 0.95:
        print(f"  WARN  IoU = {iou:.4f}  — acceptable but investigate differences")
    else:
        print(f"  FAIL  IoU = {iou:.4f}  — significant mask divergence, check rasterizer")
    print("=" * 60)


if __name__ == "__main__":
    main()
