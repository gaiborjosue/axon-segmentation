"""
Benchmark: fast_rasterize vs synthspline BSplineCurves.rasterize
================================================================

Usage (on a GPU node):
    python benchmark.py [--n_curves 200] [--shape 128] [--seg_length 1.5]

Outputs
-------
1. Timing comparison (synthspline vs fast_rasterize)
2. Numerical comparison: max |prob difference|, label agreement %, dist MAE
3. Summary table saved to  benchmark_results.txt  (same directory)

The default --n_curves 200 is representative enough for timing while
finishing in <5 min.  The full 3250-curve case takes ~2× the rasterize time.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="fast_rasterizer benchmark")
    p.add_argument("--n_curves",   type=int,   default=200,
                   help="Number of synthetic B-spline curves (default: 200)")
    p.add_argument("--shape",      type=int,   default=128,
                   help="Cubic volume side length (default: 128)")
    p.add_argument("--seg_length", type=float, default=1.5,
                   help="Target segment length for tessellation (default: 1.5)")
    p.add_argument("--batch_size", type=int,   default=512,
                   help="Segments per GPU batch (default: 512)")
    p.add_argument("--skip_ref",   action="store_true",
                   help="Skip synthspline reference (timing-only mode)")
    p.add_argument("--device",     type=str,   default="cuda",
                   help="Torch device (default: cuda)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic curve builder
# ─────────────────────────────────────────────────────────────────────────────
def make_synthetic_curves(n_curves: int, vol_size: int, device, dtype=torch.float32):
    """Create a BSplineCurves with random waypoints inside [0, vol_size-1]^3.

    BSplineCurve(waypoints, order=3, radius=1)
      waypoints : (N, D) tensor  — control points the curve interpolates
      radius    : float or (N,) tensor — tube radius in voxels
    """
    import synthspline
    from synthspline.curves import BSplineCurves, BSplineCurve
    synthspline.backend.jitfields = True

    print(f"  Building {n_curves} synthetic curves  (vol_size={vol_size}) …", flush=True)
    t0 = time.perf_counter()

    curve_list = []
    torch.manual_seed(42)
    for _ in range(n_curves):
        n_waypoints = torch.randint(4, 9, (1,)).item()
        # waypoints on CPU first (BSplineCurve does float conversion internally)
        waypoints = torch.rand(n_waypoints, 3) * (vol_size - 1)
        radius = torch.rand(1).item() * 2.0 + 1.0   # 1–3 voxels
        c = BSplineCurve(waypoints=waypoints, radius=radius)
        c = c.to(device)
        curve_list.append(c)
    curves = BSplineCurves(curve_list)

    print(f"  Curve build: {time.perf_counter()-t0:.2f}s")
    return curves


# ─────────────────────────────────────────────────────────────────────────────
# Timing helpers
# ─────────────────────────────────────────────────────────────────────────────
def time_fn(label, fn, *args, **kwargs):
    """Run fn(*args, **kwargs), return (result, elapsed_seconds)."""
    if args and args[0] is not None and hasattr(args[0], 'device'):
        dev = str(args[0].device)
    else:
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    if 'cuda' in dev:
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    result = fn(*args, **kwargs)

    if 'cuda' in dev:
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0
    print(f"  [{label:30s}]  {elapsed:8.2f} s", flush=True)
    return result, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Numerical comparison
# ─────────────────────────────────────────────────────────────────────────────
def compare(prob_ref, label_ref, dist_ref, prob_new, label_new, dist_new):
    prob_diff = (prob_ref - prob_new).abs()
    dist_diff = (dist_ref - dist_new).abs()
    # For dist: inf in ref should be inf in new
    valid_dist = ~dist_ref.isinf() & ~dist_new.isinf()
    dist_valid_diff = (dist_ref[valid_dist] - dist_new[valid_dist]).abs()

    label_agree = (label_ref == label_new).float().mean().item() * 100.0

    print("\n  Numerical comparison (fast_rasterize vs synthspline reference):")
    print(f"    prob  — max |Δ|: {prob_diff.max().item():.4f}  "
          f"mean |Δ|: {prob_diff.mean().item():.6f}")
    print(f"    dist  — max |Δ| (valid): "
          f"{dist_valid_diff.max().item() if valid_dist.any() else float('nan'):.4f}  "
          f"mean |Δ|: "
          f"{dist_valid_diff.mean().item() if valid_dist.any() else float('nan'):.6f}")
    print(f"    label — agreement: {label_agree:.2f}%")

    return {
        "prob_max_diff":  prob_diff.max().item(),
        "prob_mean_diff": prob_diff.mean().item(),
        "dist_max_diff":  dist_valid_diff.max().item() if valid_dist.any() else float('nan'),
        "dist_mean_diff": dist_valid_diff.mean().item() if valid_dist.any() else float('nan'),
        "label_agreement_pct": label_agree,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    shape  = (args.shape, args.shape, args.shape)

    print("=" * 60)
    print("  fast_rasterizer  BENCHMARK")
    print("=" * 60)
    print(f"  device:     {device}")
    print(f"  shape:      {shape}")
    print(f"  n_curves:   {args.n_curves}")
    print(f"  seg_length: {args.seg_length}")
    print(f"  batch_size: {args.batch_size}")
    print()

    # ── Setup ─────────────────────────────────────────────────────────────
    # Add experiment/ to path so we can import fast_rasterizer
    exp_dir = Path(__file__).resolve().parent.parent
    if str(exp_dir) not in sys.path:
        sys.path.insert(0, str(exp_dir))

    curves = make_synthetic_curves(args.n_curves, args.shape, device)

    # ── synthspline reference run ─────────────────────────────────────────
    import synthspline

    t_ref = None
    prob_ref = label_ref = dist_ref = None

    if not args.skip_ref:
        print("\n  Running synthspline reference (BSplineCurves.rasterize) …")
        synthspline.backend.jitfields = True

        def run_ref():
            return curves.rasterize(shape, mode='cosine')

        (prob_ref, label_ref, dist_ref), t_ref = time_fn(
            "synthspline reference", run_ref
        )
        print(f"    prob  range: [{prob_ref.min():.3f}, {prob_ref.max():.3f}]")
        print(f"    label range: [{label_ref.min()}, {label_ref.max()}]")
        print(f"    dist  range: [{dist_ref[~dist_ref.isinf()].min():.3f}, "
              f"{dist_ref[~dist_ref.isinf()].max():.3f}]")
    else:
        print("\n  [skip_ref] Skipping synthspline reference run.")

    # ── fast_rasterize run ─────────────────────────────────────────────────
    print("\n  Running fast_rasterize …")
    from fast_rasterizer import fast_rasterize

    def run_fast():
        return fast_rasterize(
            curves, shape, mode='cosine',
            seg_length=args.seg_length,
            batch_size=args.batch_size,
        )

    (prob_new, label_new, dist_new), t_fast = time_fn("fast_rasterize", run_fast)
    print(f"    prob  range: [{prob_new.min():.3f}, {prob_new.max():.3f}]")
    print(f"    label range: [{label_new.min()}, {label_new.max()}]")
    print(f"    dist  range: [{dist_new[~dist_new.isinf()].min():.3f}, "
          f"{dist_new[~dist_new.isinf()].max():.3f}]")

    # ── Speedup ───────────────────────────────────────────────────────────
    print()
    if t_ref is not None:
        speedup = t_ref / t_fast if t_fast > 0 else float('inf')
        print(f"  Speedup: {speedup:.1f}×  "
              f"({t_ref:.2f}s ref  vs  {t_fast:.2f}s fast)")

    # ── Numerical comparison ──────────────────────────────────────────────
    metrics = {}
    if prob_ref is not None:
        metrics = compare(prob_ref, label_ref, dist_ref,
                          prob_new, label_new, dist_new)

    # ── Save results ───────────────────────────────────────────────────────
    out_path = Path(__file__).parent / "benchmark_results.txt"
    with open(out_path, "w") as f:
        f.write("fast_rasterizer benchmark results\n")
        f.write("=" * 50 + "\n")
        f.write(f"device:          {device}\n")
        if torch.cuda.is_available():
            f.write(f"GPU:             {torch.cuda.get_device_name(device)}\n")
        f.write(f"shape:           {shape}\n")
        f.write(f"n_curves:        {args.n_curves}\n")
        f.write(f"seg_length:      {args.seg_length}\n")
        f.write(f"batch_size:      {args.batch_size}\n")
        f.write("\n")
        if t_ref is not None:
            f.write(f"synthspline_time_s: {t_ref:.3f}\n")
            f.write(f"fast_time_s:        {t_fast:.3f}\n")
            f.write(f"speedup:            {t_ref/t_fast:.2f}x\n")
        else:
            f.write(f"fast_time_s:        {t_fast:.3f}\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    print(f"\n  Results written to: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
