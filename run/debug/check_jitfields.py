"""
Debug script: verify jitfields / cupy / synthspline backend on a GPU node.
Reports whether CUDA rasterization is actually active or silently falling back.
"""
import sys, time, traceback

PY = sys.executable
print(f"Python: {PY}")
print(f"Python version: {sys.version}")
print()

# ── 1. CUDA / torch ─────────────────────────────────────────────────────────
print("=" * 60)
print("1. PyTorch / CUDA")
print("=" * 60)
import torch
print(f"  torch version  : {torch.__version__}")
print(f"  CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version   : {torch.version.cuda}")
    print(f"  Device name    : {torch.cuda.get_device_name(0)}")
    print(f"  Device count   : {torch.cuda.device_count()}")
print()

# ── 2. cupy ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("2. cupy")
print("=" * 60)
try:
    import cupy as cp
    print(f"  cupy version   : {cp.__version__}")
    a = cp.array([1.0, 2.0, 3.0])
    print(f"  cupy array test: {a}  ✓")
    cupy_ok = True
except Exception:
    print("  cupy FAILED:")
    traceback.print_exc()
    cupy_ok = False
print()

# ── 3. jitfields ─────────────────────────────────────────────────────────────
print("=" * 60)
print("3. jitfields")
print("=" * 60)
try:
    import jitfields
    print(f"  jitfields version: {jitfields.__version__}")
    jitfields_ok = True
except Exception:
    print("  jitfields FAILED:")
    traceback.print_exc()
    jitfields_ok = False
print()

# ── 4. synthspline backend ───────────────────────────────────────────────────
print("=" * 60)
print("4. synthspline backend")
print("=" * 60)
try:
    import synthspline
    print(f"  synthspline before toggle — backend.jitfields = {synthspline.backend.jitfields}")
    synthspline.backend.jitfields = True
    print(f"  synthspline after toggle  — backend.jitfields = {synthspline.backend.jitfields}")
    synthspline_ok = True
except Exception:
    print("  synthspline FAILED:")
    traceback.print_exc()
    synthspline_ok = False
print()

# ── 5. interpol backend ──────────────────────────────────────────────────────
print("=" * 60)
print("5. interpol backend (used by synthspline rasterization)")
print("=" * 60)
try:
    import interpol
    print(f"  interpol version : {getattr(interpol, '__version__', 'unknown')}")
    try:
        import interpol.backend as iback
        print(f"  interpol backend jitfields flag: {getattr(iback, 'jitfields', 'N/A')}")
    except Exception as e:
        print(f"  interpol.backend query failed: {e}")
except Exception:
    print("  interpol FAILED:")
    traceback.print_exc()
print()

# ── 6. Rasterization + skeleton timing (replicates actual label gen flow) ────
print("=" * 60)
print("6. Rasterization + skeleton timing")
print("=" * 60)
if not synthspline_ok:
    print("  Skipping — synthspline not available.")
else:
    try:
        import synthspline
        synthspline.backend.jitfields = True

        from synthspline.curves import BSplineCurve, BSplineCurves

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {device}")

        # ── Build synthetic BSplineCurves (straight lines along z, like axons) ──
        def make_curves(n, shape, device):
            """Create n straight-ish BSplineCurve objects running along z."""
            H, W, D = shape
            curves = []
            for _ in range(n):
                x = torch.rand(1).item() * (H - 1)
                y = torch.rand(1).item() * (W - 1)
                npts = 8
                # waypoints: shape (npts, 3) — (x, y, z) coordinates
                pts = torch.zeros(npts, 3)
                pts[:, 0] = x + torch.randn(npts) * 1.5
                pts[:, 1] = y + torch.randn(npts) * 1.5
                pts[:, 2] = torch.linspace(0, D - 1, npts)
                curves.append(BSplineCurve(pts.to(device)))
            return curves

        shape = (128, 128, 128)

        for n_curves in [10, 50, 200, 500]:
            curves_list = make_curves(n_curves, shape, device)
            bsc = BSplineCurves(curves_list)
            bsc.to(device)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()
            prob, labels, dist = bsc.rasterize(shape, mode='cosine')
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_raster = time.time() - t0

            # ── Skeleton loop (the Python for-loop from label gen) ──────────
            skeleton = torch.zeros(shape, dtype=torch.long, device=device)
            t1 = time.time()
            for i, curve in enumerate(bsc):
                ind = curve.evaluate_equidistant(0.1)
                ind = ind.round().long()
                ind = ind[(ind[:, 0] >= 0) & (ind[:, 0] < shape[0])]
                ind = ind[(ind[:, 1] >= 0) & (ind[:, 1] < shape[1])]
                ind = ind[(ind[:, 2] >= 0) & (ind[:, 2] < shape[2])]
                flatind = (ind[:, 2]
                           + ind[:, 1] * shape[2]
                           + ind[:, 0] * shape[2] * shape[1])
                skeleton.view([-1])[flatind] = i + 1
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_skel = time.time() - t1

            print(f"  n={n_curves:4d} curves | rasterize: {t_raster:7.2f}s "
                  f"({t_raster/n_curves*1000:.1f} ms/curve) | "
                  f"skeleton loop: {t_skel:7.2f}s "
                  f"({t_skel/n_curves*1000:.1f} ms/curve)")

        # ── Extrapolate to 3250 curves ───────────────────────────────────────
        print()
        print("  Extrapolated to 3250 curves (actual production volume):")
        # Use 500-curve timings as basis
        per_raster_ms = t_raster / n_curves * 1000
        per_skel_ms   = t_skel   / n_curves * 1000
        ext_r = per_raster_ms * 3250 / 1000
        ext_s = per_skel_ms   * 3250 / 1000
        print(f"    Estimated rasterize : {ext_r:.0f}s  ({ext_r/60:.1f} min)")
        print(f"    Estimated skeleton  : {ext_s:.0f}s  ({ext_s/60:.1f} min)")
        print(f"    Estimated total     : {ext_r+ext_s:.0f}s  ({(ext_r+ext_s)/60:.1f} min)")

    except Exception:
        print("  Rasterization test FAILED:")
        traceback.print_exc()
print()

# ── 7. Summary ───────────────────────────────────────────────────────────────
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  cupy        : {'✓ OK' if cupy_ok else '✗ FAILED'}")
print(f"  jitfields   : {'✓ OK' if jitfields_ok else '✗ FAILED'}")
print(f"  synthspline : {'✓ OK' if synthspline_ok else '✗ FAILED'}")
print()
if cupy_ok and jitfields_ok and synthspline_ok:
    print("  → All dependencies OK. CUDA rasterization should be active.")
elif not cupy_ok:
    print("  → cupy broken: jitfields CUDA backend will silently fall back to slow PyTorch/CPU.")
elif not jitfields_ok:
    print("  → jitfields broken: synthspline will use slow PyTorch-based rasterization.")
