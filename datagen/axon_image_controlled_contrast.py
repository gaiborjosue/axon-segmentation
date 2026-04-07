"""
Axon Image Synthesis — Controlled Contrast

Extends SynthAxonImage with guaranteed contrast separation between
axons (foreground) and background, as recommended by the Balbasty group:

    Background rescaled to  [0,            Uniform(background_upper_range)]
    Axons      rescaled to  [Uniform(fibers_lower_range),  1.0            ]

A small overlap between the two ranges is intentional — it mimics real
data where some bright background structures approach the intensity of
dim axons. Both bounds are re-sampled independently each forward call,
giving contrast variation across training samples.

Note: uses a local _minmax_rescale() helper because cc.MinMaxTransform
is not available in cornucopia 0.3.0.  The dev version (0.4.0) has a
broken convnd for iso kernels, so we pin to 0.3.0.
"""

import math as pymath
import random as pyrandom
import logging
import time as _time

import torch
import cornucopia as cc

from synthspline.imagezoo import AutoBatchTransform

_log = logging.getLogger(__name__)


def _minmax_rescale(x: torch.Tensor, vmin: float, vmax: float) -> torch.Tensor:
    """Linearly rescale tensor range [x.min(), x.max()] → [vmin, vmax].

    Falls back to filling with vmin when the tensor is constant.
    """
    x_min = x.min()
    x_max = x.max()
    if x_max == x_min:
        return x.new_full(x.shape, vmin)
    return (x - x_min) / (x_max - x_min) * (vmax - vmin) + vmin


class ControlledContrastAxonImage(AutoBatchTransform):
    """Synthesize an LSM-like image from axon labels with controlled contrast.

    Axons are always brighter than the background (with a small configurable
    overlap to mimic realistic partial-volume effects).

    Parameters
    ----------
    background : float
        Probability of adding random background structures.
        0 = plain dark background.
    fibers_lower_range : (float, float)
        Uniform range for the axon intensity floor.
        Axon intensities → [Uniform(fibers_lower_range), 1.0].
        Default: (0.3, 0.5)
    background_upper_range : (float, float)
        Uniform range for the background intensity ceiling.
        Background intensities → [0, Uniform(background_upper_range)].
        Default: (0.2, 0.4)

    Example
    -------
    >>> synth = ControlledContrastAxonImage(
    ...     background=0.5,
    ...     fibers_lower_range=(0.3, 0.5),
    ...     background_upper_range=(0.2, 0.4),
    ... )
    >>> image, prob = synth(labels, prob_map)   # labels/prob: (B,1,D,H,W)
    """

    class XForm(cc.Transform):

        def __init__(
            self,
            background: float = 0.5,
            fibers_lower_range: tuple = (0.3, 0.5),
            background_upper_range: tuple = (0.2, 0.4),
            hipct_structures: float = 0.0,
        ):
            super().__init__()
            self.background            = background
            self.fibers_lower_range    = fibers_lower_range
            self.background_upper_range = background_upper_range
            self.hipct_structures      = hipct_structures

            # --- label perturbation ---
            self.flip = cc.RandomFlipTransform()
            self.erode_axon = cc.RandomSmoothMorphoLabelTransform(
                shape=128, max_radius=4, min_radius=-4,
            )
            self.shallow = cc.RandomSmoothShallowLabelTransform(
                shape=128, max_width=3,
            ) * 0.3
            self.noisylabel = cc.randomize(cc.SmoothBernoulliTransform)(
                shape=cc.RandInt(2, 128),
                prob=cc.Uniform(0, 0.2),
            )
            self.soma = cc.randomize(cc.SmoothBernoulliDiskTransform)(
                shape=cc.RandInt(2, 16),
                prob=cc.Uniform(0, 0.02),
                radius=10,
                returns='disks',
            )

            # --- background structure ---
            self.label_map   = cc.RandomSmoothLabelMap(16, 8)
            self.erode_label = cc.RandomErodeLabelTransform(radius=5, new_labels=True)

            # --- HiP-CT-like background structures (cell bodies + myelin) ---
            # Only instantiated when hipct_structures > 0 to avoid overhead.
            if hipct_structures > 0:
                # Scattered spherical blobs: radii 6–20 voxels @ 0.8 µm ≈ 5–16 µm
                # (mimics cell bodies / nuclei in HiP-CT)
                self.bg_cell_bodies = cc.randomize(cc.SmoothBernoulliDiskTransform)(
                    shape=cc.RandInt(2, 16),
                    prob=cc.Uniform(0, 0.003),
                    radius=(6, 20),
                    returns='disks',
                )
                # Thin shell around each blob → mimics myelin / nuclear envelope
                self.bg_myelin = cc.RandomSmoothShallowLabelTransform(
                    max_width=3, min_width=1, shape=8,
                )
                # Per-label GMM to assign intensities to cytoplasm vs myelin ring
                self.gmm_struct = cc.RandomGaussianMixtureTransform()

            # --- separate GMMs for foreground / background ---
            self.gmm_fg = cc.RandomGaussianMixtureTransform(
                background=None if self.background else 0,
            )
            self.gmm_bg = cc.RandomGaussianMixtureTransform()

            # --- imaging artifacts ---
            self.gamma    = cc.RandomGammaTransform((0, 5))
            self.addbias  = cc.RandomAddFieldTransform(vmin=0, vmax=0.25)
            self.mulbias  = cc.RandomMulFieldTransform(symmetric=1)
            self.smooth   = cc.RandomSmoothTransform(2)
            self.noise    = (
                cc.RandomChiNoiseTransform() | cc.RandomGammaNoiseTransform()
            )
            self.rescale  = cc.QuantileTransform()

        def forward(self, lab, prob=None):
            """
            Parameters
            ----------
            lab  : (1, *spatial) tensor[int]   — unique per-axon label map
            prob : (1, *spatial) tensor[float] — partial volume probabilities

            Returns
            -------
            image : (1, *spatial) tensor[float]
            prob  : (1, *spatial) tensor[float]  (unchanged)
            """
            if isinstance(lab, (list, tuple)):
                lab, prob = lab

            lab, prob = self.flip(lab, prob)

            # ---- perturb axon label map ----
            # Cap retry loops to avoid infinite hangs on thin/small axons
            # whose voxels are completely annihilated by erosion.  If all
            # attempts fail, skip the step and use the pre-op tensor.
            _t0 = _time.monotonic()
            v = lab.clone()
            n_fg_in = int(v.any())
            _erode_retries = 0
            _erode_fallback = False
            v0, v = v, self.erode_axon(v)
            for _i in range(10):
                if v.any():
                    break
                _erode_retries += 1
                v = self.erode_axon(v0)
            else:
                _erode_fallback = True
                v = v0  # erosion impossible — use un-eroded labels
            _t_erode = _time.monotonic() - _t0

            _shallow_retries = 0
            _shallow_fallback = False
            _t1 = _time.monotonic()
            v0, v = v, self.shallow(v)
            for _i in range(10):
                if v.any():
                    break
                _shallow_retries += 1
                v = self.shallow(v0)
            else:
                _shallow_fallback = True
                v = v0  # shallow impossible — use pre-shallow labels
            _t_shallow = _time.monotonic() - _t1
            del v0

            if _erode_retries > 0 or _shallow_retries > 0:
                _log.info(
                    f'[synth] erode: {_erode_retries} retries, fallback={_erode_fallback} '
                    f'({_t_erode:.2f}s) | shallow: {_shallow_retries} retries, '
                    f'fallback={_shallow_fallback} ({_t_shallow:.2f}s) | '
                    f'fg_voxels_in={n_fg_in}')
            v = self.noisylabel(v)

            # group axons into shared-intensity classes
            y            = torch.zeros_like(lab, dtype=torch.int)
            vessel_labels = list(sorted(v.unique().tolist()))[1:]
            pyrandom.shuffle(vessel_labels)
            nb_groups    = cc.RandInt(1, 5)()
            nb_per_group = int(pymath.ceil(len(vessel_labels) / nb_groups))
            for i in range(nb_groups):
                group = vessel_labels[i * nb_per_group:(i + 1) * nb_per_group]
                for label in group:
                    y.masked_fill_(v == label, i + 1)
                soma = self.soma(y)
                y.masked_fill(soma > 0, i + 1)
            del v

            # ---- foreground: GMM → rescale to [fibers_lower, 1] ----
            y            = self.gmm_fg(y)
            fibers_lower = float(cc.Uniform(*self.fibers_lower_range)())
            y            = _minmax_rescale(y, vmin=fibers_lower, vmax=1.0)
            y            = y * prob    # partial-volume soft blend

            # ---- background: GMM → rescale to [0, background_upper] ----
            if cc.Uniform(1)() < self.background:
                z                = self.label_map(y)
                z                = self.erode_label(z)
                z                = self.gmm_bg(z)
                background_upper = float(cc.Uniform(*self.background_upper_range)())
                z                = _minmax_rescale(z, vmin=0.0, vmax=background_upper)
                y                = y + (1 - prob) * z
                del z

            # ---- HiP-CT-like background structures (cell bodies + myelin shells) ----
            # Toggled via hipct_structures probability.  Structures are blended
            # exclusively into non-axon voxels via (1 - prob), so they never
            # corrupt the segmentation target.
            if self.hipct_structures > 0 and cc.Uniform(1)() < self.hipct_structures:
                blob_canvas = torch.zeros_like(lab)          # int zeros, same shape/device
                cell_mask   = self.bg_cell_bodies(blob_canvas)
                if cell_mask.any():
                    cell_labels = (cell_mask > 0).to(torch.int)
                    cell_labels = self.bg_myelin(cell_labels)  # cytoplasm=1, shell=new int
                    cell_labels = self.gmm_struct(cell_labels) # per-label Gaussian intensity
                    struct_upper = float(cc.Uniform(self.background_upper_range[0], fibers_lower - 0.05)())
                    cell_labels  = _minmax_rescale(cell_labels, vmin=0.0, vmax=struct_upper)
                    y = y + (1 - prob) * cell_labels
                    del cell_labels, cell_mask
                del blob_canvas

            # ---- global imaging artifacts ----
            y = self.addbias(y)
            y = self.mulbias(y)
            y = self.gamma(y)
            y = self.smooth(y)
            y = self.noise(y)
            y = self.rescale(y)

            return y, prob

    def __init__(
        self,
        background: float = 0.5,
        fibers_lower_range: tuple = (0.3, 0.5),
        background_upper_range: tuple = (0.2, 0.4),
        hipct_structures: float = 0.0,
    ):
        super().__init__(
            background=background,
            fibers_lower_range=fibers_lower_range,
            background_upper_range=background_upper_range,
            hipct_structures=hipct_structures,
        )
