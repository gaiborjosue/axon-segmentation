import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss


class RandPerlinNoised:
    """Add low-frequency, Perlin-like additive noise to image tensors."""

    def __init__(
        self,
        keys,
        *,
        prob: float = 0.0,
        grid_min: int = 4,
        grid_max: int = 16,
        amplitude_min: float = 0.02,
        amplitude_max: float = 0.12,
        octaves: int = 2,
        persistence: float = 0.5,
    ):
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"prob must be in [0, 1], got {prob}")
        if grid_min < 2 or grid_max < grid_min:
            raise ValueError(
                f"invalid grid range: grid_min={grid_min}, grid_max={grid_max}"
            )
        if amplitude_min < 0.0 or amplitude_max < amplitude_min:
            raise ValueError(
                "invalid amplitude range: "
                f"amplitude_min={amplitude_min}, amplitude_max={amplitude_max}"
            )
        if octaves < 1:
            raise ValueError(f"octaves must be >= 1, got {octaves}")
        if not 0.0 < persistence <= 1.0:
            raise ValueError(
                f"persistence must be in (0, 1], got {persistence}"
            )

        self.keys = tuple(keys)
        self.prob = prob
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.amplitude_min = amplitude_min
        self.amplitude_max = amplitude_max
        self.octaves = octaves
        self.persistence = persistence

    @staticmethod
    def _rand_uniform(low: float, high: float, *, device: torch.device) -> float:
        if low == high:
            return float(low)
        return float(low + (high - low) * torch.rand((), device=device).item())

    @staticmethod
    def _rand_int(low: int, high: int, *, device: torch.device) -> int:
        if low == high:
            return int(low)
        return int(torch.randint(low, high + 1, (), device=device).item())

    def _sample_field(
        self,
        spatial_shape: tuple[int, int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        base_resolution = self._rand_int(self.grid_min, self.grid_max, device=device)
        amplitude = self._rand_uniform(
            self.amplitude_min,
            self.amplitude_max,
            device=device,
        )

        field = torch.zeros(spatial_shape, device=device, dtype=dtype)
        total_weight = 0.0
        octave_weight = 1.0

        for octave in range(self.octaves):
            resolution = base_resolution * (2 ** octave)
            coarse_shape = tuple(max(2, min(dim, resolution)) for dim in spatial_shape)
            coarse = torch.rand((1, 1, *coarse_shape), device=device, dtype=dtype)
            coarse = coarse * 2.0 - 1.0
            upsampled = F.interpolate(
                coarse,
                size=spatial_shape,
                mode='trilinear',
                align_corners=False,
            )[0, 0]
            field = field + octave_weight * upsampled
            total_weight += octave_weight
            octave_weight *= self.persistence

        field = field / max(total_weight, 1e-6)
        field = field / field.abs().amax().clamp_min(1e-6)
        return amplitude * field

    def __call__(self, data):
        if self.prob == 0.0 or torch.rand(()).item() >= self.prob:
            return data

        result = dict(data)
        reference = result[self.keys[0]]
        if reference.ndim < 4:
            raise ValueError(
                f"Expected channel-first 3D tensor, got shape {tuple(reference.shape)}"
            )

        field = self._sample_field(
            tuple(int(v) for v in reference.shape[-3:]),
            device=reference.device,
            dtype=reference.dtype,
        )
        noise = field.unsqueeze(0)

        for key in self.keys:
            result[key] = result[key] + noise.to(
                device=result[key].device,
                dtype=result[key].dtype,
            )

        return result


class SoftSkeletonize(nn.Module):
    def __init__(self, num_iter: int = 10):
        super().__init__()
        self.num_iter = num_iter

    @staticmethod
    def soft_erode(img: torch.Tensor) -> torch.Tensor:
        if img.ndim == 5:
            p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
            p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
            p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
            return torch.min(torch.min(p1, p2), p3)
        raise ValueError(f"Expected 5D tensor for 3D skeletonization, got ndim={img.ndim}")

    @staticmethod
    def soft_dilate(img: torch.Tensor) -> torch.Tensor:
        if img.ndim == 5:
            return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        raise ValueError(f"Expected 5D tensor for 3D skeletonization, got ndim={img.ndim}")

    def soft_open(self, img: torch.Tensor) -> torch.Tensor:
        return self.soft_dilate(self.soft_erode(img))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        opened = self.soft_open(img)
        skeleton = F.relu(img - opened)

        for _ in range(self.num_iter):
            img = self.soft_erode(img)
            opened = self.soft_open(img)
            delta = F.relu(img - opened)
            skeleton = skeleton + F.relu(delta - skeleton * delta)

        return skeleton


class SoftClDiceLoss(nn.Module):
    def __init__(self, num_iter: int = 10, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        self.soft_skeletonize = SoftSkeletonize(num_iter=num_iter)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        prediction = prediction.float()

        skel_prediction = self.soft_skeletonize(prediction)
        skel_target = self.soft_skeletonize(target)

        topology_precision = (
            torch.sum(skel_prediction * target) + self.smooth
        ) / (torch.sum(skel_prediction) + self.smooth)
        topology_sensitivity = (
            torch.sum(skel_target * prediction) + self.smooth
        ) / (torch.sum(skel_target) + self.smooth)

        return 1.0 - (
            2.0 * topology_precision * topology_sensitivity
        ) / (topology_precision + topology_sensitivity + self.smooth)


class DiceCEClDiceLoss(nn.Module):
    def __init__(
        self,
        *,
        cldice_weight: float = 0.0,
        lambda_dice: float = 0.5,
        lambda_ce: float = 0.5,
        cldice_iters: int = 10,
    ):
        super().__init__()
        if not 0.0 <= cldice_weight <= 1.0:
            raise ValueError(
                f"cldice_weight must be in [0, 1], got {cldice_weight}"
            )

        self.cldice_weight = cldice_weight
        self.base_loss = DiceCELoss(
            sigmoid=True,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
        )
        self.cldice_loss = SoftClDiceLoss(num_iter=cldice_iters)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base_loss = self.base_loss(logits, target)
        if self.cldice_weight == 0.0:
            return base_loss

        cldice_loss = self.cldice_loss(torch.sigmoid(logits), target)
        return (1.0 - self.cldice_weight) * base_loss + self.cldice_weight * cldice_loss