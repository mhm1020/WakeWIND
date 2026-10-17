"""Training objectives for WakeWIND."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pinball_loss(prediction: torch.Tensor, target: torch.Tensor, quantile: float) -> torch.Tensor:
    """Scaled pinball loss; at ``q=0.5`` it equals MAE."""
    error = target - prediction
    return 2.0 * torch.maximum(quantile * error, (quantile - 1.0) * error).mean()


def wakewind_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    corrected_nwp: torch.Tensor,
    future_era5: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    quantile: float = 0.525,
    ramp_weight: float = 0.075,
    farm_weight: float = 0.20,
    meteorological_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Four-term objective used for the reported WakeWIND configuration."""
    power = pinball_loss(prediction, target, quantile)
    ramp = F.l1_loss(prediction[:, 1:] - prediction[:, :-1], target[:, 1:] - target[:, :-1])
    farm = pinball_loss(prediction.mean(dim=-1), target.mean(dim=-1), quantile)
    meteorological_element = F.smooth_l1_loss(
        corrected_nwp[..., :2], future_era5[..., :2], reduction="none"
    )
    mask = valid_mask[..., None, None]
    denominator = mask.sum().clamp_min(1.0) * corrected_nwp.shape[2] * 2
    meteorological = (meteorological_element * mask).sum() / denominator
    total = power + ramp_weight * ramp + farm_weight * farm + meteorological_weight * meteorological
    parts = {
        "power": float(power.detach()),
        "ramp": float(ramp.detach()),
        "farm": float(farm.detach()),
        "meteorological": float(meteorological.detach()),
        "mae": float(F.l1_loss(prediction, target).detach()),
    }
    return total, parts
