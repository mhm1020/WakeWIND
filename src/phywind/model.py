"""PhyWIND model definition."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .physics import wake_adjacency


class TemporalGatedBlock(nn.Module):
    """Depthwise temporal convolution followed by gated channel mixing."""

    def __init__(self, hidden: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            hidden, hidden, kernel_size=3, padding=dilation, dilation=dilation, groups=hidden
        )
        self.channel_mixer = nn.Conv1d(hidden, hidden * 2, kernel_size=1)
        self.output = nn.Conv1d(hidden, hidden, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        mixed = self.depthwise(values.transpose(1, 2))
        mixed = F.glu(self.channel_mixer(mixed), dim=1)
        mixed = self.dropout(self.output(mixed).transpose(1, 2))
        return self.norm(values + mixed)


class HistoryEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.blocks = nn.ModuleList(TemporalGatedBlock(hidden, d, dropout) for d in (1, 2, 4, 8))
        self.pool_score = nn.Linear(hidden, 1)
        self.output = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self.input_projection(values)
        for block in self.blocks:
            encoded = block(encoded)
        weights = torch.softmax(self.pool_score(encoded).squeeze(-1), dim=1)
        pooled = torch.einsum("bt,bth->bh", weights, encoded)
        return self.output(torch.cat([encoded[:, -1], pooled], dim=-1))


class FutureEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.blocks = nn.ModuleList(TemporalGatedBlock(hidden, d, dropout) for d in (1, 2, 4))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self.input_projection(values)
        for block in self.blocks:
            encoded = block(encoded)
        return encoded


class LinearGatedFusion(nn.Module):
    """``LN(W_s a + sigmoid(W_g a) * W_t a)`` without a nonlinear transform."""

    def __init__(self, input_dim: int, hidden: int) -> None:
        super().__init__()
        self.skip = nn.Linear(input_dim, hidden)
        self.transform = nn.Linear(input_dim, hidden)
        self.gate = nn.Linear(input_dim, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.norm(self.skip(values) + torch.sigmoid(self.gate(values)) * self.transform(values))


class PhysicalWakeGraph(nn.Module):
    """Physical wake aggregation with a learned state-value projection."""

    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.message_gate = nn.Linear(hidden * 2, hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden), nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, states: torch.Tensor, physical: torch.Tensor) -> torch.Tensor:
        weights = physical / physical.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        message = torch.einsum("bhts,bhsc->bhtc", weights, self.value(states))
        gate = torch.sigmoid(self.message_gate(torch.cat([states, message], dim=-1)))
        mixed = states + gate * message
        return self.norm(mixed + self.ffn(mixed))


class PhyWIND(nn.Module):
    """Physical wake-aware forecasting with adaptive NWP correction."""

    def __init__(
        self,
        *,
        history_mean: np.ndarray,
        history_std: np.ndarray,
        nwp_mean: np.ndarray,
        nwp_std: np.ndarray,
        target_mean: np.ndarray,
        positions: np.ndarray,
        rotor_diameter: float,
        turbines: int,
        horizon: int,
        hidden: int = 64,
        dropout: float = 0.1,
        correction_start: float = 0.85,
        correction_floor: float = 0.30,
        correction_tau: float = 8.0,
        wake_expansion: float = 0.05,
        wake_length_scale: float = 750.0,
    ) -> None:
        super().__init__()
        self.turbines, self.horizon = turbines, horizon
        self.rotor_diameter = rotor_diameter
        self.wake_expansion, self.wake_length_scale = wake_expansion, wake_length_scale
        self.register_buffer("positions", torch.as_tensor(positions, dtype=torch.float32))
        self.register_buffer("nwp_mean", torch.as_tensor(nwp_mean, dtype=torch.float32)[None, None])
        self.register_buffer("nwp_std", torch.as_tensor(nwp_std, dtype=torch.float32)[None, None])
        self.register_buffer("history_mean", torch.as_tensor(history_mean, dtype=torch.float32)[None, None])
        self.register_buffer("history_std", torch.as_tensor(history_std, dtype=torch.float32)[None, None])

        self.history_encoder = HistoryEncoder(7, hidden, dropout)
        self.future_encoder = FutureEncoder(13, hidden, dropout)
        self.time_embedding = nn.Sequential(nn.Linear(6, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.turbine_embedding = nn.Embedding(turbines, hidden)
        self.fusion = LinearGatedFusion(hidden * 4, hidden)
        self.wake_graph = PhysicalWakeGraph(hidden, dropout)
        self.prediction_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        nn.init.zeros_(self.prediction_head[-1].weight)
        mean = float(np.asarray(target_mean).mean())
        mean = min(max(mean, 1e-4), 1.0 - 1e-4)
        nn.init.constant_(self.prediction_head[-1].bias, math.log(mean / (1.0 - mean)))

        initial_gain = correction_floor + (correction_start - correction_floor) * torch.exp(
            -torch.arange(horizon, dtype=torch.float32) / max(correction_tau, 1e-3)
        )
        initial_gain = initial_gain.clamp(1e-4, 1.0 - 1e-4)
        self.correction_logits = nn.Parameter(torch.logit(initial_gain)[:, None].repeat(1, 6))
        self.adaptive_gain = nn.Sequential(nn.Linear(18, 24), nn.GELU(), nn.Linear(24, 2))
        nn.init.zeros_(self.adaptive_gain[-1].weight)
        nn.init.zeros_(self.adaptive_gain[-1].bias)

    def forward(
        self,
        history: torch.Tensor,
        nwp: torch.Tensor,
        time_context: torch.Tensor,
        online_error: torch.Tensor,
        return_corrected_nwp: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch, _, turbines, features = history.shape
        if turbines != self.turbines or features != 7:
            raise ValueError("Expected history with shape [B, T, N, 7].")
        if nwp.shape != (batch, self.horizon, turbines, 6):
            raise ValueError("Unexpected NWP shape.")
        if time_context.shape != (batch, self.horizon, 6):
            raise ValueError("Unexpected time-context shape.")
        if online_error.shape != (batch, turbines, 6):
            raise ValueError("Unexpected online-error shape.")

        history_local = history.permute(0, 2, 1, 3).reshape(batch * turbines, history.shape[1], 7)
        history_state = self.history_encoder(history_local).reshape(batch, turbines, -1)
        history_state = history_state[:, None].expand(-1, self.horizon, -1, -1)

        gain_input = torch.cat(
            [online_error[:, None].expand(-1, self.horizon, -1, -1), nwp, time_context[:, :, None].expand(-1, -1, turbines, -1)], dim=-1
        )
        gain = torch.sigmoid(self.correction_logits[:, :2][None, :, None] + self.adaptive_gain(gain_input))
        corrected_nwp = torch.cat([nwp[..., :2] + gain * online_error[:, None, ..., :2], nwp[..., 2:]], dim=-1)

        physical_nwp = corrected_nwp * self.nwp_std + self.nwp_mean
        u, v = physical_nwp[..., 0], physical_nwp[..., 1]
        temperature, pressure = physical_nwp[..., 2].clamp_min(180.0), physical_nwp[..., 4].clamp_min(1000.0)
        speed = torch.hypot(u, v).clamp_min(1e-3)
        density = pressure / (287.05 * temperature)
        energy = torch.stack(
            [speed / 15.0, speed.square() / 225.0, speed.pow(3) / 3375.0,
             density / 1.225, density / 1.225 * (speed / 15.0).pow(3), u / speed, v / speed], dim=-1
        )
        future = torch.cat([corrected_nwp, energy], dim=-1)
        future_local = future.permute(0, 2, 1, 3).reshape(batch * turbines, self.horizon, 13)
        future_state = self.future_encoder(future_local).reshape(batch, turbines, self.horizon, -1).permute(0, 2, 1, 3)

        time_state = self.time_embedding(time_context)[:, :, None].expand(-1, -1, turbines, -1)
        turbine_state = self.turbine_embedding(torch.arange(turbines, device=history.device))[None, None].expand(batch, self.horizon, -1, -1)
        states = self.fusion(torch.cat([history_state, future_state, time_state, turbine_state], dim=-1))
        farm_wind = physical_nwp[..., :2].mean(dim=2)
        states = self.wake_graph(states, wake_adjacency(farm_wind, self.positions, self.rotor_diameter, self.wake_expansion, self.wake_length_scale))
        prediction = torch.sigmoid(self.prediction_head(states).squeeze(-1))
        return (prediction, corrected_nwp) if return_corrected_nwp else prediction
