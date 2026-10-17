"""Dataset loading and causal NWP-context construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


SPLITS = ("train", "val", "test")


@dataclass
class Scalers:
    history_mean: np.ndarray
    history_std: np.ndarray
    nwp_mean: np.ndarray
    nwp_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray

    def serializable(self) -> dict[str, list[float]]:
        return {key: value.tolist() for key, value in asdict(self).items()}


def _safe_std(values: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    std = values.std(axis=axes, dtype=np.float64).astype(np.float32)
    return np.where(std > 1e-6, std, 1.0).astype(np.float32)


def fit_scalers(data_dir: Path) -> Scalers:
    """Fit normalization statistics on ``train.npz`` only."""
    with np.load(data_dir / "train.npz", allow_pickle=False) as data:
        history = data["X_history"].astype(np.float32)
        nwp = data["X_nwp"].astype(np.float32)
        target = data["y"].astype(np.float32)
    if history.ndim != 4 or nwp.ndim != 4 or target.ndim != 3:
        raise ValueError("Expected X_history/X_nwp/y with dimensions 4/4/3.")
    return Scalers(
        history_mean=history.mean(axis=(0, 1), dtype=np.float64).astype(np.float32),
        history_std=_safe_std(history, (0, 1)),
        nwp_mean=nwp.mean(axis=(0, 1), dtype=np.float64).astype(np.float32),
        nwp_std=_safe_std(nwp, (0, 1)),
        target_mean=target.mean(axis=(0, 1), dtype=np.float64).astype(np.float32),
        target_std=_safe_std(target, (0, 1)),
    )


class WakeWINDDataset(Dataset):
    """Normalized model inputs and training-only future ERA5 targets.

    Each split file must contain ``X_history``, ``X_nwp``, ``y``,
    ``origin_time``, ``future_time``, ``nwp_forecast_time``, and ``turbine_id``.
    The first six historical channels are ERA5 weather variables and the final
    channel is normalized turbine power.
    """

    def __init__(self, path: Path, scalers: Scalers) -> None:
        with np.load(path, allow_pickle=False) as data:
            history = data["X_history"].astype(np.float32)
            nwp = data["X_nwp"].astype(np.float32)
            target = data["y"].astype(np.float32)
            origin = data["origin_time"].astype("datetime64[ns]")
            future = data["future_time"].astype("datetime64[ns]")
            issue_by_turbine = data["nwp_forecast_time"].astype("datetime64[ns]")
            self.turbine_id = data["turbine_id"].copy()
        if history.ndim != 4 or nwp.ndim != 4 or target.ndim != 3:
            raise ValueError("Expected X_history/X_nwp/y with dimensions 4/4/3.")
        if not np.all(issue_by_turbine == issue_by_turbine[..., :1]):
            raise ValueError("NWP issue times must be shared across turbines.")
        issue = issue_by_turbine[..., 0]
        if not np.all(issue <= origin[:, None]) or not np.all(future > origin[:, None]):
            raise ValueError("Samples must use causal NWP and future labels.")

        self.history = ((history - scalers.history_mean) / scalers.history_std).astype(np.float32)
        self.nwp = ((nwp - scalers.nwp_mean) / scalers.nwp_std).astype(np.float32)
        self.target = target
        self.origin_time, self.future_time = origin, future
        self.time_context = self._make_time_context(origin, future, issue)
        self.online_error = self._make_online_error(history, nwp, origin, future, scalers)
        self.future_weather, self.weather_mask = self._make_future_weather(
            history, origin, future, scalers
        )

    @staticmethod
    def _make_time_context(
        origin: np.ndarray, future: np.ndarray, issue: np.ndarray
    ) -> np.ndarray:
        run_age = (origin[:, None] - issue) / np.timedelta64(1, "h")
        lead = (future - issue) / np.timedelta64(1, "h")
        phase = np.mod(lead, 3.0) / 3.0
        interpolated = (np.mod(lead, 3.0) != 0.0).astype(np.float64)
        run_change = np.zeros_like(run_age, dtype=np.float64)
        run_change[:, 1:] = (issue[:, 1:] != issue[:, :-1]).astype(np.float64)
        return np.stack(
            [run_age / 7.0, lead / 27.0, np.square(lead / 27.0), phase, interpolated, run_change],
            axis=-1,
        ).astype(np.float32)

    @staticmethod
    def _make_online_error(
        history: np.ndarray,
        nwp: np.ndarray,
        origin: np.ndarray,
        future: np.ndarray,
        scalers: Scalers,
    ) -> np.ndarray:
        valid_time = future[:, 0]
        order = np.argsort(valid_time)
        insertion = np.searchsorted(valid_time[order], origin)
        matched = insertion < len(valid_time)
        matched[matched] = valid_time[order][insertion[matched]] == origin[matched]
        current_nwp = np.zeros((len(origin), nwp.shape[2], nwp.shape[3]), dtype=np.float32)
        current_nwp[matched] = nwp[order[insertion[matched]], 0]
        innovation = np.zeros_like(current_nwp)
        innovation[matched] = (history[matched, -1, :, :6] - current_nwp[matched]) / scalers.nwp_std
        return innovation.astype(np.float32)

    @staticmethod
    def _make_future_weather(
        history: np.ndarray,
        origin: np.ndarray,
        future: np.ndarray,
        scalers: Scalers,
    ) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(origin)
        flat_future = future.reshape(-1)
        insertion = np.searchsorted(origin[order], flat_future)
        matched = insertion < len(origin)
        matched[matched] = origin[order][insertion[matched]] == flat_future[matched]
        values = np.zeros((len(flat_future), history.shape[2], 6), dtype=np.float32)
        values[matched] = history[order[insertion[matched]], -1, :, :6]
        values = values.reshape(len(origin), future.shape[1], history.shape[2], 6)
        values = ((values - scalers.nwp_mean) / scalers.nwp_std).astype(np.float32)
        return values, matched.reshape(len(origin), future.shape[1]).astype(np.float32)

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return tuple(
            torch.from_numpy(values[index])
            for values in (
                self.history,
                self.nwp,
                self.time_context,
                self.online_error,
                self.future_weather,
                self.weather_mask,
                self.target,
            )
        )


def move_batch(batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(value.to(device, dtype=torch.float32, non_blocking=True) for value in batch)
