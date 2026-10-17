"""Evaluation metrics for normalized power outputs."""

from __future__ import annotations

import math

import numpy as np


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    mse = float(np.mean(np.square(error)))
    denominator = float(np.square(target - target.mean()).sum())
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": math.sqrt(mse),
        "r2": float(1.0 - np.square(error).sum() / denominator) if denominator > 0 else float("nan"),
    }


def breakdowns(prediction: np.ndarray, target: np.ndarray) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    error = prediction - target
    by_turbine = [
        {"turbine_id": index + 1, "mae": float(np.abs(error[:, :, index]).mean()), "rmse": float(np.sqrt(np.square(error[:, :, index]).mean()))}
        for index in range(prediction.shape[2])
    ]
    by_horizon = [
        {"lead_hour": index + 1, "mae": float(np.abs(error[:, index, :]).mean()), "rmse": float(np.sqrt(np.square(error[:, index, :]).mean()))}
        for index in range(prediction.shape[1])
    ]
    return by_turbine, by_horizon
