"""Physical wake geometry used by PhyWIND."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


EARTH_RADIUS_M = 6_371_000.0


def _find_column(frame: pd.DataFrame, *candidates: str) -> str:
    columns = {str(column).strip().casefold(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.casefold() in columns:
            return columns[candidate.casefold()]
    raise ValueError(f"Layout is missing one of {candidates}.")


def load_layout(path: str | Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Load turbine IDs, local east/north positions, and rotor diameter."""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    id_column = _find_column(
        frame, "Title", "Turbine Name", "turbine_id", "Turbine", "TurbID"
    )
    latitude_column = _find_column(frame, "Latitude", "latitude", "lat")
    longitude_column = _find_column(frame, "Longitude", "longitude", "lon", "lng")
    rotor_column = _find_column(
        frame, "Rotor Diameter (m)", "rotor_diameter", "rotor diameter"
    )

    raw_ids = frame[id_column]
    if pd.api.types.is_numeric_dtype(raw_ids):
        turbine_ids = raw_ids.astype(int).to_numpy()
    else:
        parsed = raw_ids.astype(str).str.extract(r"(\d+)(?!.*\d)")[0]
        if parsed.isna().any():
            raise ValueError("Turbine identifiers must end in a numeric index.")
        turbine_ids = parsed.astype(int).to_numpy()
    if len(np.unique(turbine_ids)) != len(turbine_ids):
        raise ValueError("Layout contains duplicate turbine identifiers.")

    order = np.argsort(turbine_ids)
    latitude = np.deg2rad(frame[latitude_column].to_numpy(dtype=float)[order])
    longitude = np.deg2rad(frame[longitude_column].to_numpy(dtype=float)[order])
    latitude_0, longitude_0 = latitude.mean(), longitude.mean()
    positions = np.column_stack(
        [
            (longitude - longitude_0) * np.cos(latitude_0) * EARTH_RADIUS_M,
            (latitude - latitude_0) * EARTH_RADIUS_M,
        ]
    ).astype(np.float32)

    rotor = frame[rotor_column].to_numpy(dtype=float)[order]
    if not np.isfinite(rotor).all() or np.any(rotor <= 0.0):
        raise ValueError("Rotor diameters must be finite and positive.")
    if not np.allclose(rotor, rotor[0]):
        raise ValueError("PhyWIND currently assumes a common rotor diameter.")
    return turbine_ids[order].astype(np.int64), positions, float(rotor[0])


def wake_adjacency(
    wind_uv: torch.Tensor,
    positions: torch.Tensor,
    rotor_diameter: float,
    wake_expansion: float,
    length_scale: float,
) -> torch.Tensor:
    """Return physical wake weights with shape ``[B, H, target, source]``."""
    direction = wind_uv / torch.linalg.vector_norm(
        wind_uv, dim=-1, keepdim=True
    ).clamp_min(1e-6)
    delta = positions[:, None, :] - positions[None, :, :]
    downstream = torch.einsum("bhd,tsd->bhts", direction, delta)
    crosswind = torch.abs(
        direction[..., 0, None, None] * delta[None, None, :, :, 1]
        - direction[..., 1, None, None] * delta[None, None, :, :, 0]
    )
    radius = rotor_diameter / 2.0 + wake_expansion * downstream.clamp_min(0.0)
    mask = (downstream > 0.0) & (crosswind <= radius)
    identity = torch.eye(len(positions), device=wind_uv.device, dtype=torch.bool)
    mask = mask & ~identity[None, None]
    weights = torch.exp(-downstream.clamp_min(0.0) / length_scale) * torch.exp(
        -0.5 * (crosswind / radius.clamp_min(1e-6)).square()
    )
    return torch.where(mask, weights, torch.zeros_like(weights))
