"""Training and evaluation routines for PhyWIND."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PhyWINDDataset, Scalers, move_batch
from .losses import phywind_loss
from .metrics import breakdowns, regression_metrics
from .model import PhyWIND


@dataclass
class TrainConfig:
    hidden_size: int = 64
    dropout: float = 0.1
    correction_start: float = 0.85
    correction_floor: float = 0.30
    correction_tau: float = 8.0
    wake_expansion: float = 0.05
    wake_length_scale: float = 750.0
    quantile: float = 0.525
    ramp_weight: float = 0.075
    farm_weight: float = 0.20
    meteorological_weight: float = 1.0
    learning_rate: float = 4e-4
    weight_decay: float = 1e-3
    batch_size: int = 64
    grad_clip: float = 1.0
    ema_decay: float = 0.99
    epochs: int = 100
    patience: int = 10
    min_delta: float = 1e-5
    num_workers: int = 0
    seed: int = 42
    device: str = "cuda:0"
    amp: bool = True


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: PhyWIND, loader: DataLoader, device: torch.device) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    predictions, targets = [], []
    for batch in loader:
        history, nwp, context, online_error, _, _, target = move_batch(batch, device)
        predictions.append(model(history, nwp, context, online_error).cpu().numpy())
        targets.append(target.cpu().numpy())
    prediction, target = np.concatenate(predictions), np.concatenate(targets)
    return regression_metrics(prediction, target), prediction, target


def _loader(dataset: PhyWINDDataset, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def train(
    *,
    train_set: PhyWINDDataset,
    val_set: PhyWINDDataset,
    test_set: PhyWINDDataset | None,
    scalers: Scalers,
    positions: np.ndarray,
    rotor_diameter: float,
    output_dir: Path,
    config: TrainConfig,
) -> dict:
    """Train on ``train_set`` and select the checkpoint with lowest val MAE."""
    if not 0.0 < config.quantile < 1.0:
        raise ValueError("quantile must lie in (0, 1).")
    if min(config.ramp_weight, config.farm_weight, config.meteorological_weight) < 0.0:
        raise ValueError("Loss weights must be non-negative.")
    seed_everything(config.seed)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable: {config.device}")
    device = torch.device(config.device)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader = _loader(train_set, config.batch_size, True, config.num_workers)
    val_loader = _loader(val_set, config.batch_size, False, config.num_workers)
    test_loader = _loader(test_set, config.batch_size, False, config.num_workers) if test_set else None

    model = PhyWIND(
        history_mean=scalers.history_mean,
        history_std=scalers.history_std,
        nwp_mean=scalers.nwp_mean,
        nwp_std=scalers.nwp_std,
        target_mean=scalers.target_mean,
        positions=positions,
        rotor_diameter=rotor_diameter,
        turbines=train_set.history.shape[2],
        horizon=train_set.target.shape[1],
        hidden=config.hidden_size,
        dropout=config.dropout,
        correction_start=config.correction_start,
        correction_floor=config.correction_floor,
        correction_tau=config.correction_tau,
        wake_expansion=config.wake_expansion,
        wake_length_scale=config.wake_length_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5)
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    checkpoint = output_dir / "best_model.pt"
    history: list[dict] = []
    best_mae, stale = float("inf"), 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        totals = {name: 0.0 for name in ("loss", "mae", "power", "ramp", "farm", "meteorological")}
        samples = 0
        for batch in train_loader:
            history_x, nwp, context, online_error, future_era5, weather_mask, target = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                prediction, corrected_nwp = model(history_x, nwp, context, online_error, return_corrected_nwp=True)
                loss, parts = phywind_loss(
                    prediction, target, corrected_nwp, future_era5, weather_mask,
                    quantile=config.quantile, ramp_weight=config.ramp_weight,
                    farm_weight=config.farm_weight, meteorological_weight=config.meteorological_weight,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            size = history_x.shape[0]
            samples += size
            totals["loss"] += float(loss.detach()) * size
            for name, value in parts.items():
                totals[name] += value * size
            if epoch > 1:
                with torch.no_grad():
                    for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
                        ema_parameter.lerp_(parameter, 1.0 - config.ema_decay)
        if epoch == 1:
            ema_model.load_state_dict(model.state_dict())

        val_metrics, _, _ = evaluate(ema_model, val_loader, device)
        row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"]}
        row.update({f"train_{name}": value / samples for name, value in totals.items()})
        row.update({f"val_{name}": value for name, value in val_metrics.items()})
        history.append(row)
        print(f"epoch={epoch:03d} train_mae={row['train_mae']:.6f} val_mae={row['val_mae']:.6f}", flush=True)
        scheduler.step(val_metrics["mae"])
        if val_metrics["mae"] < best_mae - config.min_delta:
            best_mae, stale = val_metrics["mae"], 0
            torch.save({"model_state": ema_model.state_dict(), "config": asdict(config), "best_epoch": epoch}, checkpoint)
        else:
            stale += 1
            if stale >= config.patience:
                break

    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state"])
    val_metrics, val_prediction, val_target = evaluate(model, val_loader, device)
    result: dict = {
        "model": "PhyWIND",
        "best_epoch": saved["best_epoch"],
        "val": val_metrics,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "config": asdict(config),
        "training_history": history,
    }
    np.savez_compressed(output_dir / "val_predictions.npz", prediction=val_prediction, target=val_target, origin_time=val_set.origin_time, turbine_id=val_set.turbine_id)
    if test_loader is not None:
        test_metrics, test_prediction, test_target = evaluate(model, test_loader, device)
        by_turbine, by_horizon = breakdowns(test_prediction, test_target)
        result.update({"test": test_metrics, "test_by_turbine": by_turbine, "test_by_horizon": by_horizon})
        np.savez_compressed(output_dir / "test_predictions.npz", prediction=test_prediction, target=test_target, origin_time=test_set.origin_time, turbine_id=test_set.turbine_id)
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
