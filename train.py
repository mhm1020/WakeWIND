"""Train PhyWIND on preprocessed multi-turbine samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from phywind.data import PhyWINDDataset, SPLITS, fit_scalers
from phywind.physics import load_layout
from phywind.trainer import TrainConfig, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing train/val/test.npz.")
    parser.add_argument("--layout", type=Path, required=True, help="Turbine layout CSV.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--correction-start", type=float, default=0.85)
    parser.add_argument("--correction-floor", type=float, default=0.30)
    parser.add_argument("--correction-tau", type=float, default=8.0)
    parser.add_argument("--wake-expansion", type=float, default=0.05)
    parser.add_argument("--wake-length-scale", type=float, default=750.0)
    parser.add_argument("--quantile", type=float, default=0.525)
    parser.add_argument("--ramp-weight", type=float, default=0.075)
    parser.add_argument("--farm-weight", type=float, default=0.20)
    parser.add_argument("--meteorological-weight", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing = [split for split in SPLITS if not (args.data_dir / f"{split}.npz").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing split files: {', '.join(missing)}")
    scalers = fit_scalers(args.data_dir)
    datasets = {split: PhyWINDDataset(args.data_dir / f"{split}.npz", scalers) for split in SPLITS}
    turbine_ids, positions, rotor_diameter = load_layout(args.layout)
    if not np.array_equal(turbine_ids, datasets["train"].turbine_id.astype(int)):
        raise ValueError("Turbine order in the layout and samples does not match.")
    torch.set_num_threads(max(torch.get_num_threads(), 1))
    config = TrainConfig(
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        correction_start=args.correction_start,
        correction_floor=args.correction_floor,
        correction_tau=args.correction_tau,
        wake_expansion=args.wake_expansion,
        wake_length_scale=args.wake_length_scale,
        quantile=args.quantile,
        ramp_weight=args.ramp_weight,
        farm_weight=args.farm_weight,
        meteorological_weight=args.meteorological_weight,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        grad_clip=args.grad_clip,
        ema_decay=args.ema_decay,
        epochs=args.epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        amp=args.amp,
    )
    result = train(
        train_set=datasets["train"],
        val_set=datasets["val"],
        test_set=datasets["test"] if args.evaluate_test else None,
        scalers=scalers,
        positions=positions,
        rotor_diameter=rotor_diameter,
        output_dir=args.output_dir,
        config=config,
    )
    print(json.dumps({"best_epoch": result["best_epoch"], "val": result["val"], "test": result.get("test")}, indent=2))


if __name__ == "__main__":
    main()
