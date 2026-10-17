# WakeWIND

Official PyTorch implementation of **WakeWIND**, a physical wake-aware model for multi-turbine wind-power forecasting with adaptive numerical weather prediction (NWP) correction.

## Model

WakeWIND contains only the final model components:

- a shared multiscale temporal encoder for historical weather and power;
- adaptive correction of future NWP wind components using causal online error;
- a future-NWP encoder augmented with wind-speed and air-density features;
- linear gated fusion of historical, future-weather, time-context, and turbine-ID states;
- a physical, wind-direction-dependent wake graph and gated message injection;
- a turbine-wise power prediction head.

The training objective contains four terms: turbine-level pinball loss, ramp loss, farm-average pinball loss, and masked Huber supervision for corrected NWP wind components.

## Installation

```bash
conda create -n wakewind python=3.10
conda activate wakewind
pip install -r requirements.txt
```

## Data format

The repository does not distribute wind-farm or meteorological data. Prepare one directory with `train.npz`, `val.npz`, and `test.npz`. Each file must contain the following arrays:

| Key | Shape | Description |
| --- | --- | --- |
| `X_history` | `[S, T, N, 7]` | Historical ERA5 variables (first six channels) and normalized power (last channel) |
| `X_nwp` | `[S, H, N, 6]` | Future NWP variables |
| `y` | `[S, H, N]` | Future normalized power targets |
| `origin_time` | `[S]` | Forecast-origin timestamps |
| `future_time` | `[S, H]` | Target timestamps |
| `nwp_forecast_time` | `[S, H, N]` | NWP issue timestamps |
| `turbine_id` | `[N]` | Turbine IDs in tensor order |

The layout CSV must provide turbine identifiers, latitude, longitude, and rotor diameter. Both the Kelmarsh and Hill of Towie metadata schemas are supported.

## Training

Run the reported configuration on Kelmarsh as follows:

```bash
python train.py \
  --data-dir /path/to/processed_multi_turbine \
  --layout /path/to/Kelmarsh_WT_static.csv \
  --output-dir outputs/kelmarsh/seed42 \
  --seed 42 \
  --evaluate-test
```

The defaults reproduce the final hyperparameter configuration:

```text
hidden size = 64       dropout = 0.1
q = 0.525              lambda_r = 0.075
lambda_f = 0.20        lambda_w = 1.0
learning rate = 4e-4   weight decay = 1e-3
batch size = 64        EMA decay = 0.99
```

Run seeds `42`, `43`, and `44` independently and report their mean and sample standard deviation. The checkpoint with the lowest validation MAE is selected; the test split is evaluated only after that checkpoint has been fixed.

## Repository layout

```text
.
├── src/wakewind/
│   ├── data.py       # Sample loading and causal context construction
│   ├── losses.py     # Four-term training objective
│   ├── metrics.py    # MAE, RMSE, R², turbine and horizon metrics
│   ├── model.py      # Encoders, correction, fusion, wake graph, prediction
│   ├── physics.py    # Layout parsing and physical wake weights
│   └── trainer.py    # Training, validation, checkpoint selection
├── train.py
├── requirements.txt
└── README.md
```
