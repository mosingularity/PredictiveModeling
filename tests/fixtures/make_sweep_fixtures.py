"""Generate the hyperparameter-sweep fixture for the sweep-panel tests.

Writes ``tidy_sweep.parquet`` — a schema-valid tidy slice carrying, for each
model, ~5 OFAT variants keyed by ``param_set_id`` (the raw ``model_parameters``
string, per ``results_analysis/figures/SWEEP_MATRIX.md``). One knob is moved off
the repo baseline per variant; the forecast error grows with the knob's distance
from a per-model "best" value, so the variant RMSEs form a clean U-shape for the
fan colours and the parallel-coordinates metric axis. Deterministic (fixed seed).

Run from the project root:
    python -m tests.fixtures.make_sweep_fixtures
"""
import os

import numpy as np
import pandas as pd

from results_analysis.tidy import validate_tidy

HERE = os.path.dirname(os.path.abspath(__file__))

TARIFFS = {"LPU": ["LPU-E1", "LPU-E2"], "SPU": ["SPU-E1"]}
CONSUMPTION_TYPES = ["OffPeakConsumption", "PeakConsumption"]

# Per model: the OFAT variants as (param_set_id, swept-knob value), plus the
# "best" knob value the error is minimised at (gives a U-shaped RMSE sweep).
SWEEPS = {
    "RandomForest": {
        "best": 150.0,
        "scale": 60.0,  # knob-distance → error normaliser
        "variants": [
            ("(50,10,2,1,5,True)", 50.0),
            ("(100,10,2,1,5,True)", 100.0),
            ("(150,10,2,1,5,True)", 150.0),
            ("(200,10,2,1,5,True)", 200.0),
            ("(300,10,2,1,5,True)", 300.0),
        ],
    },
    "XGBoost": {
        "best": 0.1,
        "scale": 0.15,
        "variants": [
            ("(100,5,0.01,0.8,0.8)", 0.01),
            ("(100,5,0.05,0.8,0.8)", 0.05),
            ("(100,5,0.1,0.8,0.8)", 0.1),
            ("(100,5,0.3,0.8,0.8)", 0.3),
            ("(100,5,0.5,0.8,0.8)", 0.5),
        ],
    },
    "ARIMA": {
        "best": 3.0,
        "scale": 1.5,
        "variants": [
            ("(1,3,3)", 1.0),
            ("(2,3,3)", 2.0),
            ("(3,3,3)", 3.0),
            ("(4,3,3)", 4.0),
            ("(5,3,3)", 5.0),
        ],
    },
    "SARIMA": {
        "best": 1.0,
        "scale": 1.0,
        "variants": [
            ("(1,1,1)(0,1,1,4)", 0.0),
            ("(1,1,1)(1,1,1,4)", 1.0),
            ("(1,1,1)(2,1,1,4)", 2.0),
            ("(1,1,1)(3,1,1,4)", 3.0),
        ],
    },
}


def build_tidy_sweep() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    ds = pd.date_range("2024-01-01", periods=12, freq="MS")
    rows = []
    for tariff, entities in TARIFFS.items():
        for e_i, entity in enumerate(entities):
            for ct in CONSUMPTION_TYPES:
                base = 100 + 40 * e_i + (20 if ct == "PeakConsumption" else 0)
                season = 15 * np.sin(np.arange(len(ds)) / 12 * 2 * np.pi)
                actual = base + season + rng.normal(0, 4, len(ds))
                for model, spec in SWEEPS.items():
                    for param_id, knob in spec["variants"]:
                        dist = abs(knob - spec["best"]) / spec["scale"]
                        err_scale = 2.0 + 6.0 * dist  # grows with knob distance
                        yhat = actual + rng.normal(0, err_scale, len(ds))
                        for k, d in enumerate(ds):
                            rows.append({
                                "EntityID": entity,
                                "TariffType": tariff,
                                "EntityType": "POD",
                                "consumption_type": ct,
                                "param_set_id": param_id,
                                "model": model,
                                "ds": d,
                                "y": float(actual[k]),
                                "y_hat": float(yhat[k]),
                                "y_hat_lower": np.nan,
                                "y_hat_upper": np.nan,
                                "scenario": None,
                                "validation_reason": None,
                            })
    return validate_tidy(pd.DataFrame(rows))


def main():
    df = build_tidy_sweep()
    out = os.path.join(HERE, "tidy_sweep.parquet")
    df.to_parquet(out, index=False)
    print(f"wrote {out}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
