"""Generate tidy-contract fixtures for the results-analysis figure tests.

Writes ``tidy_sample.parquet`` — a small, schema-valid tidy slice covering two
tariffs, multiple entities, four models, two consumption types, with actuals
(``y``) overlapping the forecast (``y_hat``) so residuals and legend metrics are
non-trivial, and confidence bounds present for ARIMA/SARIMA only (tree models
stay point-only). Deterministic (fixed seed) so the fixture is reproducible.

Run from the project root:
    python -m tests.fixtures.make_tidy_fixtures
"""
import os

import numpy as np
import pandas as pd

from results_analysis.tidy import validate_tidy

HERE = os.path.dirname(os.path.abspath(__file__))

TARIFFS = {"LPU": ["LPU-E1", "LPU-E2"], "SPU": ["SPU-E1"]}
MODELS = ["ARIMA", "SARIMA", "RandomForest", "XGBoost"]
CI_MODELS = {"ARIMA", "SARIMA"}  # only these carry confidence bounds
CONSUMPTION_TYPES = ["OffPeakConsumption", "PeakConsumption"]


def build_tidy_sample() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    ds = pd.date_range("2024-01-01", periods=12, freq="MS")
    rows = []
    for tariff, entities in TARIFFS.items():
        for e_i, entity in enumerate(entities):
            for ct in CONSUMPTION_TYPES:
                base = 100 + 40 * e_i + (20 if ct == "PeakConsumption" else 0)
                season = 15 * np.sin(np.arange(len(ds)) / 12 * 2 * np.pi)
                actual = base + season + rng.normal(0, 4, len(ds))
                scenario = "gap" if entity == "LPU-E1" and ct == "PeakConsumption" else None
                for model in MODELS:
                    bias = {"ARIMA": 1.0, "SARIMA": 0.5, "RandomForest": -2.0, "XGBoost": 1.5}[model]
                    yhat = actual + bias + rng.normal(0, 3, len(ds))
                    has_ci = model in CI_MODELS
                    for k, d in enumerate(ds):
                        rows.append({
                            "EntityID": entity,
                            "TariffType": tariff,
                            "EntityType": "POD",
                            "consumption_type": ct,
                            "param_set_id": "default",
                            "model": model,
                            "ds": d,
                            "y": float(actual[k]),
                            "y_hat": float(yhat[k]),
                            "y_hat_lower": float(yhat[k] - 8) if has_ci else np.nan,
                            "y_hat_upper": float(yhat[k] + 8) if has_ci else np.nan,
                            "scenario": scenario,
                            "validation_reason": None,
                        })
    return validate_tidy(pd.DataFrame(rows))


def main():
    df = build_tidy_sample()
    out = os.path.join(HERE, "tidy_sample.parquet")
    df.to_parquet(out, index=False)
    print(f"wrote {out}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
