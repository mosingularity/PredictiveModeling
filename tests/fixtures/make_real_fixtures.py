"""Generate the scenario-triage fixture (``tidy_real.parquet``).

A schema-valid tidy slice whose ``scenario`` / ``validation_reason`` marks are
produced by the *actual* validator (``validation.series.validate_series``),
not hand-typed — so the triage figure's prevalence provably matches the
validator's calls. One entity per data condition the validator can raise:

* ``POD_SHORT_6M``   — 6 months → too short for every method.
* ``POD_SHORT_14M``  — 14 months → the method-relative case: rejected by
  ARIMA/SARIMA (need 18), valid for the trees (need 12), which therefore carry a
  forecast while ARIMA/SARIMA do not.
* ``POD_GAP_2M``     — a 2-month hole → gap ≤ 3, proceeds with gap handling.
* ``POD_GAP_5M``     — a 5-month hole → gap > 3, rejected.
* ``COMBO_ZERO``     — 24 all-zero months → rejected.
* ``POD_NEG``        — a sub-zero dip → warning only, proceeds.

Actuals (``y``) are stamped at observed months only; the missing months simply
have no rows, so the triage figure's ``connectgaps=False`` reindex renders them as
holes. A model's ``y_hat`` is present only where the validator accepted the
series for that method (``ok``), NaN where it was rejected. Deterministic.

Run from the project root:
    python -m tests.fixtures.make_real_fixtures
"""
import os

import numpy as np
import pandas as pd

from evaluation.performance import PredictionUnit
from validation.series import validate_series
from results_analysis.tidy import validate_tidy

HERE = os.path.dirname(os.path.abspath(__file__))

MODELS = ["ARIMA", "SARIMA", "RandomForest", "XGBoost"]
CHANNEL = "PeakConsumption"


def classify(reason: str) -> str:
    """Map a ``validate_series`` reason string to a triage scenario slug.

    Mirrors the validator's own branch order (short → gap → zero → negative);
    this is the single place the reason taxonomy is bucketed into the marks the
    triage figure groups on.
    """
    if reason.startswith("series too short"):
        return "short_history"
    if reason.startswith("gaps ≤ 3 months"):
        return "gap_within_limit"
    if reason.startswith("gap of"):
        return "gap_too_large"
    if reason == "all-zero series":
        return "all_zero"
    if reason.startswith("negative values"):
        return "negative_values"
    return "ok"


def _months(start: str, periods: int):
    return pd.date_range(start, periods=periods, freq="MS")


def _entity_specs():
    """(entity_id, tariff_type, entity_type, months, values) per condition entity."""
    specs = []

    # 6-month series → short for all methods.
    m = _months("2024-07-01", 6)
    specs.append(("POD_SHORT_6M", "LPU", "POD", m, np.full(6, 200.0)))

    # 14-month series → short for ARIMA/SARIMA (18), valid for trees (12).
    m = _months("2023-11-01", 14)
    specs.append(("POD_SHORT_14M", "LPU", "POD", m, 200.0 + 10 * np.sin(np.arange(14))))

    # 24-month span with a 2-month hole → gap ≤ 3 (proceed).
    full = _months("2022-01-01", 24)
    keep = ~full.isin(_months("2022-06-01", 2))
    specs.append(("POD_GAP_2M", "LPU", "POD", full[keep],
                  200.0 + 15 * np.sin(np.arange(keep.sum()) / 12 * 2 * np.pi)))

    # 30-month span with a 5-month hole → gap > 3 (reject); 25 observed ≥ 18.
    full = _months("2021-01-01", 30)
    keep = ~full.isin(_months("2021-08-01", 5))
    specs.append(("POD_GAP_5M", "LPU", "POD", full[keep],
                  220.0 + 12 * np.sin(np.arange(keep.sum()) / 12 * 2 * np.pi)))

    # 24 all-zero months → reject.
    m = _months("2022-01-01", 24)
    specs.append(("COMBO_ZERO", "SPU", "Combo", m, np.zeros(24)))

    # 24 months with a sub-zero dip (credit) → warning only, proceed.
    m = _months("2022-01-01", 24)
    vals = 180.0 + 20 * np.sin(np.arange(24) / 12 * 2 * np.pi)
    vals[[7, 15]] = [-40.0, -25.0]
    specs.append(("POD_NEG", "LPU", "POD", m, vals))

    return specs


def build_tidy_real() -> pd.DataFrame:
    rng = np.random.default_rng(23)
    rows = []
    for entity_id, tariff, etype, months, values in _entity_specs():
        series = pd.DataFrame({CHANNEL: values}, index=pd.DatetimeIndex(months, name="ReportingMonth"))
        unit = PredictionUnit(
            entity_id=entity_id, entity_type=etype, tariff_type=tariff,
            customer_id="SYNTH", tariff_id=0, series=series,
        )
        for model in MODELS:
            ok, reason = validate_series(unit, model)
            scenario = classify(reason)
            for ds, actual in zip(months, values):
                yhat = float(actual + rng.normal(0, 5)) if ok else np.nan
                rows.append({
                    "EntityID": entity_id,
                    "TariffType": tariff,
                    "EntityType": etype,
                    "consumption_type": CHANNEL,
                    "param_set_id": "default",
                    "model": model,
                    "ds": ds,
                    "y": float(actual),
                    "y_hat": yhat,
                    "y_hat_lower": np.nan,
                    "y_hat_upper": np.nan,
                    "scenario": scenario,
                    "validation_reason": reason,
                })
    return validate_tidy(pd.DataFrame(rows))


def main():
    df = build_tidy_real()
    out = os.path.join(HERE, "tidy_real.parquet")
    df.to_parquet(out, index=False)
    print(f"wrote {out}  ({len(df)} rows)")
    print(df.groupby("scenario")["EntityID"].nunique().to_string())


if __name__ == "__main__":
    main()
