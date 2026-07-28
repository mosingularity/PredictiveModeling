"""The tidy forecast contract and its non-invasive adapter.

This module defines the single tidy long-format table that the whole
results-analysis pack reads, a ``to_tidy()`` that produces it from the runners'
existing collation, and a forgiving ``validate_tidy()`` that guards
hand-produced files. It is the only coupling point between the generate side
(the four ESF-* runners) and the analyze side (every figure in this scope).

Tidy schema
-----------
One row per ``(EntityID, consumption_type, model, param_set_id, ds)``:

==================  ========  ========================================================
column              null?     meaning
==================  ========  ========================================================
EntityID            no        entity key (POD / Combo / CSA id) the forecast belongs to
TariffType          no        "LPU" | "SPU" | "PPU"
EntityType          no        "POD" | "Combo" | "CSA" | "Bundle"
consumption_type    no        consumption channel, e.g. "PeakConsumption"
param_set_id        no        hyperparameter-set label; "default" for a base run, a
                              distinct id per OFAT variant in a sweep
model               no        forecasting method name, e.g. "ARIMA" / "RF" / "XGBoost"
ds                  no        forecast timestamp (month-start), datetime64[ns]
y                   yes       observed actual at ``ds`` (NaN where unavailable — the
                              runners' forecast Series carries predictions only)
y_hat               no        point forecast at ``ds``
y_hat_lower         yes       lower confidence bound (NaN ⇒ point-only)
y_hat_upper         yes       upper confidence bound (NaN ⇒ point-only)
scenario            yes       series_validator scenario mark, where present
validation_reason   yes       series_validator reason string, where present
==================  ========  ========================================================

The confidence columns and ``y`` are nullable by design: the unbundled runners
emit point forecasts only, so the default frame is point-only and actual-free.
``validate_tidy()`` materialises any missing nullable column as all-NaN so
downstream figures always see the full schema.
"""
from typing import Optional

import pandas as pd

from evaluation.performance import EntityPerformanceData, ForecastResults

# Full ordered column set of the tidy contract.
TIDY_COLUMNS = [
    "EntityID",
    "TariffType",
    "EntityType",
    "consumption_type",
    "param_set_id",
    "model",
    "ds",
    "y",
    "y_hat",
    "y_hat_lower",
    "y_hat_upper",
    "is_forecast",
    "RMSE",
    "MAE",
    "R2",
    "scenario",
    "validation_reason",
]

# Columns that must be present on any frame claiming to be tidy. The remaining
# columns are nullable and are materialised as all-NaN when absent.
REQUIRED_COLUMNS = [
    "EntityID",
    "TariffType",
    "EntityType",
    "consumption_type",
    "param_set_id",
    "model",
    "ds",
    "y_hat",
]

OPTIONAL_COLUMNS = [c for c in TIDY_COLUMNS if c not in REQUIRED_COLUMNS]

# Default param-set label for a base (non-sweep) run.
DEFAULT_PARAM_SET_ID = "default"

# Per-row keys that, when present on the performance frame, carry the
# series_validator marks through to the tidy table.
_SCENARIO_KEYS = ("scenario", "Scenario")
_VALIDATION_REASON_KEYS = ("validation_reason", "ValidationReason", "reason")
_PARAM_SET_KEYS = ("param_set_id", "ParamSetID")


def _first_present(row: pd.Series, keys) -> Optional[object]:
    """First non-null value among ``keys`` on a performance-frame row, else None."""
    for key in keys:
        if key in row and pd.notna(row[key]):
            return row[key]
    return None


def to_tidy(
    results: ForecastResults,
    *,
    param_set_id: str = DEFAULT_PARAM_SET_ID,
) -> pd.DataFrame:
    """Flatten an ``ForecastResults`` into the tidy forecast contract.

    Non-invasive: this reads only the public ``ForecastResults`` /
    ``EntityPerformanceData`` shapes — the nested ``forecast`` Series inside each
    ``performance_data_frame`` row plus the entity identifiers on the wrapper. It
    touches no runner internals and produces point-only, actual-free rows
    (``y`` / ``y_hat_lower`` / ``y_hat_upper`` are NaN unless carried on the
    frame). ``scenario`` / ``validation_reason`` are carried from
    series_validator marks where the frame exposes them.

    The ``model`` column is stamped from the run's ``forecast_method_name`` so
    that frames from the four runners can be ``pd.concat``-unioned into one table.

    Parameters
    ----------
    results : ForecastResults
        A single model's unbundled run (one ``forecast_method_name``).
    param_set_id : str, optional
        Label for this run's hyperparameter set. Defaults to ``"default"`` for a
        base run; a sweep stamps a distinct id per OFAT variant. Overridden
        per-row if the frame carries its own ``param_set_id`` column.

    Returns
    -------
    pd.DataFrame
        Tidy long-format frame with columns :data:`TIDY_COLUMNS`, validated and
        column-ordered. Empty (but correctly-typed) when the run produced no
        forecasts.
    """
    rows = []
    for entity in results.entity_performance:
        model = entity.forecast_method_name or results.forecast_method_name
        rows.extend(_entity_rows(entity, model, param_set_id))

    if not rows:
        return validate_tidy(pd.DataFrame(columns=TIDY_COLUMNS))

    return validate_tidy(pd.DataFrame(rows))


def _metric(frame_row, key):
    """A scalar metric off the performance row (the value your runners scored), or None."""
    if key in frame_row and pd.notna(frame_row[key]):
        return float(frame_row[key])
    return None


def _entity_rows(entity: EntityPerformanceData, model: str, param_set_id: str):
    """Yield one tidy row dict per (consumption_type, ds) for a single entity.

    Emits each model's *in-sample* prediction (``is_forecast=False`` — the
    fitted-history line) and its *future* forecast (``is_forecast=True`` — the
    forecast line) into the same ``y_hat`` column, distinguished by
    ``is_forecast``, and stamps the run's real ``RMSE/MAE/R²`` (from
    ``evaluation`` via the runners) onto every row. ``y`` stays NaN — actuals are
    the input, joined by the figure layer, not a forecast output.
    """
    pdf = entity.performance_data_frame
    if pdf is None or len(pdf) == 0:
        return

    for _, frame_row in pdf.iterrows():
        forecast = frame_row.get("forecast")
        in_sample = frame_row.get("in_sample")
        has_forecast = isinstance(forecast, pd.Series) and not forecast.empty
        has_in_sample = isinstance(in_sample, pd.Series) and not in_sample.empty
        if not has_forecast and not has_in_sample:
            continue

        consumption_type = frame_row.get("consumption_type")
        scenario = _first_present(frame_row, _SCENARIO_KEYS)
        validation_reason = _first_present(frame_row, _VALIDATION_REASON_KEYS)
        row_param_set = _first_present(frame_row, _PARAM_SET_KEYS) or param_set_id
        rmse, mae, r2 = _metric(frame_row, "RMSE"), _metric(frame_row, "MAE"), _metric(frame_row, "R2")

        def _row(ds, y_hat, is_forecast):
            return {
                "EntityID": entity.entity_id,
                "TariffType": entity.tariff_type,
                "EntityType": entity.entity_type,
                "consumption_type": consumption_type,
                "param_set_id": row_param_set,
                "model": model,
                "ds": ds,
                "y": None,
                "y_hat": y_hat,
                "y_hat_lower": None,
                "y_hat_upper": None,
                "is_forecast": is_forecast,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2,
                "scenario": scenario,
                "validation_reason": validation_reason,
            }

        if has_in_sample:
            for ds, y_hat in in_sample.items():
                yield _row(ds, y_hat, is_forecast=False)
        if has_forecast:
            for ds, y_hat in forecast.items():
                yield _row(ds, y_hat, is_forecast=True)


def validate_tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalise a tidy forecast frame.

    Forgiving by design so it can guard both ``to_tidy()`` output and
    hand-produced files: it asserts the required columns are present, coerces
    ``ds`` to ``datetime64[ns]``, and materialises any missing nullable column
    (``y`` / ``y_hat_lower`` / ``y_hat_upper`` / ``scenario`` /
    ``validation_reason``) as all-NaN — so a point-only file is accepted and
    returned with the full schema.

    Raises
    ------
    ValueError
        Naming the first missing required column.

    Returns
    -------
    pd.DataFrame
        The same data, ``ds``-coerced and column-ordered to :data:`TIDY_COLUMNS`.
    """
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"tidy frame missing required column: {col!r}")

    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])

    # Tolerate point-only / mark-free files: fill in any absent nullable column.
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            # A frame that predates the in-sample/forecast split is all forecast.
            df[col] = True if col == "is_forecast" else pd.NA

    return df[TIDY_COLUMNS]
