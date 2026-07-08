"""Forecast engine — runs the *real* production code, returns the tidy contract.

This is the heart of the "dashboard as a viewer/test-harness for our own code"
idea: it does **not** reimplement any forecasting. For each (model, entity) it
calls the same per-entity entry point the ESF-* runners call —
``forecast_for_entity`` in ``models/algorithms/autoarima`` and
``…/tree_algorithms/{rf,xgb}`` — collects the ``EntityPerformanceData`` they
return, and flattens it through the real :func:`results_analysis.tidy.to_tidy`.
So what the dashboard shows is exactly what production produces; visualizing it
*is* exercising the code.

It runs locally with no Spark: the per-entity functions take an already-loaded
``PredictionUnit`` (pandas), a plain ``ForecastConfig`` (whose ``start_date`` /
``end_date`` set the forecast horizon), and only touch ``forecast_model.config.log``
— so a duck-typed stub stands in. Profiler error-logging is disabled so nothing
reaches the database.
"""
import logging
from types import SimpleNamespace
from typing import Dict, List, Optional

import pandas as pd

from db.queries import ForecastConfig
from evaluation.performance import PredictionUnit, UnbundledResults
from hyperparameters import get_model_hyperparameters
from models.algorithms import autoarima
from models.algorithms.tree_algorithms import rf as rf_alg
from models.algorithms.tree_algorithms import xgb as xgb_alg
from results_analysis.tidy import to_tidy, validate_tidy
from validation.series import LEGACY_CONSUMPTION_COLUMNS

logger = logging.getLogger(__name__)

# The four production models, by tidy ``model`` name.
ARIMA_METHODS = {"ARIMA", "SARIMA"}
TREE_ALGOS = {"RandomForest": rf_alg, "XGBoost": xgb_alg}
ALL_MODELS = ["ARIMA", "SARIMA", "RandomForest", "XGBoost"]

# Shared held-out window every model is scored on, so the legend RMSE/MAE/R² are
# comparable across families (ARIMA/SARIMA via backtest_months, trees via
# test_months — both default to 3 otherwise, but only here are they forced equal).
BACKTEST_MONTHS = 6


def _forecast_model_stub():
    """Minimal stand-in — the per-entity functions only read ``config.log``."""
    return SimpleNamespace(config=SimpleNamespace(log=False))


def _ufm_config(model: str, param_str: str, start, end) -> ForecastConfig:
    """A plain ForecastConfig; start/end define the forecast horizon."""
    return ForecastConfig(
        forecast_method_id=1,
        forecast_method_name=model,
        model_parameters=param_str,
        region="LOCAL",
        status="Active",
        user_forecast_method_id=99,
        start_date=pd.Timestamp(start),
        end_date=pd.Timestamp(end),
        databrick_task_id=0,
    )


def _entity_series(entity_df: pd.DataFrame) -> pd.DataFrame:
    """One entity's wide consumption frame, month-indexed, with a PodID column.

    ``forecast_for_podel_id`` filters ``df[df["PodID"] == pod_id]`` and reads the
    consumption columns, so the unit's series must carry both. Duplicate months
    (a Combo's sub-meter rows) are summed — the bottom-up aggregation the runners
    receive pre-aggregated.
    """
    cols = [c for c in LEGACY_CONSUMPTION_COLUMNS if c in entity_df.columns]
    frame = (entity_df[["ReportingMonth", *cols]]
             .groupby("ReportingMonth", as_index=True).sum(min_count=1)
             .sort_index())
    frame.index = pd.DatetimeIndex(frame.index)
    frame.insert(0, "PodID", str(entity_df["EntityID"].iloc[0]))
    return frame


def _horizon(df: pd.DataFrame, months: int):
    """Global forecast window: ``months`` month-starts beyond the latest observed."""
    last = pd.to_datetime(df["ReportingMonth"]).max()
    start = last + pd.DateOffset(months=1)
    end = last + pd.DateOffset(months=months)
    return start, end


def _entity_meta(entity_df: pd.DataFrame) -> dict:
    return {
        "entity_id": str(entity_df["EntityID"].iloc[0]),
        "tariff_type": str(entity_df["TariffType"].iloc[0]),
        "entity_type": str(entity_df["EntityType"].iloc[0]) if "EntityType" in entity_df else "POD",
        "customer_id": str(entity_df["CustomerID"].iloc[0]) if "CustomerID" in entity_df else "",
        "tariff_id": 0,
    }


def forecast_entity_model(unit: PredictionUnit, model: str, param_str: str, ufm_config, forecast_model):
    """Call the real per-entity entry point for one model → EntityPerformanceData."""
    if model in ARIMA_METHODS:
        order, seasonal_order = get_model_hyperparameters(model, param_str)
        return autoarima.forecast_for_entity(
            unit, order, ufm_config, forecast_model, seasonal_order=seasonal_order,
            backtest_months=BACKTEST_MONTHS,
        )
    if model in TREE_ALGOS:
        return TREE_ALGOS[model].forecast_for_entity(
            unit, ufm_config, forecast_model, test_months=BACKTEST_MONTHS,
        )
    raise ValueError(f"unknown model {model!r}; expected one of {ALL_MODELS}")


def run_forecasts(
    df: pd.DataFrame,
    model_params: Dict[str, str],
    *,
    horizon_months: int = 12,
    entities: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run the real forecast code for the given models → one validated tidy frame.

    ``model_params`` maps each model name to its ``model_parameters`` string (the
    same encoding the runners parse), which becomes the tidy ``param_set_id``. One
    ``EntityPerformanceData`` per entity is collected per model and flattened with
    the real :func:`to_tidy`. Per-entity failures are logged and skipped so one bad
    series never sinks the run.
    """
    forecast_model = _forecast_model_stub()
    start, end = _horizon(df, horizon_months)
    keep = entities or sorted(df["EntityID"].dropna().unique())

    frames: List[pd.DataFrame] = []
    for model, param_str in model_params.items():
        ufm_config = _ufm_config(model, param_str, start, end)
        results = UnbundledResults(forecast_method_name=model)
        for entity_id in keep:
            entity_df = df[df["EntityID"] == entity_id]
            if entity_df.empty:
                continue
            meta = _entity_meta(entity_df)
            unit = PredictionUnit(series=_entity_series(entity_df), **meta)
            try:
                epd = forecast_entity_model(unit, model, param_str, ufm_config, forecast_model)
                results.entity_performance.append(epd)
            except Exception as e:  # a single entity/model failing must not sink the run
                logger.warning("forecast failed for %s / %s: %s", entity_id, model, e)
        frames.append(to_tidy(results, param_set_id=param_str))

    if not frames:
        return validate_tidy(pd.DataFrame())
    return validate_tidy(pd.concat(frames, ignore_index=True))


# Repo-default param strings (mirror config.yaml model_defaults); override per call.
DEFAULT_PARAMS = {
    "ARIMA": "(3,3,3)",
    "SARIMA": "(1,1,1)(1,1,1,12)",
    "RandomForest": "(100,10,2,1,5,true)",
    "XGBoost": "(100,5,0.1,0.8,0.8)",
}
