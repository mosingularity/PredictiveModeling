"""Run production forecasters locally and return their validated tidy output."""
import logging
from types import SimpleNamespace
from typing import Dict, List, Optional

import pandas as pd

from db.queries import ForecastConfig
from evaluation.performance import PredictionUnit, ForecastResults
from hyperparameters import get_model_hyperparameters
from models.algorithms import autoarima
from models.algorithms.tree_algorithms import rf as rf_alg
from models.algorithms.tree_algorithms import xgb as xgb_alg
from results_analysis.tidy import to_tidy, validate_tidy
from validation.series import LEGACY_CONSUMPTION_COLUMNS

logger = logging.getLogger(__name__)

ARIMA_METHODS = {"ARIMA", "SARIMA"}
TREE_ALGOS = {"RandomForest": rf_alg, "XGBoost": xgb_alg}
ALL_MODELS = ["ARIMA", "SARIMA", "RandomForest", "XGBoost"]

# Use the same held-out window so metrics are comparable across model families.
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
    """Build one entity's month-indexed frame, summing duplicate sub-meter rows."""
    cols = [c for c in LEGACY_CONSUMPTION_COLUMNS if c in entity_df.columns]
    frame = (
        entity_df[["ReportingMonth", *cols]]
        .groupby("ReportingMonth", as_index=True)
        .sum(min_count=1)
        .sort_index()
    )
    frame.index = pd.DatetimeIndex(frame.index)
    frame.insert(0, "PodID", str(entity_df["EntityID"].iloc[0]))
    return frame


def _horizon(df: pd.DataFrame, months: int):
    """Return a forecast window beyond the latest observed month."""
    observed_months = pd.to_datetime(df["ReportingMonth"])
    last = observed_months.max()
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


def forecast_entity_model(
    unit: PredictionUnit, model: str, param_str: str, ufm_config, forecast_model
):
    """Run one production model for one entity."""
    if model in ARIMA_METHODS:
        order, seasonal_order = get_model_hyperparameters(model, param_str)
        return autoarima.forecast_for_entity(
            unit,
            order,
            ufm_config,
            forecast_model,
            seasonal_order=seasonal_order,
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
    """Run each model for the selected entities and return one validated tidy frame."""
    forecast_model = _forecast_model_stub()
    start, end = _horizon(df, horizon_months)
    entity_ids = df["EntityID"].dropna().unique()
    keep = entities or sorted(entity_ids)

    frames: List[pd.DataFrame] = []
    for model, param_str in model_params.items():
        ufm_config = _ufm_config(model, param_str, start, end)
        results = ForecastResults(forecast_method_name=model)
        for entity_id in keep:
            entity_df = df[df["EntityID"] == entity_id]
            if entity_df.empty:
                continue
            meta = _entity_meta(entity_df)
            series = _entity_series(entity_df)
            unit = PredictionUnit(series=series, **meta)
            try:
                epd = forecast_entity_model(unit, model, param_str, ufm_config, forecast_model)
                results.entity_performance.append(epd)
            except Exception as exc:  # one failed entity must not sink the run
                logger.warning("forecast failed for %s / %s: %s", entity_id, model, exc)
        tidy = to_tidy(results, param_set_id=param_str)
        frames.append(tidy)

    if not frames:
        return validate_tidy(pd.DataFrame())
    combined = pd.concat(frames, ignore_index=True)
    return validate_tidy(combined)


# Mirrors config.yaml model_defaults; callers can override each value.
DEFAULT_PARAMS = {
    "ARIMA": "(3,3,3)",
    "SARIMA": "(1,1,1)(1,1,1,12)",
    "RandomForest": "(100,10,2,1,5,true)",
    "XGBoost": "(100,5,0.1,0.8,0.8)",
}
