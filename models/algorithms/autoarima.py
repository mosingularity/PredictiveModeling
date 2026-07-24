import logging
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from data.dml import get_forecast_range
from db.error_logger import report_validation_error
from evaluation.performance import EntityPerformanceData, ForecastResults, PodIDPerformanceData, PredictionUnit
from hyperparameters import get_model_hyperparameters
from models.algorithms.helper import _apply_log, _collect_metrics
from models.algorithms.unbundled import run_unbundled
from models.algorithms.utilities import evaluate_predictions, prepare_time_series_data
from models.base import ForecastModel
from validation.metadata import get_error_metadata
from validation.series import consumption_columns

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning,
                        message=r"Non-(stationary|invertible) starting")


def fit_time_series_model(series, order, seasonal_order, log=False, exog=None):
    if seasonal_order:
        return train_sarima_model(series, order, seasonal_order, log, exog)
    return train_arima_model(series, order, log, exog)

def train_arima_model(series: pd.Series, order: Tuple[int, int, int], log: bool = False, exog: Optional[pd.DataFrame] = None):
    series = _apply_log(series, log)
    model = ARIMA(series, order=order, exog=exog)
    return model.fit(method_kwargs={"maxiter": 500, "disp": False})


def train_sarima_model(series: pd.Series, order: Tuple[int, int, int], seasonal_order: Tuple[int, int, int, int], log: bool = False, exog: Optional[pd.DataFrame] = None):
    series = _apply_log(series, log)
    model = SARIMAX(endog=series, exog=exog, order=order, seasonal_order=seasonal_order)
    return model.fit(maxiter=200, disp=False)




def forecast_arima_unbundled(model: ForecastModel, spark) -> ForecastResults:
    """ARIMA/SARIMA unbundled run: delegates to run_unbundled via an adapter that
    threads the order/seasonal_order into each entity's forecast call."""
    ufm_config = model.dataset.ufm_config
    order, seasonal_order = get_model_hyperparameters(
        ufm_config.forecast_method_name, ufm_config.model_parameters)

    def _forecast_arima_entity(unit, cfg, m):
        return forecast_for_entity(unit, order, cfg, m, seasonal_order)

    return run_unbundled(model, spark, _forecast_arima_entity)


def forecast_for_pod_id(
    df: pd.DataFrame,
    order: Tuple[int, int, int],
    customer_id: str,
    pod_id: str,
    consumption_types: List[str],
    ufm_config,
    forecast_model: ForecastModel,
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    gap_handling: str = "skip",
    backtest_months: Optional[int] = None,
) -> PodIDPerformanceData:
    """Forecast one pod with ARIMA/SARIMA: fit, backtest, and forecast the
    horizon, per consumption type.

    ``backtest_months`` sets the held-out window the metrics are scored on;
    it defaults to the forecast horizon so results stay comparable across models.
    """
    data = []
    forecast_horizon = get_forecast_range(ufm_config)
    steps = len(forecast_horizon)

    for consumption_type in consumption_types:
        pod_df = df[df["PodID"] == pod_id].sort_index()
        series = prepare_time_series_data(pod_df, consumption_type)
        if series.isnull().all() or series.nunique() <= 1:
            meta = get_error_metadata("InvalidSeries", {"pod_id": pod_id, "consumption_type":consumption_type})
            report_validation_error(
                log_id=None,
                error=meta["message"],
                traceback="",  # or traceback.format_exc()
                error_type="InvalidSeries",
                severity=meta["severity"],
                component=meta["component"],
                entity_id=pod_id,
            )
            forecast = pd.Series([0] * steps, index=forecast_horizon)
            row = _collect_metrics(pod_id, customer_id, consumption_type, forecast,
                                   validation_reason="all-zero / flat series")
            data.append(row)
            continue

        if len(series) < 12:
            meta = get_error_metadata("SplitConfigurationError", {
                "series_length": len(series),
                "consumption_type": consumption_type
            })
            report_validation_error(
                log_id=None,
                error=meta["message"],
                traceback="",  # or traceback.format_exc()
                error_type="SplitConfigurationError",
                severity=meta["severity"],
                component=meta["component"],
                entity_id=pod_id,
            )
            forecast = pd.Series([0] * steps, index=forecast_horizon)
            row = _collect_metrics(pod_id, customer_id, consumption_type, forecast,
                                   validation_reason="series too short")
            data.append(row)
            continue

        last_date = series.index[-1]
        forecast_start,forecast_end = forecast_horizon[0], forecast_horizon[-1]

        gap_months = (forecast_end.year - last_date.year) * 12 + (forecast_end.month - last_date.month)
        total_steps = steps
        if last_date < forecast_horizon[0] - pd.DateOffset(months=1):
            if gap_handling == "skip":
                meta = get_error_metadata("ForecastGapTooLarge", {
                    "pod_id": pod_id,
                    "last_observed": str(last_date.date()),
                    "requested_start": str(forecast_start.date())
                })
                report_validation_error(
                    log_id=None,
                    error=meta["message"],
                    traceback="",  # or traceback.format_exc()
                    error_type="ForecastGapTooLarge",
                    severity=meta["severity"],
                    component=meta["component"],
                    entity_id=pod_id,
                )
                forecast = pd.Series([0] * steps, index=forecast_horizon)
                row = _collect_metrics(pod_id, customer_id, consumption_type, forecast,
                                       validation_reason="gap too large")
                data.append(row)
                continue
            elif gap_handling == "fill":
                total_steps = gap_months
                logger.warning(
                    f"⚠️ Filling gap of {gap_months} months from {last_date.date()} to {forecast_start.date()}"
                )
        # Fit model
        try:
            model = fit_time_series_model(series, order, seasonal_order, log=forecast_model.config.log)
            future_forecast = model.get_forecast(steps=total_steps).predicted_mean
            std = future_forecast.std() / future_forecast.mean()
            if std < 1e-2:
                meta = get_error_metadata("FlatForecast", {"gap_months": gap_months, "last_date":last_date.date(),"forecast_start":forecast_start.date(), "pod_id": pod_id,"consumption_type": consumption_type})
                report_validation_error(
                    log_id=None,
                    error=meta["message"],
                    traceback="",
                    error_type="FlatForecast",
                    severity=meta["severity"],
                    component=meta["component"],
                    entity_id=pod_id,
                )
            if forecast_model.config.log:
                future_forecast = np.exp(future_forecast)
            evaluation_window = min(backtest_months or steps, len(series))
            test_actual = series[-evaluation_window:]
            train = series.iloc[:-evaluation_window]
            metrics, baseline_metrics = None, None
            if len(train) >= max(evaluation_window, 4):
                try:
                    bt_model = fit_time_series_model(
                        train, order, seasonal_order, log=forecast_model.config.log)
                    bt_pred = bt_model.get_forecast(steps=evaluation_window).predicted_mean
                    if forecast_model.config.log:
                        bt_pred = np.exp(bt_pred)
                    bt_pred.index = test_actual.index
                    metrics, baseline_metrics = evaluate_predictions(test_actual, bt_pred)
                except Exception:
                    metrics, baseline_metrics = None, None
            in_sample_fit = model.predict(start=series.index[0], end=series.index[-1])
            if forecast_model.config.log:
                in_sample_fit = np.exp(in_sample_fit)
            if gap_handling == "fill":
                future_forecast = future_forecast[future_forecast.index.isin(forecast_horizon)].copy()
                future_forecast = future_forecast.reindex(forecast_horizon).dropna()
            row = _collect_metrics(
                pod_id, customer_id, consumption_type,
                future_forecast, metrics, baseline_metrics, in_sample=in_sample_fit
            )
            data.append(row)
        except Exception as e:
            meta = get_error_metadata("ModelFitFailure", {"exception": str(e)})
            report_validation_error(log_id=None, error=meta["message"], traceback="",  error_type="ModelFitFailure",severity=meta["severity"], component=meta["component"],
                                    entity_id=pod_id)
            forecast = pd.Series([0] * steps, index=forecast_horizon)
            row = _collect_metrics(pod_id, customer_id, consumption_type, forecast,
                                   validation_reason="model fit failed")
            data.append(row)
            continue
    performance_frame = pd.DataFrame(data)
    return PodIDPerformanceData(
        pod_id=pod_id,
        forecast_method_name=ufm_config.forecast_method_name,
        customer_id=customer_id,
        user_forecast_method_id=ufm_config.user_forecast_method_id,
        performance_data_frame=performance_frame
    )


def forecast_for_entity(
    unit: PredictionUnit,
    order: Tuple[int, int, int],
    ufm_config,
    forecast_model: ForecastModel,
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    gap_handling: str = "skip",
    backtest_months: Optional[int] = None,
) -> EntityPerformanceData:
    """Adapt forecast_for_pod_id to the unbundled path's PredictionUnit contract."""
    consumption_types = consumption_columns(unit.series)
    pod_perf = forecast_for_pod_id(
        df=unit.series,
        order=order,
        customer_id=unit.customer_id,
        pod_id=unit.entity_id,
        consumption_types=consumption_types,
        ufm_config=ufm_config,
        forecast_model=forecast_model,
        seasonal_order=seasonal_order,
        gap_handling=gap_handling,
        backtest_months=backtest_months,
    )
    return EntityPerformanceData(
        entity_id=unit.entity_id,
        entity_type=unit.entity_type,
        tariff_type=unit.tariff_type,
        customer_id=unit.customer_id,
        forecast_method_name=ufm_config.forecast_method_name,
        user_forecast_method_id=ufm_config.user_forecast_method_id,
        performance_data_frame=pod_perf.performance_data_frame,
        tariff_id=unit.tariff_id,
    )