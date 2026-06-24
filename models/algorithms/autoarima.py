from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX
from utils.exit_handler import safe_exit

from data.dml import *
import pandas as pd
from typing import Tuple, NamedTuple

from db.error_logger import report_validation_error
from models.algorithms._bundled import run_bundled
from evaluation.performance import *
from hyperparameters import get_model_hyperparameters
from models.algorithms.helper import _convert_to_model_performance_row, \
    _convert_forecast_map_to_df, _get_customer_data, _collect_metrics, _apply_log, ensure_numeric_consumption_types
from models.algorithms.utilities import prepare_time_series_data, evaluate_predictions
from models.base import ForecastModel
from validation.forecast import run_forecast_sanity_checks
from validation.series import validate_series, consumption_columns

# Setup logger with basic configuration
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

performance_metrics_table = "dbo.StatisticalPerformanceMetrics"
target_table_name = "dbo.ForecastFact"



# Optionally, define a named tuple for forecast results
class ForecastResult(NamedTuple):
    forecast: pd.Series
    metrics: dict
    baseline_metrics: dict


def fit_time_series_model(series, order, seasonal_order, log=False, exog=None):
    if seasonal_order:
        return train_sarima_model(series, order, seasonal_order, log, exog)
    return train_arima_model(series, order, log, exog)

def train_arima_model(series: pd.Series, order: Tuple[int, int, int], log: bool = False, endog: Optional[pd.DataFrame] = None, exog: Optional[pd.DataFrame] = None):
    # logger.info("🔧 Training ARIMA")
    series = _apply_log(series, log)
    model = ARIMA(series, order=order, exog=exog)
    # NB: tsa.arima.model.ARIMA.fit() has no `disp` arg — it goes in method_kwargs
    # (SARIMAX.fit below takes it directly). Passing it directly here silently
    # crashes every ARIMA fit into the zero-fallback.
    return model.fit(method_kwargs={"maxiter": 500, "disp": False})


def train_sarima_model(series: pd.Series, order: Tuple[int, int, int], seasonal_order: Tuple[int, int, int, int], log: bool = False, endog: Optional[pd.DataFrame] = None, exog: Optional[pd.DataFrame] = None):
    # logger.info(f"🔧 Training seasonal SARIMA")
    series = _apply_log(series, log)
    model = SARIMAX(endog=series,exog=exog, order=order, seasonal_order=seasonal_order)
    return model.fit(method_kwargs={"maxiter": 200}, disp=False)




def predict_time_series_model(model, steps, return_ci=False, log = False):
    forecast = model.get_forecast(steps=steps)
    mean_forecast = forecast.predicted_mean
    if log:
        mean_forecast = np.exp(mean_forecast)
    if return_ci:
        conf_int = forecast.conf_int()
        return mean_forecast, conf_int
    return mean_forecast

def _forecast_arima_pod(pod_df, customer_id, pod_id, consumption_types, ufm_config, model):
    order, seasonal_order = get_model_hyperparameters(
        ufm_config.forecast_method_name, ufm_config.model_parameters)
    return forecast_for_podel_id(
        pod_df, order, customer_id, pod_id, consumption_types, ufm_config,
        forecast_model=model, seasonal_order=seasonal_order)


def forecast_arima_for_single_customer(model: ForecastModel, spark):
    return run_bundled(model, spark, _forecast_arima_pod)

def forecast_arima_unbundled(model: ForecastModel, spark) -> UnbundledResults:
    ufm_config = model.dataset.ufm_config
    order, seasonal_order = get_model_hyperparameters(ufm_config.forecast_method_name, ufm_config.model_parameters)
    data = get_predictive_data(spark, ufm_config.user_forecast_method_id)
    results = UnbundledResults(forecast_method_name=ufm_config.forecast_method_name)
    for (tariff_type, entity_id), group in data.groupby(["TariffType", "EntityID"]):
        series = group.set_index("ReportingMonth").sort_index()
        if "PodID" not in series.columns:
            series = series.copy()
            series["PodID"] = entity_id
        unit = PredictionUnit(
            entity_id=entity_id,
            entity_type=group["EntityType"].iloc[0],
            tariff_type=tariff_type,
            # Real unbundled exports (Ermelo) are entity-keyed and carry no
            # CustomerID; it is reporting metadata only, so default to "" when absent.
            customer_id=str(group["CustomerID"].iloc[0]) if "CustomerID" in group.columns else "",
            tariff_id=group["TariffID"].iloc[0],
            series=series,
        )
        ok, reason = validate_series(unit, ufm_config.forecast_method_name)
        if not ok:
            meta = get_error_metadata("SeriesValidationFailed", {"entity_id": entity_id, "reason": reason})
            report_validation_error(
                log_id=None,
                error=meta["message"],
                traceback="",
                error_type="SeriesValidationFailed",
                severity=meta["severity"],
                component=meta["component"],
            )
            continue
        if reason != "ok":
            logger.warning(f"⚠️ Series validation warning for entity {entity_id}: {reason}")
        result = forecast_for_entity(unit, order, ufm_config, model, seasonal_order)
        results.entity_performance.append(result)
    return results



def plot_prediction(series, future_forecast, consumption_type):
    plt.figure()
    plt.plot(series.index, series, label='Historical')
    plt.plot(future_forecast.index, future_forecast, label='Forecast')
    plt.title(f'{consumption_type} Forecast')
    plt.xlabel('Reporting Month')
    plt.ylabel(f'{consumption_type}')
    plt.legend()
    plt.show()


def forecast_for_podel_id(
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
    """
    Refactored forecasting function handling in-sample, out-of-sample, and future forecasts.

    ``backtest_months`` sets the held-out window the in-sample metrics are scored
    on. Defaults to the forecast horizon (legacy behaviour); pass an explicit value
    to decouple scoring from how far ahead you forecast and to keep the window
    comparable across models.
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
                component=meta["component"]
            )
            # logger.warning(f"⚠️ Invalid series for {consumption_type} @ Pod {pod_id}. Skipping.")
            forecast = pd.Series([0] * steps, index=forecast_horizon)
            data.append(_collect_metrics(pod_id, customer_id, consumption_type, forecast, validation_reason="all-zero / flat series"))
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
                component=meta["component"]
            )
            forecast = pd.Series([0] * steps, index=forecast_horizon)
            data.append(_collect_metrics(pod_id, customer_id, consumption_type, forecast, validation_reason="series too short"))
            # logger.info(meta["message"])
            continue

        # Check for large time gap
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
                    component=meta["component"]
                )
                # logger.info(f"⛔ Forecast gap too large for {consumption_type} @ Pod {pod_id}. Skipping.")
                forecast = pd.Series([0] * steps, index=forecast_horizon)
                data.append(_collect_metrics(pod_id, customer_id, consumption_type, forecast, validation_reason="gap too large"))
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
                # logger.warning(
                #     f"⚠️ Forecast for Pod {pod_id}, {consumption_type} is flat. Model may be underfit or data insufficient. Please try different parameters (e.g (2,1,2)) or a different model")
                meta = get_error_metadata("FlatForecast", {"gap_months": gap_months, "last_date":last_date.date(),"forecast_start":forecast_start.date(), "pod_id": pod_id,"consumption_type": consumption_type})
                report_validation_error(
                    log_id=None,
                    error=meta["message"],
                    traceback="",  # or traceback.format_exc()
                    error_type="ForecastGapTooLarge",
                    severity=meta["severity"],
                    component=meta["component"]
                )
            if forecast_model.config.log:
                future_forecast = np.exp(future_forecast)
            # plot_prediction(series, future_forecast, consumption_type)
            # Honest OUT-OF-SAMPLE backtest: refit on the training portion only and
            # forecast the held-out window — not an in-sample predict of months the
            # full-data model already saw. This matches the trees' train/test regime
            # so the metrics are comparable across all four models. Metrics are NaN
            # (unscored) when the training portion is too short to backtest.
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
            # Full-history in-sample fitted line for the diagnostic overlay (the solid
            # "predicted historical" line), from the full-data model — display only.
            in_sample_fit = model.predict(start=series.index[0], end=series.index[-1])
            if forecast_model.config.log:
                in_sample_fit = np.exp(in_sample_fit)
            if gap_handling == "fill":
                future_forecast = future_forecast[future_forecast.index.isin(forecast_horizon)].copy()
                future_forecast = future_forecast.reindex(forecast_horizon).dropna()
            data.append(_collect_metrics(
                pod_id, customer_id, consumption_type,
                future_forecast, metrics, baseline_metrics, in_sample=in_sample_fit
            ))
        except Exception as e:
            meta = get_error_metadata("ModelFitFailure", {"exception": str(e)})
            report_validation_error(log_id=None, error=meta["message"], traceback="",  error_type="ModelFitFailure",severity=meta["severity"], component=meta["component"])
            forecast = pd.Series([0] * steps, index=forecast_horizon)
            data.append(_collect_metrics(pod_id, customer_id, consumption_type, forecast, validation_reason="model fit failed"))
            continue
    return PodIDPerformanceData(
        pod_id=pod_id,
        forecast_method_name=ufm_config.forecast_method_name,
        customer_id=customer_id,
        user_forecast_method_id=ufm_config.user_forecast_method_id,
        performance_data_frame=pd.DataFrame(data)
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
    """
    Entity-aware wrapper around forecast_for_podel_id for the unbundled path.
    Accepts a PredictionUnit (carrying entity_id, entity_type, tariff_type) and
    returns EntityPerformanceData instead of PodIDPerformanceData.
    """
    consumption_types = consumption_columns(unit.series)
    pod_perf = forecast_for_podel_id(
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