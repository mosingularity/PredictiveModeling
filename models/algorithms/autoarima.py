from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX
from utils.exit_handler import safe_exit

from data.dml import *
import pandas as pd
from typing import Tuple, NamedTuple

from db.error_logger import insert_profiling_error
from evaluation.performance import *
from hyperparameters import get_model_hyperparameters
from models.algorithms.helper import _convert_to_model_performance_row, \
    _convert_forecast_map_to_df, _get_customer_data, _collect_metrics, _apply_log, ensure_numeric_consumption_types
from models.algorithms.utilities import prepare_time_series_data, evaluate_predictions
from models.base import ForecastModel
from models.forecast_validation import run_forecast_sanity_checks

# Setup logger with basic configuration
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

performance_metrics_table = "dbo.StatisticalPerformanceMetrics"
target_table_name = "dbo.ForecastFact"
write_url = "jdbc:sqlserver://fortrack-maz-sdb-san-prod-01.database.windows.net:1433;database=FortrackDB;Authentication=ActiveDirectoryMSI;trustServerCertificate=true"
write_properties = {
    "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver"
}



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
    return model.fit(method_kwargs={"maxiter": 500}, disp=False)


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

@profiled_function(category="model_training",enabled=profiling_switch.enabled)
def forecast_arima_for_single_customer(model: ForecastModel, spark):
    """
    Function: Forecast ARIMA/SARIMA models for a single customer using information embedded in the model instance.

    This function extracts all necessary parameters from the ForecastModel instance, including:
      - The processed DataFrame from the ForecastDataset.
      - The customer identifier.
      - Forecast configuration and selected consumption columns.

    It then:
      1. Extracts hyperparameters (order and seasonal_order) using the configured model_parameters string.
      2. Filters the data for the specific customer.
      3. Iterates over unique pod IDs for the customer.
      4. For each pod, trains a time series model and performs forecasting.
      5. Aggregates the performance metrics into a consolidated DataFrame.
    Args:
        model (ForecastModel): An instance of ForecastModel (or subclass) containing all forecasting parameters.
        log (bool, optional): Whether to apply log transformation. Defaults to False.

    Returns:
        pd.DataFrame: Aggregated forecasting results in long format.
    """
    try:
        forecast_method_id = getattr(model.dataset.ufm_config, "forecast_method_id", None)
        # Step 1: Validate dataset
        if model.dataset is None or model.dataset.processed_df is None:
            meta = get_error_metadata("ModelConfigMissing", {"field": "dataset.df"})
            insert_profiling_error(
                log_id=None,
                error=meta["message"],
                traceback="",  # or traceback.format_exc()
                error_type="ModelConfigMissing",
                severity=meta["severity"],
                component=meta["component"]
            )
            safe_exit(meta["code"], meta["message"])

        if model.dataset.processed_df.empty:
            meta = get_error_metadata("EmptySeries", {"forecast_method_id": forecast_method_id})
            insert_profiling_error(
                log_id=None,
                error=meta["message"],
                traceback="",  # or traceback.format_exc()
                error_type="EmptySeries",
                severity=meta["severity"],
                component=meta["component"]
            )
            safe_exit(meta["code"], meta["message"])

        df = model.dataset.processed_df
        df = ensure_numeric_consumption_types(df, model)
        unique_customers, unique_pod_ids = model.dataset.extract_unique_customers_and_pods()
        ufm_config = model.dataset.ufm_config
        consumption_types = getattr(model.dataset, 'variable_ids', None) or model.config.consumption_types

        order, seasonal_order = get_model_hyperparameters(ufm_config.forecast_method_name, ufm_config.model_parameters)
        # logger.info(f"💡 Extracted hyperparameters: order={order}, seasonal_order={seasonal_order}")

        all_forecasts = []
        arima_model_performances_dataframes: List[pd.DataFrame] = []
        for customer_id in unique_customers:
            customer_data = _get_customer_data(df, customer_id)
            if customer_data.empty:
                logger.warning(f"🚫 No data found for customer {customer_id}, skipping.")
                continue

            consumer_perf = CustomerPerformanceData(customer_id=customer_id, columns=consumption_types)
            arima_rows: List[ModelPodPerformance] = []

            unique_pod_ids = customer_data['PodID'].unique().tolist()
            for pod_id in unique_pod_ids:
                pod_df = customer_data[customer_data["PodID"] == pod_id].sort_values('ReportingMonth')
                # logger.info(f"🚀 Forecasting Customer {customer_id}, Pod {pod_id}")
                pod_perf = forecast_for_podel_id(
                    pod_df, order, customer_id, pod_id, consumption_types, ufm_config,
                    forecast_model=model, seasonal_order=seasonal_order)
                consumer_perf.pod_by_id_performance.append(pod_perf)
                # --- Performance Dataclass (Modularized) ---
                mpp = _convert_to_model_performance_row(pod_perf, customer_id, pod_id, ufm_config)
                arima_rows.append(mpp)

                # --- Forecast DataFrame (Pandas) ---
                forecast_df = _convert_forecast_map_to_df(pod_perf, customer_id, pod_id, ufm_config)
                all_forecasts.append(forecast_df)
                # logger.info(f"✅ Processed pod {pod_id} for customer {customer_id}.")

            arima_performance_df = pd.DataFrame([m.to_row() for m in arima_rows])
            arima_model_performances_dataframes.append(arima_performance_df)

            logger.info(f"✅ Forecast aggregation complete for {customer_id} complete.")

        # Combine across all customers
        arima_performance = pd.concat(arima_model_performances_dataframes).reset_index().drop(columns=['index'])
        forecast_combined_df = pd.concat(all_forecasts, ignore_index=True)
        run_forecast_sanity_checks(forecast_combined_df,arima_performance,consumption_types,model)
        # return arima_performance, forecast_combined_df
        write_to_spark(spark, forecast_combined_df, target_table_name)
        write_to_spark(spark, arima_performance, performance_metrics_table)
    except Exception as e:
        meta = get_error_metadata("ModelFitFailure", {"exception": str(e)})
        insert_profiling_error(
            log_id=None,
            error=meta["message"],
            traceback="",  # or traceback.format_exc()
            error_type="ModelFitFailure",
            severity=meta["severity"],
            component=meta["component"]
        )
        raise

def write_to_spark(spark, dataframe, target_table):
    try:
        spark.createDataFrame(dataframe).write.jdbc(url=write_url, table=target_table, mode="append",properties=write_properties)
        logger.info(f"✅ Successfully wrote {len(dataframe)} rows to x{target_table}")
    except Exception as e:
        logger.info(f"🚫 Failed to write to {target_table}")
        safe_exit("E0000", f"Failed to write to {target_table}")

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
    gap_handling: str = "skip"
) -> PodIDPerformanceData:
    """
    Refactored forecasting function handling in-sample, out-of-sample, and future forecasts.
    """
    data = []
    forecast_horizon = get_forecast_range(ufm_config)
    steps = len(forecast_horizon)

    for consumption_type in consumption_types:
        pod_df = df[df["PodID"] == pod_id].sort_index()
        series = prepare_time_series_data(pod_df, consumption_type)
        if series.isnull().all() or series.nunique() <= 1:
            meta = get_error_metadata("InvalidSeries", {"pod_id": pod_id, "consumption_type":consumption_type})
            insert_profiling_error(
                log_id=None,
                error=meta["message"],
                traceback="",  # or traceback.format_exc()
                error_type="InvalidSeries",
                severity=meta["severity"],
                component=meta["component"]
            )
            # logger.warning(f"⚠️ Invalid series for {consumption_type} @ Pod {pod_id}. Skipping.")
            forecast = pd.Series([0] * steps, index=forecast_horizon)
            data.append(_collect_metrics(pod_id, customer_id, consumption_type, forecast))
            continue

        if len(series) < 12:
            meta = get_error_metadata("SplitConfigurationError", {
                "series_length": len(series),
                "consumption_type": consumption_type
            })
            insert_profiling_error(
                log_id=None,
                error=meta["message"],
                traceback="",  # or traceback.format_exc()
                error_type="SplitConfigurationError",
                severity=meta["severity"],
                component=meta["component"]
            )
            forecast = pd.Series([0] * steps, index=forecast_horizon)
            data.append(_collect_metrics(pod_id, customer_id, consumption_type, forecast))
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
                insert_profiling_error(
                    log_id=None,
                    error=meta["message"],
                    traceback="",  # or traceback.format_exc()
                    error_type="ForecastGapTooLarge",
                    severity=meta["severity"],
                    component=meta["component"]
                )
                # logger.info(f"⛔ Forecast gap too large for {consumption_type} @ Pod {pod_id}. Skipping.")
                forecast = pd.Series([0] * steps, index=forecast_horizon)
                data.append(_collect_metrics(pod_id, customer_id, consumption_type, forecast))
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
                insert_profiling_error(
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
            # Evaluate on last portion of real data
            evaluation_window = min(steps, len(series))
            test_actual = series[-evaluation_window:]
            test_pred = model.predict(start=test_actual.index[0], end=test_actual.index[-1])
            if forecast_model.config.log:
                test_pred = np.exp(test_pred)
            metrics, baseline_metrics = evaluate_predictions(test_actual, test_pred)
            if gap_handling == "fill":
                future_forecast = future_forecast[future_forecast.index.isin(forecast_horizon)].copy()
                future_forecast = future_forecast.reindex(forecast_horizon).dropna()
            data.append(_collect_metrics(
                pod_id, customer_id, consumption_type,
                future_forecast, metrics, baseline_metrics
            ))
        except Exception as e:
            meta = get_error_metadata("ModelFitFailure", {"exception": str(e)})
            insert_profiling_error(log_id=None, error=meta["message"], traceback="",  error_type="ModelFitFailure",severity=meta["severity"], component=meta["component"])
            forecast = pd.Series([0] * steps, index=forecast_horizon)
            data.append(_collect_metrics(pod_id, customer_id, consumption_type, forecast))
            continue
    return PodIDPerformanceData(
        pod_id=pod_id,
        forecast_method_name=ufm_config.forecast_method_name,
        customer_id=customer_id,
        user_forecast_method_id=ufm_config.user_forecast_method_id,
        performance_data_frame=pd.DataFrame(data)
    )