from data.dml import *
import pandas as pd

from db.error_logger import report_validation_error
from db.utilities import jdbc_write
from evaluation.performance import *
from hyperparameters import get_model_hyperparameters
from models.algorithms.helper import _convert_to_model_performance_row, \
    _convert_forecast_map_to_df, _get_customer_data, _collect_metrics, ensure_numeric_consumption_types
from models.algorithms.tree_algorithms.helper import engineer_data, split_train_test, plot_train_test, \
    recursive_forecast, plot_forecast, train_xgb
from models.algorithms.utilities import  evaluate_predictions, process_reporting_months
from models.algorithms._unbundled import run_unbundled
from models.base import ForecastModel
from models.forecast_validation import run_forecast_sanity_checks
from models.series_validator import validate_series, consumption_columns
from profiler.errors.validation import invalid_length, invalid_series, invalid_forecast_horizon

# Setup logger with basic configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

performance_metrics_table = "dbo.StatisticalPerformanceMetrics"
target_table_name = "dbo.ForecastFact"

def forecast_xgb_unbundled(model: ForecastModel, spark) -> UnbundledResults:
    return run_unbundled(model, spark, forecast_for_entity)


    

def forecast_xgb_for_single_customer(model: ForecastModel, spark):
    """
    Function: Forecast XGBoost models for a single customer using information embedded in the model instance.

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

    Returns:
        pd.DataFrame: Aggregated forecasting results in long format.
    """
    try:
        forecast_method_id = getattr(model.dataset.ufm_config, "forecast_method_id", None)
        # Step 1: Validate dataset
        if model.dataset is None or model.dataset.processed_df is None:
            meta = get_error_metadata("ModelConfigMissing", {"field": "dataset.df"})
            report_validation_error(
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
            report_validation_error(
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

        all_forecasts = []
        xgb_model_performances_dataframes: List[pd.DataFrame] = []
        for customer_id in unique_customers:
            customer_data = _get_customer_data(df, customer_id)
            if customer_data.empty:
                logger.warning(f"🚫 No data found for customer {customer_id}, skipping.")
                continue

            consumer_perf = CustomerPerformanceData(customer_id=customer_id, columns=consumption_types)
            xgb_rows: List[ModelPodPerformance] = []

            unique_pod_ids = customer_data['PodID'].unique().tolist()
            for pod_id in unique_pod_ids:
                pod_df = customer_data[customer_data["PodID"] == pod_id].sort_values('ReportingMonth')
                # logger.info(f"🚀 Forecasting Customer {customer_id}, Pod {pod_id}")
                pod_perf = forecast_for_podel_id(pod_df, customer_id, pod_id, consumption_types, ufm_config)
                consumer_perf.pod_by_id_performance.append(pod_perf)

                # --- Performance Dataclass (Modularized) ---
                mpp = _convert_to_model_performance_row(pod_perf, customer_id, pod_id, ufm_config)
                xgb_rows.append(mpp)

                # --- Forecast DataFrame (Pandas) ---
                forecast_df = _convert_forecast_map_to_df(pod_perf, customer_id, pod_id, ufm_config)
                all_forecasts.append(forecast_df)
                # logger.info(f"✅ Processed pod {pod_id} for customer {customer_id}.")

            xgb_performance_df = pd.DataFrame([m.to_row() for m in xgb_rows])
            xgb_model_performances_dataframes.append(xgb_performance_df)

            # logger.info(f"✅ Forecast aggregation complete for {customer_id} complete.")

        # Combine across all customers
        xgb_performance = pd.concat(xgb_model_performances_dataframes).reset_index().drop(columns=['index'])
        forecast_combined_df = pd.concat(all_forecasts, ignore_index=True)
        run_forecast_sanity_checks(forecast_combined_df,xgb_performance,consumption_types,model)
        # return xgb_performance, forecast_combined_df
        jdbc_write(spark, forecast_combined_df, target_table_name)
        jdbc_write(spark, xgb_performance, performance_metrics_table)
    except Exception as z:
        meta = get_error_metadata("ModelFitFailure", {"exception": str(z)})
        report_validation_error(
            log_id=None,
            error=meta["message"],
            traceback="",  # or traceback.format_exc()
            error_type="ModelFitFailure",
            severity=meta["severity"],
            component=meta["component"]
        )
        raise

def forecast_for_podel_id(
    df: pd.DataFrame,
    customer_id: str,
    pod_id: str,
    consumption_types: List[str],
    ufm_config,
    base_lags: List[int] = [1, 2, 3, 6],
    base_windows: List[int] = [3, 6],
    test_months=3,
) -> PodIDPerformanceData:
    """
    Refactored forecasting function handling in-sample, out-of-sample, and future forecasts.
    """
    data = []
    forecast_horizon = get_forecast_range(ufm_config)
    start_fc = ufm_config.start_date
    end_fc = ufm_config.end_date

    for consumption_type in consumption_types:
        pod_df = df[df["PodID"] == pod_id].sort_index()
        pod_df = process_reporting_months(pod_df)

        if invalid_series(pod_id, pod_df[consumption_type], consumption_type):
            forecast = pd.Series([0] * len(forecast_horizon), index=forecast_horizon)
            data.append(_collect_metrics(pod_id, customer_id, consumption_type, forecast))
            continue

        if invalid_length(pod_df[consumption_type], consumption_type):
            forecast = pd.Series([0] * len(forecast_horizon), index=forecast_horizon)
            data.append(_collect_metrics(pod_id, customer_id, consumption_type, forecast))
            continue

        if invalid_forecast_horizon(pod_id, pod_df[consumption_type], consumption_type, forecast_horizon):
            forecast = pd.Series([0] * len(forecast_horizon), index=forecast_horizon)
            data.append(_collect_metrics(pod_id, customer_id, consumption_type, forecast))
            continue

        forecast_horizon_months = ((end_fc.year - pod_df.index.max().year) * 12 +
                                   (end_fc.month - pod_df.index.max().month))
        year_lags = [12 * i for i in range(1, forecast_horizon_months // 12 + 1)]
        lags = base_lags + year_lags

        year_windows = [12 * i for i in range(1, forecast_horizon_months // 12 + 1)]
        windows = base_windows + year_windows

        pod_df, feature_cols, stl_obj, history_series = engineer_data(pod_df, consumption_type, lags, windows)

        train_df, test_df = split_train_test(pod_df, test_months)
        if train_df.empty or test_df.empty:
            logger.warning(
                f"🚫 Not enough data after split for {consumption_type} @ Pod {pod_id}. Skipping."
            )
            forecast = pd.Series([0] * len(forecast_horizon), index=forecast_horizon)
            data.append(_collect_metrics(pod_id, customer_id, consumption_type, forecast))
            continue
        X_train, y_train = train_df[feature_cols], train_df["deseasoned"]
        X_test, y_test = test_df[feature_cols], test_df["deseasoned"]

        xgb_params_tuple = get_model_hyperparameters("xgboost", ufm_config.model_parameters)
        # logger.info(f"💡 Extracted hyperparameters={xgb_params_tuple}")

        try:
            xgb_model = train_xgb(X_train, y_train, xgb_params_tuple)
        except Exception as model_fit_exception:
            meta = get_error_metadata("ModelFitFailure", {"exception": str(model_fit_exception)})
            report_validation_error(log_id=None, error=meta["message"], traceback="", error_type="ModelFitFailure",
                                   severity=meta["severity"], component=meta["component"])
            forecast = pd.Series([0] * len(forecast_horizon), index=forecast_horizon)
            data.append(_collect_metrics(pod_id, customer_id, consumption_type, forecast))
            continue

        train_pred_ds = xgb_model.predict(X_train)
        test_pred_ds = xgb_model.predict(X_test)

        # re-add seasonality
        train_pred = train_pred_ds + train_df["seasonal"]
        test_pred = test_pred_ds + test_df["seasonal"]

        # plot_train_test(train_df, test_df, consumption_type, train_pred, test_pred)

        metrics, baseline_metrics = evaluate_predictions(y_test, test_pred)

        # Forecast from a model refit on ALL data (train+test) so the future uses
        # the most recent months; the train-only xgb_model above is kept solely for
        # the held-out backtest metrics. Mirrors the ARIMA full-data forecast.
        xgb_full = train_xgb(pod_df[feature_cols], pod_df["deseasoned"], xgb_params_tuple)
        fc_df = recursive_forecast(history_series, stl_obj, xgb_full,
                                   feature_cols, start_fc, end_fc,
                                   lags, windows)
        future_forecast = fc_df['forecast']
        # Full-history fitted line (train fit + held-out test) for the solid
        # "predicted historical" overlay; metrics stay on the held-out test above.
        in_sample_fit = pd.concat([train_pred, test_pred]).sort_index()
        data.append(_collect_metrics(
            pod_id, customer_id, consumption_type,
            future_forecast, metrics, baseline_metrics, in_sample=in_sample_fit
        ))
        # plot_forecast(pod_df, fc_df, consumption_type, end_fc)
    return PodIDPerformanceData(
        pod_id=pod_id,
        forecast_method_name=ufm_config.forecast_method_name,
        customer_id=customer_id,
        user_forecast_method_id=ufm_config.user_forecast_method_id,
        performance_data_frame=pd.DataFrame(data)
    )


def forecast_for_entity(
    unit: PredictionUnit,
    ufm_config,
    forecast_model: ForecastModel,
    base_lags: List[int] = [1, 2, 3, 6],
    base_windows: List[int] = [3, 6],
    test_months: int = 3,
) -> EntityPerformanceData:
    """
    Entity-aware wrapper around forecast_for_podel_id for the unbundled path.
    Accepts a PredictionUnit and returns EntityPerformanceData.
    """
    consumption_types = consumption_columns(unit.series)
    pod_perf = forecast_for_podel_id(
        df=unit.series,
        customer_id=unit.customer_id,
        pod_id=unit.entity_id,
        consumption_types=consumption_types,
        ufm_config=ufm_config,
        base_lags=base_lags,
        base_windows=base_windows,
        test_months=test_months,
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