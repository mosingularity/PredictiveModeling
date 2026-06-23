"""Shared driver for the bundled (customer/pod-keyed) forecasting path.

The per-model ``forecast_<model>_for_single_customer`` functions were ~95%
identical apart from the per-pod dispatch call (and ARIMA's up-front
hyperparameter extraction), so the loop lives here once. Each model passes a
``forecast_pod`` callback with the uniform signature
``(pod_df, customer_id, pod_id, consumption_types, ufm_config, model)``.
"""
import logging
from typing import Callable, List

import pandas as pd

from db.error_logger import report_validation_error
from db.utilities import jdbc_write
from evaluation.performance import CustomerPerformanceData, ModelPodPerformance
from models.algorithms.helper import (
    _convert_forecast_map_to_df,
    _convert_to_model_performance_row,
    _get_customer_data,
    ensure_numeric_consumption_types,
)
from models.base import ForecastModel
from validation.forecast import run_forecast_sanity_checks
from validation.metadata import get_error_metadata
from utils.exit_handler import safe_exit

logger = logging.getLogger(__name__)

performance_metrics_table = "dbo.StatisticalPerformanceMetrics"
target_table_name = "dbo.ForecastFact"


def run_bundled(model: ForecastModel, spark, forecast_pod: Callable):
    """Bundled customer→pod forecasting loop shared by ARIMA/SARIMA/RF/XGB.

    ``forecast_pod(pod_df, customer_id, pod_id, consumption_types, ufm_config, model)``
    returns a ``PodIDPerformanceData``; ARIMA's adapter wraps its own
    order/seasonal_order extraction inside the callback.
    """
    try:
        forecast_method_id = getattr(model.dataset.ufm_config, "forecast_method_id", None)
        if model.dataset is None or model.dataset.processed_df is None:
            meta = get_error_metadata("ModelConfigMissing", {"field": "dataset.df"})
            report_validation_error(log_id=None, error=meta["message"], traceback="",
                                    error_type="ModelConfigMissing",
                                    severity=meta["severity"], component=meta["component"])
            safe_exit(meta["code"], meta["message"])

        if model.dataset.processed_df.empty:
            meta = get_error_metadata("EmptySeries", {"forecast_method_id": forecast_method_id})
            report_validation_error(log_id=None, error=meta["message"], traceback="",
                                    error_type="EmptySeries",
                                    severity=meta["severity"], component=meta["component"])
            safe_exit(meta["code"], meta["message"])

        df = model.dataset.processed_df
        df = ensure_numeric_consumption_types(df, model)
        unique_customers, _ = model.dataset.extract_unique_customers_and_pods()
        ufm_config = model.dataset.ufm_config
        consumption_types = getattr(model.dataset, "variable_ids", None) or model.config.consumption_types

        all_forecasts = []
        model_performances_dataframes: List[pd.DataFrame] = []
        for customer_id in unique_customers:
            customer_data = _get_customer_data(df, customer_id)
            if customer_data.empty:
                logger.warning(f"🚫 No data found for customer {customer_id}, skipping.")
                continue

            consumer_perf = CustomerPerformanceData(customer_id=customer_id, columns=consumption_types)
            perf_rows: List[ModelPodPerformance] = []

            for pod_id in customer_data["PodID"].unique().tolist():
                pod_df = customer_data[customer_data["PodID"] == pod_id].sort_values("ReportingMonth")
                pod_perf = forecast_pod(pod_df, customer_id, pod_id, consumption_types, ufm_config, model)
                consumer_perf.pod_by_id_performance.append(pod_perf)
                perf_rows.append(_convert_to_model_performance_row(pod_perf, customer_id, pod_id, ufm_config))
                all_forecasts.append(_convert_forecast_map_to_df(pod_perf, customer_id, pod_id, ufm_config))

            model_performances_dataframes.append(pd.DataFrame([m.to_row() for m in perf_rows]))

        logger.info(f"📊 Processed {len(unique_customers)} customer(s); "
                    f"{len(all_forecasts)} pod-forecast(s) produced.")
        performance = pd.concat(model_performances_dataframes).reset_index().drop(columns=["index"])
        forecast_combined_df = pd.concat(all_forecasts, ignore_index=True)
        run_forecast_sanity_checks(forecast_combined_df, performance, consumption_types, model)
        jdbc_write(spark, forecast_combined_df, target_table_name)
        jdbc_write(spark, performance, performance_metrics_table)
    except Exception as z:
        meta = get_error_metadata("ModelFitFailure", {"exception": str(z)})
        report_validation_error(log_id=None, error=meta["message"], traceback="",
                                error_type="ModelFitFailure",
                                severity=meta["severity"], component=meta["component"])
        raise
