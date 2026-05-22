from typing import List

import numpy as np
import pandas as pd

from db.error_logger import insert_profiling_error
from models.base import ForecastModel
from profiler.errors.utils import get_error_metadata

import logging

from utils.exit_handler import safe_exit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

performance_columns = [
    "RMSE_Avg", "R2_Avg",
    "RMSE_OffPeakConsumption", "R2_OffPeakConsumption",
    "RMSE_PeakConsumption", "R2_PeakConsumption",
    "RMSE_StandardConsumption", "R2_StandardConsumption",
    "RMSE_Block1Consumption", "R2_Block1Consumption",
    "RMSE_Block2Consumption", "R2_Block2Consumption",
    "RMSE_Block3Consumption", "R2_Block3Consumption",
    "RMSE_Block4Consumption", "R2_Block4Consumption",
    "RMSE_NonTOUConsumption", "R2_NonTOUConsumption"
]

def validate_nonzero_forecast(df: pd.DataFrame, consumption_types: List[str]) -> bool:
    for col in consumption_types:
        if col in df.columns and df[col].abs().sum() > 0:
            return True  # ✅ At least one useful forecast
    return False  # ❌ All zero


def validate_nonzero_metrics(perf_df: pd.DataFrame, metric_columns: List[str]) -> bool:
    for col in metric_columns:
        if col in perf_df.columns:
            col_sum = perf_df[col].replace([np.nan, None], 0).abs().sum()
            if col_sum > 0:
                return True  # ✅ Valid metric found
    return False  # ❌ All metrics invalid or zero

def run_forecast_sanity_checks(
    forecast_df: pd.DataFrame,
    performance_df: pd.DataFrame,
    consumption_types: List[str],
    model: ForecastModel
):
    ufm_config = model.dataset.ufm_config
    unique_customers = model.dataset.unique_customers or ["unknown"]

    # Forecast Output Check
    if not validate_nonzero_forecast(forecast_df, consumption_types):
        meta = get_error_metadata("AllZeroForecast", {"databrick_id": ufm_config.databrick_task_id,"ufmid": ufm_config.forecast_method_id})
        insert_profiling_error(
            log_id=None,
            error=meta["message"],
            traceback="",  # or traceback.format_exc()
            error_type="AllZeroForecast",
            severity=meta["severity"],
            component=meta["component"]
        )
        logger.info("🚫 All forecasted values are zero. Aborting model output due to invalid forecasts.")
        safe_exit(meta["code"], meta["message"])

    # Performance Metric Check
    if not validate_nonzero_metrics(performance_df, performance_columns):
        meta = get_error_metadata("AllZeroMetrics", {
            "task_id": ufm_config.databrick_task_id,
            "customer_ids": unique_customers,
            "metric_columns": performance_columns
        })
        insert_profiling_error(
            log_id=None,
            error=meta["message"],
            traceback="",  # or traceback.format_exc()
            error_type="AllZeroMetrics",
            severity=meta["severity"],
            component=meta["component"]
        )
        logger.info("🚫 All performance metrics are zero or invalid. Aborting due to failed model evaluation.")
        safe_exit(meta["code"], meta["message"])
