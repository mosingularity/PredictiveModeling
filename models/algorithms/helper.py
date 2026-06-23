import logging

import numpy as np
import pandas as pd

from evaluation.performance import ModelPodPerformance, build_forecast_df
from models.base import ForecastModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _get_customer_data(df: pd.DataFrame, customer_id: str) -> pd.DataFrame:
    filtered = df[df['CustomerID'] == customer_id].sort_values('PodID')
    # logging.info(f"📊 Found {len(filtered)} rows for CustomerID={customer_id}")
    return filtered

def _convert_to_model_performance_row(pod_perf, customer_id, pod_id, config):
    df_long = pod_perf.performance_data_frame
    metrics = {
        row.consumption_type: {"RMSE": row.RMSE, "R2": row.R2}
        for row in df_long.itertuples(index=False)
    }
    return ModelPodPerformance(
        ModelName=config.forecast_method_name,
        CustomerID=customer_id,
        PodID=pod_id,
        DataBrickID=getattr(config, "databrick_task_id", None),
        UserForecastMethodID=config.user_forecast_method_id,
        metrics=metrics
    )

def _convert_forecast_map_to_df(pod_perf, customer_id, pod_id, config):
    df_long = pod_perf.performance_data_frame
    forecast_map = df_long.set_index("consumption_type")["forecast"].to_dict()
    return build_forecast_df(
        forecast_map, customer_id, pod_id, list(forecast_map.keys()), config.user_forecast_method_id
    )

def _apply_log(series: pd.Series, log: bool) -> pd.Series:
    """
    Optionally apply a log transformation to the series.
    Adds a small constant for numerical stability.
    """
    if log:
        return np.log(series + 1e-6)
    return series

def ensure_numeric_consumption_types(df_raw: pd.DataFrame, model: ForecastModel) -> pd.DataFrame:
    """
    For each column in model.config.consumption_types:
      - If it exists in df_raw and is not already numeric, coerce it to numeric.
      - Log each conversion via logger.info.
    """
    for col in model.config.consumption_types:
        if col not in df_raw.columns:
            # logger.info(f"⚠️Column '{col}' not found in DataFrame; skipping.")
            continue

        if not pd.api.types.is_numeric_dtype(df_raw[col]):
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
            # logger.info(f"💡 Converted column '{col}' to numeric dtype.")
        else:
            # logger.info(f"⚠️ Column '{col}' already numeric; no conversion needed.")
            pass
    return df_raw


def _collect_metrics(pod_id, customer_id, consumption_type, forecast, metrics=None, baseline_metrics=None, in_sample=None):
    # ``in_sample`` is the model's in-sample/backtest prediction Series (indexed by
    # historical ReportingMonth) that the metrics were scored on. It is computed
    # anyway during evaluation; carrying it lets the results layer draw each
    # model's fitted-history line. Additive — None for the skip/zero paths.
    row = {'pod_id': pod_id, 'customer_id': customer_id, 'consumption_type': consumption_type,
           'forecast': forecast, 'in_sample': in_sample}
    # A skipped/failed fit carries no metrics — record NaN, not 0.0. A fake 0.0
    # reads as a perfect score and silently deflates any RMSE_Avg it's mixed into;
    # NaN is the honest "unscored" signal the results layer keys off.
    for metric in ['RMSE', 'MAE', 'R2']:
        row[metric] = (metrics or {}).get(metric, float("nan"))
        row[f'{metric}_baseline'] = (baseline_metrics or {}).get(metric, float("nan"))
    return row