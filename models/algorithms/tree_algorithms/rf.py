from typing import List

from db.error_logger import report_validation_error
from evaluation.performance import EntityPerformanceData, ForecastResults, PodIDPerformanceData, PredictionUnit
from models.algorithms.tree_algorithms.helper import forecast_pod_with_tree, train_rf
from models.algorithms.unbundled import run_unbundled
from models.base import ForecastModel
from validation.series import consumption_columns


def forecast_rf_unbundled(model: ForecastModel, spark) -> ForecastResults:
    return run_unbundled(model, spark, forecast_for_entity)


def forecast_for_pod_id(
    df,
    customer_id: str,
    pod_id: str,
    consumption_types: List[str],
    ufm_config,
    base_lags: List[int] = [1, 2, 3, 6],
    base_windows: List[int] = [3, 6],
    test_months=3,
) -> PodIDPerformanceData:
    """Forecast one pod with a Random Forest: engineer features, fit, backtest,
    and forecast the horizon, per consumption type."""
    return forecast_pod_with_tree(
        df, customer_id, pod_id, consumption_types, ufm_config,
        method_key="randomforest", train_fn=train_rf,
        report_validation_error=report_validation_error,
        base_lags=base_lags, base_windows=base_windows, test_months=test_months,
    )


def forecast_for_entity(
    unit: PredictionUnit,
    ufm_config,
    forecast_model: ForecastModel,
    base_lags: List[int] = [1, 2, 3, 6],
    base_windows: List[int] = [3, 6],
    test_months: int = 3,
) -> EntityPerformanceData:
    """Adapt forecast_for_pod_id to the unbundled path's PredictionUnit contract."""
    consumption_types = consumption_columns(unit.series)
    pod_perf = forecast_for_pod_id(
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