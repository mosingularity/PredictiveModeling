import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def invalid_length(series: pd.Series, consumption_type:str, period: int = 12):
    from profiler.errors.utils import get_error_metadata
    from db.error_logger import insert_profiling_error
    if len(series) < period:
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
        logger.info(meta["message"])
        return True
    else:
        return False


def invalid_series(pod_id: str,series: pd.Series, consumption_type:str, nunique: int = 1):
    from profiler.errors.utils import get_error_metadata
    from db.error_logger import insert_profiling_error

    if series.isnull().all() or series.nunique() <= nunique:
        meta = get_error_metadata("InvalidSeries", {"pod_id": pod_id, "consumption_type": consumption_type})
        insert_profiling_error(
            log_id=None,
            error=meta["message"],
            traceback="",  # or traceback.format_exc()
            error_type="InvalidSeries",
            severity=meta["severity"],
            component=meta["component"]
        )
        logger.warning(f"⚠️ Invalid series for {consumption_type} @ Pod {pod_id}. Skipping.")
        return True
    else:
        return False
def invalid_forecast_horizon(pod_id: str, series: pd.Series, consumption_type:str, forecast_horizon: pd.DatetimeIndex, gap_handling: str = 'skip'):
    from profiler.errors.utils import get_error_metadata
    from db.error_logger import insert_profiling_error
    last_date = series.index[-1]
    forecast_start, forecast_end = forecast_horizon[0], forecast_horizon[-1]

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
            logger.info(f"⛔ Forecast gap too large for {consumption_type} @ Pod {pod_id}. Skipping.")
            return True
    else:
        return False