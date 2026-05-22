import pandas as pd

from models.algorithms.autoarima import forecast_arima_for_single_customer
from models.base import ForecastModel
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ARIMAModel(ForecastModel):
    def train(self, spark) -> pd.DataFrame:
        """
        Train and forecast using the ARIMA/SARIMA model.

        Returns:
            pd.DataFrame: Aggregated forecast results.
        """
        logger.info("🚀 Starting ARIMA forecast training...")
        result = forecast_arima_for_single_customer(self, spark)
        logger.info("✅ ARIMA forecast training complete.")
        return result