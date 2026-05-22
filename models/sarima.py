import logging

import pandas as pd

from models.algorithms.autoarima import forecast_arima_for_single_customer
from models.base import ForecastModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class SARIMAModel(ForecastModel):
    def train(self, spark) -> pd.DataFrame:
        # logger.info("🚀 [SARIMAModel] Starting training...")
        result = forecast_arima_for_single_customer(self, spark)
        # logger.info("✅ [SARIMAModel] Training complete.")
        return result