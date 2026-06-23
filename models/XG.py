import logging

import pandas as pd

from models.algorithms.tree_algorithms.xgb import forecast_xgb_for_single_customer
from models.base import ForecastModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class XGBoostModel(ForecastModel):
    def train(self, spark) -> pd.DataFrame:
        logger.info("🚀 [XGBoostModel] Starting training...")
        result = forecast_xgb_for_single_customer(self, spark)
        logger.info("✅ [XGBoostModel] Training complete.")
        return result