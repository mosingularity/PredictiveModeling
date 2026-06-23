import logging

from config_loader import HyperParameterConfig
from data.dataset import ForecastDataset
from models.base import ForecastModel
from models.algorithms.autoarima import forecast_arima_for_single_customer
from models.algorithms.tree_algorithms.rf import forecast_rf_for_single_customer
from models.algorithms.tree_algorithms.xgb import forecast_xgb_for_single_customer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j.clientserver").setLevel(logging.WARNING)

# forecast_method_name (lowercased, spaces stripped) -> bundled entry point.
# SARIMA shares the ARIMA path (the per-pod hyperparameters carry the seasonal order).
BUNDLED_FORECASTERS = {
    "arima": forecast_arima_for_single_customer,
    "sarima": forecast_arima_for_single_customer,
    "randomforest": forecast_rf_for_single_customer,
    "xgboost": forecast_xgb_for_single_customer,
}


class ForecastPipeline:
    def __init__(self, dataset: ForecastDataset, config: HyperParameterConfig):
        name = dataset.ufm_config.forecast_method_name.lower().replace(" ", "")
        logging.info(f"Model name: {name}")
        if name not in BUNDLED_FORECASTERS:
            raise ValueError(f"Unknown model type: {name}")
        self.forecast_fn = BUNDLED_FORECASTERS[name]
        self.model = ForecastModel(dataset, config)

    def run(self, spark):
        self.model.prepare_data()
        return self.forecast_fn(self.model, spark)
