from config_loader import HyperParameterConfig
from data.dataset import ForecastDataset


class ForecastModel:
    """Carries a run's dataset and config. The forecasting paths (run_bundled,
    run_unbundled) take this and fetch their own entity data; they don't read
    ``dataset.processed_df``, so it's only validated when present, not required."""

    def __init__(self, dataset: ForecastDataset, config: HyperParameterConfig):
        self.dataset = dataset
        self.config = config
        if self.dataset.processed_df is not None and self.dataset.processed_df.empty:
            raise ValueError("Dataset must be loaded and preprocessed before initializing the model.")
