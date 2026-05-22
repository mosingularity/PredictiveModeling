# base.py
from config_loader import HyperParameterConfig
from data.dataset import ForecastDataset


class ForecastModel:
    def __init__(self, dataset: ForecastDataset, config: HyperParameterConfig):

        self.dataset = dataset
        self.config = config
        # Ensure that the dataset is loaded and preprocessed.
        if self.dataset.processed_df.empty:
            raise ValueError("Dataset must be loaded and preprocessed before initializing the model.")

    def prepare_data(self):
        # Optional shared data preparation logic (e.g., filtering customer-specific data)
        # This probably where we will perform fine-tuning
        pass

    def train(self, spark):
        raise NotImplementedError("Subclasses must implement the train() method.")

    # Do we need a fine_tune(self): and other methods that leverage Data Science & Machine Learning Principles

    def evaluate(self):
        raise NotImplementedError("Subclasses must implement the evaluate() method.")

    def predict(self):
        raise NotImplementedError("Subclasses must implement the predict() method.")
