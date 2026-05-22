from abc import ABC, abstractmethod
from pyspark.sql import SparkSession
from typing import Dict

class BaseLogger(ABC):
    def __init__(self, spark: SparkSession, config_path="config.yaml"):
        self.spark = spark
        self.config_path = config_path

    @abstractmethod
    def log(self, record: Dict):
        pass

    @abstractmethod
    def log_error(self, error_info: Dict):
        pass

    def flush(self):
        pass  # For compatibility with Spark
