import pandas as pd

from profiler.logger.BaseLogger import BaseLogger


class SparkLogger(BaseLogger):
    def __init__(self, spark_session, view_name: str = "profiling_logs"):
        self.spark = spark_session
        self.view_name = view_name
        self.buffer = []

    def log(self, record: dict):
        self.buffer.append(record)

    def flush(self):
        if self.buffer:
            df = pd.DataFrame(self.buffer)
            spark_df = self.spark.createDataFrame(df)
            spark_df.createOrReplaceTempView(self.view_name)
            self.buffer.clear()