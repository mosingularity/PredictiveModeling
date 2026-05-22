import logging
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, TimestampType, FloatType, LongType
from profiler.logger.BaseLogger import BaseLogger
from db.utilities import get_jdbc_options

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class SQLServerLogger(BaseLogger):
    def __init__(self, spark: SparkSession, config_path="config.yaml"):
        super().__init__(spark, config_path)
        if self.spark is None:
            logger.error("❌ SparkSession is None during SQLServerLogger initialization.")
            raise ValueError("SQLServerLogger requires a valid SparkSession instance.")
        self.url, self.user, self.password = get_jdbc_options(config_path)
        logger.info(f"✅ Initialized SQLServerLogger with JDBC URL: {self.url}")

    def log(self, record: dict):
        logger.info(f"Attempting to log profiling data: {record}")
        schema = StructType([
            StructField("Module", StringType(), True),
            StructField("Function", StringType(), True),
            StructField("Message", StringType(), True),
            StructField("StartTime", TimestampType(), True),
            StructField("EndTime", TimestampType(), True),
            StructField("DurationMS", FloatType(), True),
            StructField("DurationText", StringType(), True),
            StructField("Hostname", StringType(), True),
            StructField("ThreadID", LongType(), True),
            StructField("Status", StringType(), True),
            StructField("RunID", StringType(), True),
            StructField("AppName", StringType(), True),
            StructField("Category", StringType(), True),
            StructField("ForecastMethodID", IntegerType(), True),
            StructField("ForecastMethodName", StringType(), True),
            StructField("DatabricksTaskID", IntegerType(), True),
            StructField("UserForecastMethodID", IntegerType(), True),
            StructField("Error", StringType(), True),
            StructField("Traceback", StringType(), True)
        ])

        if self.spark is None:
            logger.error("❌ SparkSession is None in log method.")
            raise ValueError("SparkSession is required but is None.")

        try:
            df = self.spark.createDataFrame([record], schema)
            logger.info(f"✅ Created Spark DataFrame:\n{df.show(truncate=False)}")
            df.write.format("jdbc") \
                .option("url", self.url) \
                .option("dbtable", "dbo.PredictiveProfilingLogs") \
                .option("user", self.user) \
                .option("password", self.password) \
                .mode("append") \
                .save()
            logger.info("✅ Profiling log inserted successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to insert profiling log: {e}")
            raise

    def log_error(self, error_info: dict):
        logger.info(f"Attempting to log profiling error: {error_info}")
        schema = StructType([
            StructField("ProfilingLogID", IntegerType(), True),
            StructField("Traceback", StringType(), True),
            StructField("Severity", StringType(), True),
            StructField("ErrorType", StringType(), True),
            StructField("Component", StringType(), True),
            StructField("Error", StringType(), True)
        ])

        if self.spark is None:
            logger.error("❌ SparkSession is None in log_error method.")
            raise ValueError("SparkSession is required but is None.")

        try:
            df = self.spark.createDataFrame([error_info], schema)
            logger.info(f"✅ Created Spark DataFrame for error:\n{df.show(truncate=False)}")
            
            df.write.format("jdbc") \
                .option("url", self.url) \
                .option("dbtable", "dbo.PredictiveProfilingErrors") \
                .option("user", self.user) \
                .option("password", self.password) \
                .mode("append") \
                .save()
            logger.info("✅ Profiling error logged successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to log profiling error: {e}")
            raise
