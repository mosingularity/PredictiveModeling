from pyspark.sql import SparkSession
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_spark_session = None

def get_spark_session(app_name="Energy Consumption Prediction"):
    global _spark_session
    if _spark_session is None:
        logger.info(f"🚀 Initializing new SparkSession for '{app_name}'")
        _spark_session = SparkSession.builder.appName(app_name).getOrCreate()
    else:
        logger.info("♻️ Reusing existing SparkSession.")
    return _spark_session