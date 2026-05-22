import os
import logging
from dotenv import load_dotenv
from config.environment import is_databricks

logger = logging.getLogger(__name__)
load_dotenv()


def get_parameter(name: str, default=None):
    """
    Safely retrieve runtime parameter across both local and Databricks environments.
    Priority:
    1. dbutils.widgets (Databricks)
    2. .env (Local)
    3. Default fallback
    """
    if is_databricks():
        try:
            from pyspark.dbutils import DBUtils  # This import may fail locally
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.getOrCreate()
            dbutils = DBUtils(spark)
            value = dbutils.widgets.get(name)
            logger.info(f"🧪 Resolved {name} using Databricks widgets: {value}")
            return value
        except Exception as e:
            logger.warning(f"⚠️ dbutils unavailable or widget '{name}' not found. Reason: {e}")

    value = os.environ.get(name, default)
    if value is not None:
        logger.info(f"🌍 Resolved {name} from .env or fallback: {value}")
    else:
        logger.warning(f"⚠️ {name} not found in any config source. Using default: {default}")
    return value
