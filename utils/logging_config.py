# logging_config.py
import logging
from config.environment import is_databricks

def get_logging_logger(name: str, level=logging.WARNING) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        print("We are databricks")
        if is_databricks():
            logging.getLogger().setLevel(logging.WARNING)
            logger.setLevel(logging.WARNING)
            for noisy in ["py4j", "urllib3", "azure", "matplotlib", "asyncio", "pyspark"]:
                logging.getLogger(noisy).setLevel(logging.WARNING)

    return logger
