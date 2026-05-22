# utils/exit_handler.py
import logging
from config.environment import is_databricks
from utils.dbutils_singleton import get_dbutils

logger = logging.getLogger(__name__)

def safe_exit(code: str, message: str):
    full_msg = f"{message}"
    logger.info(f"✅ Safely exiting the notebook with {full_msg}.")

    if is_databricks():
        dbutils = get_dbutils()
        if dbutils is not None:
            try:
                dbutils.notebook.exit(f"[Notebook Exit] {full_msg}")
            except Exception as e:
                logger.warning(f"⚠️ dbutils.notebook.exit failed: {e}")
                pass

    # Fallback for non-Notebook contexts or missing dbutils
    raise SystemExit(f"[Local Exit] {full_msg}")
