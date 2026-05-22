import traceback
from profiler.errors.utils import get_error_metadata
from db.error_logger import insert_profiling_error
from utils.exit_handler import safe_exit


def databricks_safe(error_key: str = "UnknownError", context: dict = None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                full_context = {**(context or {}), "exception": str(e)}
                metadata = get_error_metadata(error_key, full_context)

                insert_profiling_error(
                    log_id=None,
                    error=metadata["message"],
                    traceback=traceback.format_exc(),
                    error_type=error_key,
                    severity=metadata["severity"],
                    component=metadata["component"]
                )

                safe_exit(metadata["code"], metadata["message"])  # ✅ USE HERE
                raise  # optional re-raise
        return wrapper
    return decorator