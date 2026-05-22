# Profiler context manager
import threading
import traceback
from datetime import timedelta,datetime
import time
from functools import wraps
import socket
from db.utilities import logger as global_logger
from docstring.validate_category import category_validator
from profiler.logger.SparkLogger import SparkLogger


class ProfilerTimer:
    def __init__(
        self,
        module,
        function,
        logger_backend,
        message="",
        category="general",
        run_id=None,
        app_name=None,
        context=None
    ):
        self.module = module
        self.function = function
        self.message = message
        self.logger = logger_backend
        self.hostname = socket.gethostname()
        self.thread_id = threading.get_ident()
        self.category = category
        if not category_validator.validate(category):
            self.category = "unknown"
        self.run_id = run_id
        self.app_name = app_name
        self.log_id = None  # ✅ Will store log insert ID for FK to errors
        self.context = context or {}

    def __enter__(self):
        self.start_time = time.time()
        global_logger.info(f"🚀 Entering profiler timer for {self.module}.{self.function}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global_logger.info(f"🏁 Exiting profiler timer for {self.module}.{self.function}")
        end_time = time.time()
        duration_ms = round((end_time - self.start_time) * 1000, 3)
        duration_readable = str(timedelta(milliseconds=duration_ms)).split(".")[0]

        self.status = "failed" if exc_type else "completed"
        error_msg = str(exc_val) if exc_type else None
        trace = traceback.format_exc() if exc_type else None

        # Convert to schema-aligned dict
        record = {
            "Module": self.module,
            "Function": self.function,
            "Message": self.message,
            "StartTime": datetime.fromtimestamp(self.start_time),
            "EndTime": datetime.fromtimestamp(end_time),
            "DurationMS": duration_ms,
            "DurationText": duration_readable,
            "Hostname": self.hostname,
            "ThreadID": self.thread_id,
            "Status": self.status,
            "RunID": self.run_id,
            "AppName": self.app_name,
            "Category": self.category,
            "ForecastMethodID": self.context.get("forecast_method_id"),
            "ForecastMethodName": self.context.get("forecast_method_name"),
            "DatabricksTaskID": self.context.get("databrick_task_id"),
            "UserForecastMethodID": self.context.get("user_forecast_method_id"),
            "Error": error_msg,
            "Traceback": trace
        }

        global_logger.info(f"🔍 Profiling record: {record}")

        try:
            # ✅ Log to main table
            self.log_id = self.logger.log(record)  # should return log_id for relational backends
            from db.error_logger import insert_profiling_error
            # ✅ If failed, also write to profiling_errors
            if self.status == "failed" and self.log_id:
                insert_profiling_error(
                    log_id=self.log_id,
                    error=error_msg,
                    traceback=trace,
                    error_type=type(exc_val).__name__ if exc_val else "UnknownError",
                    severity="high",
                    component=self.function
                )

        except Exception as logging_error:
            global_logger.warning(f"❌ Logging failed in ProfilerTimer for {self.function}: {logging_error}")
            raise
        global_logger.info(f"🏁 Exiting profiler timer for {self.module}.{self.function}")

    @staticmethod
    def timer(logger_backend, module, function, message=""):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with ProfilerTimer(module, function, logger_backend, message):
                    return func(*args, **kwargs)
            return wrapper
        return decorator


# logger Factory
def get_logger(engine="sqlserver", spark_session=None, env=None):
    from db.utilities import logger
    if engine == "sqlserver":
        from profiler.logger.SQLServerLogger import SQLServerLogger
        if not spark_session:
            logger.error("❌ Spark session is required for SQLServerLogger but was None.")
            raise ValueError("SparkSession required for SQLServerLogger")
        return SQLServerLogger(spark_session)
    elif engine == "pyspark":
        if spark_session is None:
            raise ValueError("Spark session required for SparkLogger")
        return SparkLogger(spark_session)
    else:
        raise ValueError(f"Unsupported logging engine: {engine}")