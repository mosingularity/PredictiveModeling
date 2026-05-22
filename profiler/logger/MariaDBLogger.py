# MariaDB Strategy
import pandas as pd


from db.utilities import with_db_connection
from profiler.logger.BaseLogger import BaseLogger


class MariaDBLogger(BaseLogger):
    def __init__(self, env: str = None):
        self.env = env

    @with_db_connection(engine="mariadb")
    def log(self, conn, record: dict) -> int:
        from db.queries import insert_profiling_log
        return insert_profiling_log(record)

    def log_error(self, log_id, error, traceback, error_type=None, severity=None, component=None):
        from db.queries import insert_profiling_error
        return insert_profiling_error(
            log_id=log_id,
            error=error,
            traceback=traceback,
            error_type=error_type,
            severity=severity,
            component=component
        )

# Spark Strategy
