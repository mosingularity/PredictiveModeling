# logger_factory.py
from config.config_service import get_profiling_engine, get_active_env

def get_logger():
    engine = get_profiling_engine()
    env = get_active_env()

    if engine == "mariadb":
        from profiler.logger.MariaDBLogger import MariaDBLogger
        return MariaDBLogger(env=env)
    elif engine == "sqlserver":
        from profiler.logger.SQLServerLogger import SQLServerLogger
        return SQLServerLogger(env=env)
    else:
        raise ValueError(f"Unsupported logging engine: {engine}")
