from uuid import uuid4

from db.utilities import load_yaml_config


from profiler.timer.ProfilerTimer import get_logger, ProfilerTimer
from profiler.profiler_run import run_context


class NoOpProfiler:
    def __enter__(self):
        return self  # or None, as it does nothing

    def __exit__(self, exc_type, exc_value, traceback):
        pass  # does nothing, just safely exits the context


from contextlib import nullcontext

def conditional_timer(module, function, message="", category="general", context=None, spark_session=None):
    from db.utilities import logger, load_yaml_config
    config = load_yaml_config()
    profiling_cfg = config.get("profiling", {})
    enabled = profiling_cfg.get("enabled", False)
    logger.info(f"✅ Profiler enabled: {enabled}")

    if enabled:
        logger_backend = get_logger(
            engine=profiling_cfg.get("engine", "sqlserver"),
            spark_session=spark_session,  # explicitly pass SparkSession
            env=profiling_cfg.get("environment", "DEV")
        )
        return ProfilerTimer(
            module=module,
            function=function,
            logger_backend=logger_backend,
            message=message,
            category=category,
            run_id=run_context.run_id,
            app_name=run_context.app_name,
            context=context
        )
    else:
        return nullcontext()
