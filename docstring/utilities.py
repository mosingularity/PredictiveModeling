# profiler_utils.py

import functools
import re
import traceback
from contextlib import nullcontext


from docstring.validate_category import category_validator
from profiler.profiler_context_builder import profiler_context_builder
from profiler.profiler_utils import conditional_timer

from db.spark_session import get_spark_session




def profiled_function(
    category: str,
    enabled: bool = True,
    message_template: str = "{message}"
):
    """
    Decorator to conditionally profile a method/function.

    Args:
        category (str): e.g. 'dataset', 'model'
        enabled (bool): Toggle profiling
        message_template (str): Supports:
            - {message}: First line of docstring with "Function:"
            - {task_id}: Extracted from self.databrick_task_id, if available
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            spark = get_spark_session()
            from db.utilities import logger, load_yaml_config
            config = load_yaml_config()
            profiling_cfg = config.get("profiling", {})
            enabled = profiling_cfg.get("enabled", False)
            logger.info(f"✅ Docstring utilities: profiled_function() - The Profiler is enabled: {enabled}")
            category_validator.validate(category)
            instance = args[0] if args else None
            context = profiler_context_builder(
                instance=instance,
                func=func,
                args=args,
                kwargs=kwargs,
                nested_keys=[
                    "model.dataset.ufm_config.forecast_method_name",
                    "model.dataset.ufm_config.forecast_method_id",
                    "model.dataset.ufm_config.databrick_task_id",
                    "model.dataset.ufm_config.user_forecast_method_id"
                ]
            )
            try:
                message = message_template.format(**context)
            except Exception as e:
                message = f"[Profiler] Failed to format message for {func.__name__}"
            module = f"{func.__module__.replace('.','/')}.py"
            timer = conditional_timer(
                module=module,
                function=func.__name__,
                message=message,
                category=category,
                context=context,
                spark_session=spark  # explicitly pass SparkSession
            ) if enabled else nullcontext()

            with timer:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    config = load_yaml_config()
                    profiling_cfg = config.get("profiling", {})
                    enabled = profiling_cfg.get("enabled", False)
                    logger.error("⚠️ Profiling failed with error: %s", str(e))
                    logger.debug("Traceback: %s", traceback.format_exc())
                    if enabled:
                        from db.error_logger import insert_profiling_error
                        insert_profiling_error(
                            log_id=None,  # Or attach if available
                            error=clean_error_message(str(e)),
                            traceback=clean_error_message(traceback.format_exc()),
                            error_type=type(e).__name__,
                            severity="high",
                            component=func.__name__
                        )
                    raise  # Re-raise after logging

        return wrapper
    return decorator

def clean_error_message(message):
    # Remove non-printable or non-ASCII characters
    return re.sub(r'[^\x20-\x7E]+', '', message)