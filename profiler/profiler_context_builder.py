import re
FORECAST_CONFIG_KEYS = [
    "forecast_method_id",
    "forecast_method_name",
    "databrick_task_id",
    "user_forecast_method_id"
]
def extract_docstring_field(func, key="Function") -> str:
    """
    Extracts the first line matching a key from a function's docstring.
    """
    doc = func.__doc__ or ""
    lines = doc.strip().splitlines()

    for line in lines:
        if line.strip().startswith(f"{key}:"):
            value = line.split(f"{key}:", 1)[-1].strip()
            return re.sub(r"[^\x00-\x7F]+", "",
                          value)  # Strip non-ASCII (or replace with re.sub(r"[\uFFFD]", "?", val))

    return "[No docstring key found]"


import inspect

def profiler_context_builder(instance=None, func=None, args=None, kwargs=None, nested_keys=None):

    context = {}
    context["message"] = extract_docstring_field(func, key="Function")

    # 1. Instance attributes (if class method)
    if instance:
        context["databrick_task_id"] = getattr(instance, "databrick_task_id", None)
        ufm_config = getattr(instance, "ufm_config", None)
        if ufm_config:
            context["forecast_method_id"] = getattr(ufm_config, "forecast_method_id", None)
            context["forecast_method_name"] = getattr(ufm_config, "forecast_method_name", None)
            context["databrick_task_id"] = getattr(ufm_config, "databrick_task_id", None)
            context["user_forecast_method_id"] = getattr(ufm_config, "user_forecast_method_id", None)

    # 2. Bind positional args into named arguments
    bound_args = inspect.signature(func).bind(*args, **(kwargs or {}))
    bound_args.apply_defaults()
    call_args = dict(bound_args.arguments)  # includes both positional and keyword

    # 3. Inject all available call_args into context
    context.update(call_args)

    # Extract from any object in args
    for _, obj in call_args.items():
        for key in FORECAST_CONFIG_KEYS:
            value = getattr_path(obj, key, None)
            if value is not None:
                context[key] = value

        # If nested object like model.dataset.ufm_config exists
        nested = getattr_path(obj, "dataset.ufm_config", None)
        if nested:
            for key in FORECAST_CONFIG_KEYS:
                val = getattr(nested, key, None)
                if val is not None:
                    context[key] = val

    return context


def getattr_path(obj, path: str, default=None):
    """Resolves 'a.b.c' on obj safely."""
    for attr in path.split("."):
        if obj is None:
            return default
        obj = getattr(obj, attr, default)
    return obj