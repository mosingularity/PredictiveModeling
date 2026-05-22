import os

def is_databricks():
    try:
        import IPython
        return "DATABRICKS_RUNTIME_VERSION" in os.environ or "dbutils" in globals()
    except ImportError:
        return False

def get_env() -> str:
    return os.getenv("ENV", "LOCAL").upper()
