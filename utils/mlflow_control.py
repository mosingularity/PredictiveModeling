import logging

def disable_all_autologgers():
    try:
        import mlflow
        mlflow.autolog(disable=True)
        try:
            mlflow.statsmodels.autolog(disable=True)
        except Exception:
            pass
        logging.getLogger("mlflow").setLevel(logging.WARNING)
    except ImportError:
        pass
