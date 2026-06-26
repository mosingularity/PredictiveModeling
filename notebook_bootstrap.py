"""Shared bootstrap for the ESF-*.py Databricks notebooks.

The four notebooks duplicated their Spark init, local-workspace guard, task-id
resolution, fixture-dump, and unbundled-fixture entrypoint — and had drifted
(RandomForest forced ENV=PROD, SARIMA forced DEV, only ARIMA/XGBoost ran
locally). This module is the single source of truth so all four agree.

ENV resolution lives at import-callers' top cells via ``resolve_env()``:
an explicitly-injected ENV always wins; otherwise PROD on a cluster
(``DATABRICKS_RUNTIME_VERSION`` set), DEV locally (databricks-connect).
"""
import logging
import os

logger = logging.getLogger(__name__)

DEV_WORKSPACE_ID = "adb-7405614865299820"


def resolve_env():
    """Default ENV by where we run, without clobbering an injected value."""
    os.environ.setdefault(
        "ENV", "PROD" if os.environ.get("DATABRICKS_RUNTIME_VERSION") else "DEV"
    )
    where = "cluster" if os.environ.get("DATABRICKS_RUNTIME_VERSION") else "local databricks-connect"
    logger.info(f"🌍 ENV resolved to {os.environ['ENV']} ({where}).")


def init_spark():
    """databricks-connect locally, SparkSession on a cluster; best-effort log level."""
    from pyspark.sql import SparkSession
    from pyspark.dbutils import DBUtils
    if not os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        from databricks.connect import DatabricksSession
        spark = DatabricksSession.builder.getOrCreate()
    else:
        spark = SparkSession.builder.appName("Energy Consumption Prediction").getOrCreate()
    try:
        spark.sparkContext.setLogLevel("ERROR")
    except Exception:
        # Unsupported on Shared-access clusters / via databricks-connect.
        pass
    try:
        ws = spark.conf.get("spark.databricks.workspaceUrl", "<unknown>")
    except Exception:
        ws = "<unknown>"
    logger.info(f"⚡ Spark session ready — workspace '{ws}'.")
    return spark, DBUtils(spark)


def assert_local_workspace(spark, dev_workspace_id=DEV_WORKSPACE_ID):
    """No-op on a cluster; locally, refuse to run against the wrong workspace."""
    if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        return
    workspace_url = ""
    try:
        workspace_url = spark.conf.get("spark.databricks.workspaceUrl", "")
    except Exception:
        pass
    assert dev_workspace_id in workspace_url, (
        f"Refusing to run locally — expected DEV ({dev_workspace_id}), "
        f"got workspace '{workspace_url}'. Check DATABRICKS_CLUSTER_ID in .env."
    )


def resolve_task_id(dbutils, default="2"):
    """Local: DATABRICK_TASK_ID env (default 2). Cluster: the DatabrickTaskID widget."""
    if not os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        return int(os.environ.get("DATABRICK_TASK_ID", default))
    return int(dbutils.widgets.get("DatabrickTaskID"))


def _truthy(value):
    return str(value).strip().lower() in ("1", "true", "yes", "y")


def resolve_unbundled(dbutils, default=False):
    """Mode flag — True → unbundled (entity-keyed LPU/SPU/PPU), False → bundled (customer/pod).

    Local: the ``UNBUNDLED`` env var (default ``false``). Cluster: the ``Unbundled``
    widget (created here if absent, default ``false``). Mirrors :func:`resolve_task_id`
    — the flag only selects the iteration path; the model and its parameters still
    come from the UFM config. Bundled is the default so existing jobs are unchanged;
    unbundled is opt-in.
    """
    if not os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        raw = os.environ.get("UNBUNDLED", str(default))
    else:
        default_str = "true" if default else "false"
        try:
            dbutils.widgets.dropdown(
                "Unbundled", default_str, ["true", "false"],
                "Unbundled (entity-keyed) mode?")
            raw = dbutils.widgets.get("Unbundled")
        except Exception:
            # Widget machinery unavailable / undefined → fall back to the default.
            raw = default_str
    unbundled = _truthy(raw)
    logger.info(f"🔀 Mode flag resolved: {'unbundled' if unbundled else 'bundled'}.")
    return unbundled


def run_forecast(dataset, spark, config, forecast_unbundled_fn, unbundled):
    """Dispatch a loaded ``ForecastDataset`` to the unbundled or bundled forecaster.

    Both read the model and parameters from ``dataset.ufm_config``; ``unbundled``
    only chooses entity-keyed iteration vs the customer/pod pipeline. Returns the
    forecaster's result.
    """
    if unbundled:
        from models.base import ForecastModel
        logger.info("🔀 Running UNBUNDLED (entity-keyed) forecast.")
        return forecast_unbundled_fn(ForecastModel(dataset, config), spark)
    from programs.pipeline import ForecastPipeline
    logger.info("🔀 Running BUNDLED (customer/pod) forecast.")
    return ForecastPipeline(dataset=dataset, config=config).run(spark)


def render_unbundled(result, unbundled=True):
    """Display-only terminal step for the unbundled path.

    Renders the unbundled forecast output inline in the notebook cell (a summary
    line plus a forecast/metrics table) and performs **no** DB writes — the
    unbundled Ermelo path is a cells-only display demo. No-op for the bundled
    path, which persists to ForecastFact / StatisticalPerformanceMetrics instead.
    Returns the previewed DataFrame, or None when skipped.
    """
    if not unbundled:
        return None
    perf = result.get_performance_data()
    if perf is None or perf.empty:
        print("⚠️ Unbundled run produced no forecast rows (nothing to display).")
        return perf
    n_entities = perf["EntityID"].nunique() if "EntityID" in perf.columns else "?"
    print(f"📊 Unbundled forecast — {len(perf)} row(s) across {n_entities} "
          f"entit{'y' if n_entities == 1 else 'ies'}; display-only, no DB writes.")
    cols = [c for c in ("EntityID", "EntityType", "TariffType", "ReportingMonth",
                        "forecast", "RMSE", "MAE", "R2") if c in perf.columns]
    preview = perf[cols] if cols else perf
    try:
        from IPython.display import display
        display(preview)
    except Exception:
        print(preview.to_string(index=False))
    return preview


def save_fixture(dataset):
    """If SAVE_FIXTURE is set, dump dataset.processed_df to data/fixtures and exit; else no-op."""
    if not os.environ.get("SAVE_FIXTURE"):
        return
    import pathlib
    import sys
    pathlib.Path("data/fixtures").mkdir(parents=True, exist_ok=True)
    df = dataset.processed_df.copy()
    df.attrs = {}
    print("\n=== FIXTURE SCHEMA ===")
    print(df.dtypes.to_string())
    print(f"\nShape: {df.shape}")
    print(f"\nSample (3 rows):\n{df.head(3).to_string()}")
    df.to_parquet("data/fixtures/raw_predictive_input.parquet", index=False)
    df.to_csv("data/fixtures/raw_predictive_input.csv", index=False)
    print("\nSaved to data/fixtures/raw_predictive_input.parquet and .csv")
    sys.exit(0)


def run_unbundled_fixture(method_id, method_name, forecast_unbundled_fn, config):
    """If ``--mode unbundled`` and PREDICTIVE_FIXTURE_PATH is set, run the unbundled
    forecast against the fixture and exit; otherwise no-op (the bundled path proceeds).
    """
    import argparse
    import sys
    import types
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", default="bundled", choices=["bundled", "unbundled"])
    run_args, _ = parser.parse_known_args()
    if not (run_args.mode == "unbundled" and os.environ.get("PREDICTIVE_FIXTURE_PATH")):
        return
    import pandas as pd
    from db.queries import ForecastConfig
    fixture_df = pd.read_parquet(os.environ["PREDICTIVE_FIXTURE_PATH"])
    max_date = fixture_df["ReportingMonth"].max()
    start = max_date + pd.DateOffset(months=1)
    end = start + pd.DateOffset(months=11)
    ufm_config = ForecastConfig(
        forecast_method_id=method_id,
        forecast_method_name=method_name,
        model_parameters=config.get("model_parameters", ""),
        region="LOCAL", status="Active", user_forecast_method_id=0,
        start_date=start, end_date=end, databrick_task_id=0,
    )
    model_stub = types.SimpleNamespace(
        dataset=types.SimpleNamespace(ufm_config=ufm_config),
        config=types.SimpleNamespace(log=config.get("log", False)),
    )
    result = forecast_unbundled_fn(model_stub, spark=None)
    render_unbundled(result, unbundled=True)
    sys.exit(0)
