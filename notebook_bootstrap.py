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


def _get_dbutils(spark):
    """A dbutils handle that works in both contexts.

    On a Databricks cluster ``pyspark.dbutils.DBUtils`` exists; under local
    databricks-connect it does not, so fall back to the SDK's
    ``WorkspaceClient().dbutils`` (unified auth — reads DATABRICKS_CONFIG_PROFILE).
    """
    try:
        from pyspark.dbutils import DBUtils
        return DBUtils(spark)
    except (ImportError, ModuleNotFoundError):
        from databricks.sdk import WorkspaceClient
        return WorkspaceClient().dbutils


def init_spark():
    """databricks-connect locally, SparkSession on a cluster; best-effort log level."""
    from pyspark.sql import SparkSession
    if not os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        from databricks.connect import DatabricksSession
        if _truthy(os.environ.get("DATABRICKS_SERVERLESS", "")):
            # Serverless needs no cluster — used locally when no Connect-compatible
            # (Single-User/Shared) cluster is available. DATABRICKS_CLUSTER_ID ignored.
            logger.info("⚡ Local compute: serverless (DATABRICKS_SERVERLESS set).")
            spark = DatabricksSession.builder.serverless(True).getOrCreate()
        else:
            spark = DatabricksSession.builder.getOrCreate()
    else:
        spark = SparkSession.builder.appName("Energy Consumption Prediction").getOrCreate()
    dbutils = _get_dbutils(spark)
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
    return spark, dbutils


def assert_local_workspace(spark, dev_workspace_id=DEV_WORKSPACE_ID):
    """No-op on a cluster; locally, refuse to run against the wrong workspace."""
    if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        return
    workspace_url = ""
    try:
        workspace_url = spark.conf.get("spark.databricks.workspaceUrl", "")
    except Exception:
        pass
    if not workspace_url:
        # databricks-connect doesn't expose workspaceUrl on spark.conf — resolve the
        # configured host from the SDK (reads DATABRICKS_CONFIG_PROFILE) / env instead.
        try:
            from databricks.sdk import WorkspaceClient
            workspace_url = WorkspaceClient().config.host or ""
        except Exception:
            workspace_url = os.environ.get("DATABRICKS_HOST", "")
    assert dev_workspace_id in workspace_url, (
        f"Refusing to run locally — expected DEV ({dev_workspace_id}), "
        f"got workspace '{workspace_url}'. Check DATABRICKS_CLUSTER_ID in .env."
    )


def resolve_task_id(dbutils, default="2"):
    """Local: DATABRICK_TASK_ID env (default 2). Cluster: the DatabrickTaskID widget.

    Declares the widget (with a default) before reading it, so it renders on an
    interactive cluster run too — a job that passes the parameter still overrides
    the value. Mirrors resolve_unbundled's declare-then-get pattern.
    """
    if not os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        return int(os.environ.get("DATABRICK_TASK_ID", default))
    try:
        dbutils.widgets.text("DatabrickTaskID", default, "DatabrickTaskID")
        return int(dbutils.widgets.get("DatabrickTaskID"))
    except Exception:
        # Widget machinery unavailable / undefined → fall back to the default.
        return int(default)


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


def bootstrap_run(method_id, method_name, forecast_unbundled_fn, config):
    """Resolve the whole run environment in one call → ``(spark, dataset, unbundled)``.

    Collapses the notebook's setup boilerplate. Order is load-bearing:

    1. :func:`maybe_run_offline` — if this is an offline fixture run, forecast
       locally and ``sys.exit(0)`` before any Spark/cluster/DB is touched.
    2. :func:`init_spark` + ``set_dbutils`` + :func:`assert_local_workspace` —
       attach to the cluster (DEV via databricks-connect locally, the job's
       SparkSession on Databricks) and refuse to run against the wrong workspace.
    3. :func:`resolve_task_id` + :func:`resolve_unbundled` — read the DatabrickTaskID
       and Unbundled widgets (env vars locally).
    4. ``ForecastDataset(task_id, spark)`` — resolves the UFM config from the DB;
       ``load_data()`` runs only on the bundled path (the unbundled path fetches
       its own PodID data downstream).
    5. :func:`save_fixture` (guarded bundled dump) + ``define_forecast_range()``
       (sets ``dataset.forecast_dates``).

    Returns what :func:`run_forecast` / :func:`render_unbundled` need; inspect
    ``dataset.ufm_config`` in the next cell to see what the run resolved to.
    """
    from data.dataset import ForecastDataset
    from utils.dbutils_singleton import set_dbutils

    maybe_run_offline(method_id, method_name, forecast_unbundled_fn, config)
    spark, dbutils = init_spark()
    set_dbutils(dbutils)
    assert_local_workspace(spark)
    task_id = resolve_task_id(dbutils)
    unbundled = resolve_unbundled(dbutils)
    dataset = ForecastDataset(task_id, spark)
    if not unbundled:
        dataset.load_data()
    save_fixture(dataset)
    dataset.define_forecast_range()
    return spark, dataset, unbundled


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
    n_pods = perf["PodID"].nunique() if "PodID" in perf.columns else "?"
    print(f"📊 Unbundled forecast — {len(perf)} row(s) across {n_pods} "
          f"pod{'' if n_pods == 1 else 's'}; display-only, no DB writes.")
    cols = [c for c in ("PodID", "TariffType", "ReportingMonth",
                        "forecast", "RMSE", "MAE", "R2") if c in perf.columns]
    preview = perf[cols] if cols else perf
    try:
        from IPython.display import display
        display(preview)
    except Exception:
        print(preview.to_string(index=False))

    # ForecastFact-shaped preview — the wide write-shape the bundled writer would
    # persist (one row per ReportingMonth, a column per consumption type), molded from
    # the same builder. Display only — the unbundled path performs NO DB writes.
    try:
        ff = result.to_forecast_fact()
        if ff is not None and not ff.empty:
            print(f"🧱 ForecastFact preview — {len(ff)} row(s), write-shape; NOT written.")
            try:
                from IPython.display import display
                display(ff)
            except Exception:
                print(ff.to_string(index=False))
    except Exception as exc:
        print(f"⚠️ ForecastFact preview unavailable: {exc}")
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


def maybe_run_offline(method_id, method_name, forecast_unbundled_fn, config):
    """Local-only offline gate — no-op on Databricks and on any live run.

    If ``--mode unbundled`` and ``PREDICTIVE_FIXTURE_PATH`` are both set (the offline
    fixture launch configs), forecast against that local CSV/parquet, render the
    ForecastFact preview, and exit before any Spark/cluster/DB is touched. Otherwise
    return immediately so the real flow (Databricks job or live-DEV fetch) proceeds.
    This is the single place the notebook diverts to a local fixture run; on the
    cluster no CLI args are present, so it always falls through.
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
    fixture_path = os.environ["PREDICTIVE_FIXTURE_PATH"]
    if fixture_path.endswith(".csv"):
        # utf-8-sig strips the BOM Results.csv carries on its header.
        fixture_df = pd.read_csv(fixture_path, encoding="utf-8-sig")
    else:
        fixture_df = pd.read_parquet(fixture_path)
    # CSV loads ReportingMonth as strings; the horizon arithmetic needs datetimes.
    max_date = pd.to_datetime(fixture_df["ReportingMonth"]).max()
    start = max_date + pd.DateOffset(months=1)
    end = start + pd.DateOffset(months=11)
    # The preview stamps UserForecastMethodID from this config (the one UFMID
    # source); align the stub with the fixture's own UFMID so preview rows carry
    # the real value (421 for Results.csv) instead of a made-up 0.
    ufmid = (int(fixture_df["UserForecastMethodID"].iloc[0])
             if "UserForecastMethodID" in fixture_df.columns and len(fixture_df) else 0)
    ufm_config = ForecastConfig(
        forecast_method_id=method_id,
        forecast_method_name=method_name,
        model_parameters=config.get("model_parameters", ""),
        region="LOCAL", status="Active", user_forecast_method_id=ufmid,
        start_date=start, end_date=end, databrick_task_id=0,
    )
    model_stub = types.SimpleNamespace(
        dataset=types.SimpleNamespace(ufm_config=ufm_config),
        config=types.SimpleNamespace(log=config.get("log", False)),
    )
    result = forecast_unbundled_fn(model_stub, spark=None)
    render_unbundled(result, unbundled=True)
    sys.exit(0)
