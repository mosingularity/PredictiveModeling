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
    """ENV selects which SQL Server config.yaml points at (DEV/UAT/QA/PROD).

    It comes straight from the environment — each Databricks cluster sets ``ENV`` in
    its config once (DEV cluster → ``ENV=DEV``, QA cluster → ``ENV=QA``, PROD cluster
    → ``ENV=PROD``); locally ``.env`` sets it. Defaults to DEV when unset, so a
    misconfigured run never silently reads/writes prod.
    """
    env = os.environ.setdefault("ENV", "DEV")
    logger.info(f"🌍 ENV = {env}.")
    return env


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
    the value. This is now the notebooks' only widget — the mode comes from the UFM's
    BundledInd, not from an operator (plan 11).
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


def run_forecast(dataset, spark, config, forecasters):
    """Route the loaded dataset to the unbundled or bundled forecaster.

    The mode is a property of the UFM, not of the run: ``BundledInd = 1`` forecasts the
    UFM's entities as one aggregate disaggregated to members, ``0`` or ``NULL`` forecasts
    each separately. There is no widget, env var or flag to override it, so a run can never
    contradict the configuration it was launched for.

    ``forecasters`` is ``{"unbundled": fn, "bundled": fn}``; both take ``(model, spark)``
    and return a ``ForecastResults``. The model and its parameters also come from
    ``dataset.ufm_config``. Neither path writes to the database.

    Validation warnings are collected for the whole run and emitted once at the end,
    grouped by type with the pods affected — a run over many pods otherwise repeats the
    same warning per pod·channel.
    """
    from db.error_logger import collect_validation_errors
    from models.base import ForecastModel
    ufm = dataset.ufm_config
    mode = "bundled" if ufm.bundled else "unbundled"
    # print, not logger.info: the notebooks set logging to WARNING, and since the mode is no
    # longer visible in the invocation, the resolved decision *and its reason* must always
    # reach the operator. Mirrors the offline gate's line so both paths read identically.
    print(f"🔀 {mode.upper()} (UFM {ufm.user_forecast_method_id}, "
          f"BundledInd={ufm.bundled}).")
    label = f"{mode} {ufm.forecast_method_name} (UFM {ufm.user_forecast_method_id})"
    with collect_validation_errors(label):
        return forecasters[mode](ForecastModel(dataset, config), spark)


def bootstrap_run(method_id, method_name, forecasters, config):
    """Resolve the whole run environment in one call → ``(spark, dataset)``.

    Order is load-bearing: the offline-fixture gate runs first and exits before any
    Spark/cluster/DB is touched; only then attach Spark, guard the workspace, read the
    task-id widget, and resolve the dataset. No up-front ``load_data()`` — both modes fetch
    their own PodID data downstream. ``forecasters`` is the ``{"unbundled", "bundled"}``
    routing table (see :func:`run_forecast`).

    The task id is the only input. The mode arrives with the dataset, on
    ``dataset.ufm_config.bundled``.
    """
    from data.dataset import ForecastDataset
    from utils.dbutils_singleton import set_dbutils
    from utils.quiet_warnings import quiet_third_party_warnings

    quiet_third_party_warnings()
    resolve_env()
    run_offline_fixture(method_id, method_name, forecasters, config)
    spark, dbutils = init_spark()
    set_dbutils(dbutils)
    assert_local_workspace(spark)
    task_id = resolve_task_id(dbutils)
    dataset = ForecastDataset(task_id, spark)
    save_fixture(dataset)
    dataset.define_forecast_range()
    return spark, dataset


def render_forecast(result):
    """Render a forecast result inline (summary + forecast/metrics table + ForecastFact
    preview). Mode-agnostic — both paths return the same ``ForecastResults``. Display only:
    NO DB writes on either path. Returns the previewed DataFrame, or None when empty.
    """
    perf = result.get_performance_data()
    if perf is None or perf.empty:
        print("⚠️ Run produced no forecast rows (nothing to display).")
        return perf
    n_pods = "?"
    if "PodID" in perf.columns:
        n_pods = perf["PodID"].nunique()
    pod_word = "pod" if n_pods == 1 else "pods"
    print(f"📊 Forecast — {len(perf)} row(s) across {n_pods} {pod_word}; "
          f"display-only, no DB writes.")
    wanted_cols = ("PodID", "TariffType", "ReportingMonth", "forecast", "RMSE", "MAE", "R2")
    cols = [c for c in wanted_cols if c in perf.columns]
    preview = perf[cols] if cols else perf
    try:
        from IPython.display import display
        display(preview)
    except Exception:
        print(preview.to_string(index=False))
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


def _offline_model(ufm_config, log):
    """A ``ForecastModel`` stand-in for offline fixture runs.

    Both forecasters read only ``model.dataset.ufm_config`` and ``model.config.log``; a
    real ``ForecastModel`` needs a ``ForecastDataset``, which needs a live Spark session,
    so offline we hand them this lightweight double instead (the same SimpleNamespace
    idiom the tests and ``forecasting/engine.py`` use).
    """
    import types
    return types.SimpleNamespace(
        dataset=types.SimpleNamespace(ufm_config=ufm_config),
        config=types.SimpleNamespace(log=log),
    )


def run_offline_fixture(method_id, method_name, forecasters, config):
    """Local-only offline gate — no-op unless ``PREDICTIVE_FIXTURE_PATH`` is set.

    When set (the offline launch configs), forecast the local fixture, render the preview,
    and ``sys.exit(0)`` before any Spark is touched. On a cluster the variable is unset, so
    it falls through to the live flow.

    The mode comes from the fixture's own ``BundledInd`` column, mirroring how a live run
    reads it off the UFM row. Fixtures predating bundled modelling have no such column,
    which reads as NULL and runs unbundled — the same rule the database follows.
    """
    import sys
    if not os.environ.get("PREDICTIVE_FIXTURE_PATH"):
        return
    import pandas as pd
    from db.queries import ForecastConfig
    fixture_path = os.environ["PREDICTIVE_FIXTURE_PATH"]
    if fixture_path.endswith(".csv"):
        fixture_df = pd.read_csv(fixture_path, encoding="utf-8-sig")
    else:
        fixture_df = pd.read_parquet(fixture_path)
    max_date = pd.to_datetime(fixture_df["ReportingMonth"]).max()
    start = max_date + pd.DateOffset(months=1)
    end = start + pd.DateOffset(months=11)
    ufmid = 0
    if "UserForecastMethodID" in fixture_df.columns and len(fixture_df):
        ufmid = int(fixture_df["UserForecastMethodID"].iloc[0])
    bundled = False
    if "BundledInd" in fixture_df.columns and len(fixture_df):
        bundled_value = fixture_df["BundledInd"].iloc[0]
        if pd.notna(bundled_value):
            bundled = bool(bundled_value)
    ufm_config = ForecastConfig(
        forecast_method_id=method_id,
        forecast_method_name=method_name,
        model_parameters=config.get("model_parameters", ""),
        region="LOCAL", status="Active", user_forecast_method_id=ufmid,
        start_date=start, end_date=end, databrick_task_id=0, bundled=bundled,
    )
    mode = "bundled" if bundled else "unbundled"
    print(f"🔀 {mode.upper()} (UFM {ufmid}, BundledInd={bundled}).")
    model = _offline_model(ufm_config, log=config.get("log", False))
    from db.error_logger import collect_validation_errors
    with collect_validation_errors(f"{mode} {method_name} (UFM {ufmid}, offline fixture)"):
        result = forecasters[mode](model, spark=None)
    render_forecast(result)
    sys.exit(0)
