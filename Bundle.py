# Databricks notebook source

# COMMAND ----------

import sys
import logging
sys.path.append("/Workspace/Shared")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j.clientserver").setLevel(logging.WARNING)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from notebook_bootstrap import resolve_env, init_spark, assert_local_workspace, resolve_task_id
from models.bundle import build_member_series, aggregate_members, compute_shares


def show(df):
    try:
        from IPython.display import display
        display(df)
    except Exception:
        print(df.to_string())

# COMMAND ----------

# Part 1 — small-CSV units (LPU/SPU1/SPU2, 12 months)

DUMMY_CSV_PATH = "data/fixtures/bundle_dummy_12mo.csv"
DUMMY_CHANNEL = "TotalConsumption"

# COMMAND ----------

dummy_df = pd.read_csv(DUMMY_CSV_PATH, parse_dates=["ReportingMonth"])
show(dummy_df)

# COMMAND ----------

fig, ax = plt.subplots()
for pod in sorted(dummy_df["PodID"].unique()):
    pod_rows = dummy_df[dummy_df["PodID"] == pod].sort_values("ReportingMonth")
    ax.plot(pod_rows["ReportingMonth"], pod_rows[DUMMY_CHANNEL], marker="o", label=pod)
ax.set_title("Dummy CSV — actuals per member")
ax.legend()
plt.show()

# COMMAND ----------

dummy_members = build_member_series(dummy_df, DUMMY_CHANNEL)
show(dummy_members)

# COMMAND ----------

dummy_bundle = aggregate_members(dummy_members)
show(dummy_bundle)

# COMMAND ----------

fig, ax = plt.subplots()
for pod in dummy_members.columns:
    ax.plot(dummy_members.index, dummy_members[pod], alpha=0.55, label=pod)
ax.plot(dummy_bundle.index, dummy_bundle.iloc[:, 0], linewidth=3, marker="o", label="Bundle (row-sum)")
ax.set_title("Dummy CSV — bundle vs members")
ax.legend()
plt.show()

# COMMAND ----------

dummy_shares = compute_shares(dummy_members)
dummy_member_totals = dummy_members.sum(axis=0)
dummy_share_table = pd.DataFrame({
    "member_total": dummy_member_totals,
    "share": dummy_shares,
})
show(dummy_share_table)

# COMMAND ----------

fig, ax = plt.subplots()
ax.bar(dummy_shares.index, dummy_shares.to_numpy())
for i, v in enumerate(dummy_shares.to_numpy()):
    ax.text(i, v, f"{v:.0%}", ha="center", va="bottom")
ax.set_title("Dummy CSV — share of bundle total")
plt.show()

# COMMAND ----------

# Disaggregation of history: bundle actual x each member's share, per month.
# No forecaster is involved — this is the same outer-product identity
# forecast_for_bundle uses for a real forecast, applied here to the historical
# bundle instead. See docs/evidence/bundle-small-csv-units/QUERY.md.
dummy_bundle_col = dummy_bundle.iloc[:, 0]
dummy_disaggregated = pd.DataFrame(
    np.outer(dummy_bundle_col.to_numpy(), dummy_shares.to_numpy()),
    index=dummy_bundle_col.index, columns=dummy_shares.index,
)
show(dummy_disaggregated)

# COMMAND ----------

fig, ax = plt.subplots()
for pod in dummy_members.columns:
    ax.plot(dummy_members.index, dummy_members[pod], alpha=0.4, label=f"{pod} actual")
    ax.plot(dummy_disaggregated.index, dummy_disaggregated[pod], linestyle="--", marker="o",
           label=f"{pod} disaggregated")
ax.set_title("Dummy CSV — disaggregation reconstructs each member")
ax.legend(fontsize=8)
plt.show()

# COMMAND ----------

dummy_reconstructed = dummy_disaggregated.sum(axis=1)
dummy_max_gap = float((dummy_reconstructed - dummy_bundle_col).abs().max())
print(f"Bundle total: {float(dummy_bundle_col.sum()):g}  (expect 11900)")
print(f"Shares: {dummy_shares.round(4).to_dict()}  (expect 0.20 / 0.25 / 0.55)")
print(f"Max |sum(disaggregated) - bundle| across months: {dummy_max_gap:.2e}  (expect 0)")

# COMMAND ----------

# Part 2 — live DatabrickTaskID path

resolve_env()

# COMMAND ----------

try:
    spark, dbutils = init_spark()
    assert_local_workspace(spark)
    task_id = resolve_task_id(dbutils)
    live_cluster = True
except Exception as exc:
    logger.warning(f"⚠️ No live Spark/cluster available ({type(exc).__name__}: {exc}); "
                   f"falling back to the fixture-backed task-ID path.")
    spark, dbutils, task_id = None, None, None
    live_cluster = False

# COMMAND ----------

from db.queries import ForecastConfig, get_user_forecast_data, row_to_config, get_unbundled_predictive_data
from data.dml import convert_to_pandas

REAL_FIXTURE_PATH = "data/fixtures/Results_421.csv"
REAL_CHANNEL = "PeakConsumption"


def load_ufm_config(spark, task_id):
    """Real fetch on a live cluster; fixture-backed fallback offline — same
    construction notebook_bootstrap.run_offline_fixture already uses, so the rest
    of this section behaves identically whichever path resolved the config."""
    if spark is not None and task_id is not None:
        rows = get_user_forecast_data(spark, task_id)
        rows = convert_to_pandas(rows)
        if not rows.empty:
            return row_to_config(rows.iloc[0]), True
    fixture_df = pd.read_csv(REAL_FIXTURE_PATH, encoding="utf-8-sig")
    ufmid = int(fixture_df["UserForecastMethodID"].iloc[0])
    bundled = bool(fixture_df["BundledInd"].iloc[0]) if "BundledInd" in fixture_df.columns else False
    last_month = pd.to_datetime(fixture_df["ReportingMonth"]).max()
    start = last_month + pd.DateOffset(months=1)
    end = start + pd.DateOffset(months=11)
    config = ForecastConfig(
        forecast_method_id=1, forecast_method_name="Bundle-units-only",
        model_parameters="", region="LOCAL", status="Active",
        user_forecast_method_id=ufmid, start_date=start, end_date=end,
        databrick_task_id=0, bundled=bundled,
    )
    return config, False


ufm_config, task_id_resolved = load_ufm_config(spark, task_id)
print(f"UFMID: {ufm_config.user_forecast_method_id}  BundledInd: {int(ufm_config.bundled)}  "
      f"task_id_resolved: {task_id_resolved}")

# COMMAND ----------

if not task_id_resolved:
    os.environ["UNBUNDLED_FIXTURE_PATH"] = REAL_FIXTURE_PATH
live_raw = get_unbundled_predictive_data(spark, UFMID=ufm_config.user_forecast_method_id)
live_raw = convert_to_pandas(live_raw)
show(live_raw.head(20))

# COMMAND ----------

live_actual_by_pod = live_raw.pivot_table(index="ReportingMonth", columns="PodID",
                                          values=REAL_CHANNEL, aggfunc="sum")
live_actual_by_pod.index = pd.to_datetime(live_actual_by_pod.index)
live_actual_by_pod = live_actual_by_pod.sort_index()

fig, ax = plt.subplots()
for pod in live_actual_by_pod.columns:
    ax.plot(live_actual_by_pod.index, live_actual_by_pod[pod], marker="o", label=pod)
ax.set_title("Live members — actuals")
ax.legend(fontsize=8)
plt.show()

# COMMAND ----------

live_members = build_member_series(live_raw, REAL_CHANNEL)
show(live_members)

# COMMAND ----------

live_bundle = aggregate_members(live_members)
show(live_bundle)

# COMMAND ----------

fig, ax = plt.subplots()
for pod in live_members.columns:
    ax.plot(live_members.index, live_members[pod], alpha=0.55, label=pod)
ax.plot(live_bundle.index, live_bundle.iloc[:, 0], linewidth=3, marker="o", label="Bundle (row-sum)")
ax.set_title("Live — bundle vs members")
ax.legend(fontsize=8)
plt.show()

# COMMAND ----------

live_shares = compute_shares(live_members)
live_member_totals = live_members.sum(axis=0)
live_share_table = pd.DataFrame({
    "member_total": live_member_totals,
    "share": live_shares,
})
show(live_share_table)

# COMMAND ----------

fig, ax = plt.subplots()
ax.bar(range(len(live_shares)), live_shares.to_numpy())
ax.set_xticks(range(len(live_shares)))
ax.set_xticklabels(list(live_shares.index), rotation=20, ha="right", fontsize=7)
for i, v in enumerate(live_shares.to_numpy()):
    ax.text(i, v, f"{v:.1%}", ha="center", va="bottom")
ax.set_title("Live — share of bundle total")
plt.show()

# COMMAND ----------

live_bundle_col = live_bundle.iloc[:, 0]
live_disaggregated = pd.DataFrame(
    np.outer(live_bundle_col.to_numpy(), live_shares.to_numpy()),
    index=live_bundle_col.index, columns=live_shares.index,
)
show(live_disaggregated)

# COMMAND ----------

fig, axes = plt.subplots(1, len(live_members.columns), figsize=(5 * len(live_members.columns), 4))
for ax, pod in zip(axes, live_members.columns):
    ax.plot(live_members.index, live_members[pod], alpha=0.4, label="actual")
    ax.plot(live_disaggregated.index, live_disaggregated[pod], linestyle="--", label="disaggregated")
    ax.set_title(pod, fontsize=9)
    ax.legend(fontsize=7)
fig.suptitle("Live — disaggregation reconstructs each member")
plt.show()

# COMMAND ----------

live_reconstructed = live_disaggregated.sum(axis=1)
live_max_gap = float((live_reconstructed - live_bundle_col).abs().max())
print(f"Bundle = row-sum of members: {abs(float(live_bundle_col.sum()) - float(live_member_totals.sum())) < 1e-6}")
print(f"Shares sum to 1: {abs(float(live_shares.sum()) - 1.0) < 1e-6}")
print(f"Disaggregation reconstructs the bundle every month: {live_max_gap < 1e-6}")
