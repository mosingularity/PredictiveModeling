"""One readable row per edge case: what it contains, what the path did, and why.

For each named case in the shared three-member fixture (tests/fixtures/
edge_cases.py) this runs the *actual* unbundled path over the affected member —
the real validator and the real forecast_for_pod_id, not a re-description — and
prints where the member ended up: forecast, skipped at validation, or logged and
zero-filled. The point is that someone can read each case's outcome without
opening a test file.

Nothing here writes to a database. It builds an in-memory fixture and re-runs the
same functions the pipeline uses; the fits are real but their output is discarded.

Usage (run from the predictive_modeling folder, or anywhere — it chdirs itself):

    python scripts/edge_case_evidence.py                 # unbundled path, all cases
    python scripts/edge_case_evidence.py --path unbundled
    python scripts/edge_case_evidence.py --case negative_months
"""
import argparse
import os
import sys
import types
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PACKAGE_ROOT)
sys.path.insert(0, str(PACKAGE_ROOT))


def _ensure_pyspark_stubs() -> None:
    """The DB layer imports pyspark at module load. When it is not installed
    (this script runs outside the Databricks runtime and outside pytest, so
    conftest's stubs never fire), install the same minimal stubs conftest uses so
    the offline, fixture-backed path imports cleanly."""
    try:
        import pyspark  # noqa: F401
        return
    except ImportError:
        pass

    class _Stub:
        def __init__(self, *a, **kw): pass
        def __call__(self, *a, **kw): return self
        def __getattr__(self, _): return self

    stub = _Stub()

    def _pkg(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        mod.__path__ = []
        mod.__package__ = name
        return mod

    mods = {
        "pyspark": _pkg("pyspark"),
        "pyspark.sql": _pkg("pyspark.sql", DataFrame=_Stub, SparkSession=_Stub),
        "pyspark.sql.functions": _pkg(
            "pyspark.sql.functions", col=stub, to_date=stub, expr=stub, explode=stub,
            array=stub, lit=stub, when=stub, to_timestamp=stub, month=stub, year=stub),
        "pyspark.sql.window": _pkg("pyspark.sql.window", Window=_Stub),
        "pyspark.sql.types": _pkg(
            "pyspark.sql.types", StructType=_Stub, StructField=_Stub, IntegerType=_Stub,
            StringType=_Stub, TimestampType=_Stub, FloatType=_Stub, LongType=_Stub),
        "pyspark.dbutils": _pkg("pyspark.dbutils", DBUtils=_Stub),
    }
    for name, mod in mods.items():
        sys.modules.setdefault(name, mod)


_ensure_pyspark_stubs()
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from evaluation.performance import PredictionUnit  # noqa: E402
from validation.series import validate_series  # noqa: E402
from models.algorithms.autoarima import forecast_for_pod_id  # noqa: E402
from models.bundle import (  # noqa: E402
    BUNDLE_CONSUMPTION_COLUMN, aggregate_members, build_member_series,
    compute_shares, forecast_for_bundle,
)
from tests.fixtures.edge_cases import (  # noqa: E402
    AFFECTED_MEMBER, CASE_NAMES, CASES, build_case_frame,
    default_ufm_config, long_horizon_ufm_config,
)

RULE = "─" * 100


def _model_stub():
    return types.SimpleNamespace(config=types.SimpleNamespace(log=False))


def _last_value_fit(series, horizon):
    """A trivial, deterministic aggregate fit_fn (the real fits are exercised by the
    test suite; here the point is where the MEMBER ends up, not the estimator)."""
    idx = pd.date_range(series.index[-1], periods=horizon + 1, freq="MS")[1:]
    return pd.Series(float(series.iloc[-1]), index=idx)


def _pod_indexed(frame: pd.DataFrame, pod_id: str) -> pd.DataFrame:
    sub = frame[frame["PodID"] == pod_id].copy().set_index("ReportingMonth")
    sub.index = pd.to_datetime(sub.index)
    return sub.sort_index()


def _unit_for(frame: pd.DataFrame, pod_id: str) -> PredictionUnit:
    series = _pod_indexed(frame, pod_id)
    return PredictionUnit(
        entity_id=pod_id, entity_type="",
        tariff_type=series["TariffType"].iloc[0],
        customer_id=str(series["CustomerID"].iloc[0]),
        tariff_id=series["TariffID"].iloc[0], series=series,
    )


def _contains(frame: pd.DataFrame) -> str:
    """A concrete one-liner about the affected member's series."""
    df = _pod_indexed(frame, AFFECTED_MEMBER)
    months = pd.PeriodIndex(df.index, freq="M").drop_duplicates()
    peak = pd.to_numeric(df["PeakConsumption"])
    span = f"{months.min()}→{months.max()}"
    facts = [f"{len(df)} rows", f"{len(months)} distinct months", span]
    if df.index.duplicated().any():
        facts.append("has a duplicate month")
    if (peak < 0).any():
        facts.append("has credit (negative) months")
    if peak.max() == 0:
        facts.append("all-zero")
    elif peak.max() >= 10 * max(peak[peak > 0].median(), 1):
        facts.append(f"outlier peak {peak.max():.0f}")
    return ", ".join(facts)


def evaluate_case(case: str) -> dict:
    """Run the real unbundled path over the affected member and classify the
    outcome as forecast / skipped / logged, with the reason recorded against it."""
    frame = build_case_frame(case)
    cfg = long_horizon_ufm_config() if case == "horizon_longer_than_history" \
        else default_ufm_config()
    unit = _unit_for(frame, AFFECTED_MEMBER)

    ok, reason = validate_series(unit, cfg.forecast_method_name)
    if not ok:
        return {"case": case, "contains": _contains(frame),
                "did": "skipped", "why": f"validation: {reason}"}

    # Passed validation → run the real per-member forecast, capturing any error
    # the function logs (gap-too-large, flat, fit failure) and whether a real
    # (non-zero) forecast came back.
    logged: list[str] = []

    def _capture(**kwargs):
        logged.append(kwargs.get("error_type", "?"))

    with patch("models.algorithms.autoarima.report_validation_error",
               side_effect=_capture):
        perf = forecast_for_pod_id(
            df=unit.series, order=(1, 1, 1), customer_id=unit.customer_id,
            pod_id=AFFECTED_MEMBER, consumption_types=["PeakConsumption"],
            ufm_config=cfg, forecast_model=_model_stub(),
        )

    row = perf.performance_data_frame.iloc[0]
    forecast = row["forecast"]
    n_steps = len(forecast) if hasattr(forecast, "__len__") else 0
    nonzero = bool(np.any(np.asarray(forecast, dtype=float) != 0))
    hard = {"ForecastGapTooLarge", "InvalidSeries",
            "SplitConfigurationError", "ModelFitFailure"}
    hard_hit = [e for e in logged if e in hard]

    if hard_hit:
        return {"case": case, "contains": _contains(frame),
                "did": "logged", "why": f"{hard_hit[0]} → zero-filled "
                f"({row['validation_reason']})"}
    warn = f" (warned: {logged[0]})" if logged else ""
    scored = "scored" if not np.isnan(row["R2"]) else "unscored"
    return {"case": case, "contains": _contains(frame), "did": "forecast",
            "why": f"{n_steps}-month horizon, {'non-zero' if nonzero else 'zero'} "
            f"forecast, backtest {scored}{warn}"}


def evaluate_case_bundled(case: str) -> dict:
    """Run the REAL bundled path over the case's three members and classify what happens
    to the affected member: forecast (with its share, and whether the aggregate
    zero-filled its hole), or rejected because the AGGREGATE failed validation.

    The bundled path aggregates the members before validating, so a member's own defect
    (a six-month hole, an all-zero series, a short history) is usually invisible here —
    the aggregate is fine and the member is forecast anyway, at a share the defect has
    quietly moved. That asymmetry with the unbundled path is the whole point of the row.
    """
    frame = build_case_frame(case)
    horizon = 36 if case == "horizon_longer_than_history" else 6
    wide = build_member_series(frame, "PeakConsumption")
    agg = aggregate_members(wide)[BUNDLE_CONSUMPTION_COLUMN]

    try:
        parts = forecast_for_bundle(wide, method=default_ufm_config().forecast_method_name,
                                    horizon=horizon, fit_fn=_last_value_fit)
    except ValueError as exc:
        return {"case": case, "contains": _contains(frame),
                "did": "rejected", "why": f"aggregate validation: {exc}"}

    share = float(compute_shares(wide).get(AFFECTED_MEMBER, float("nan")))
    healthy = float(compute_shares(build_member_series(build_case_frame("happy"),
                                                       "PeakConsumption"))[AFFECTED_MEMBER])
    moved = ""
    if abs(share - healthy) > 1e-6:
        moved = f", share {'depressed' if share < healthy else 'inflated'} " \
                f"({share:.3f} vs {healthy:.3f} healthy)"
    member_fc = parts[AFFECTED_MEMBER]
    kind = "non-zero" if bool((member_fc != 0).any()) else "zero"
    return {"case": case, "contains": _contains(frame), "did": "forecast",
            "why": f"{horizon}-month horizon, {kind} member forecast, "
            f"share {share:.3f}{moved}"}


def _print_single(path: str, rows: list) -> None:
    """One block per case for a single path."""
    print(f"\nEdge-case evidence — path: {path}, affected member: "
          f"{AFFECTED_MEMBER}\n{RULE}")
    wc = max(len(r["case"]) for r in rows)
    wd = max(len(r["did"]) for r in rows)
    for r in rows:
        print(f"{r['case']:<{wc}}  [{r['did']:^{wd}}]  {r['why']}")
        print(f"{'':<{wc}}  contains: {r['contains']}")
        print(f"{'':<{wc}}  intent:   {CASES[r['case']]}")
        print()
    did = pd.Series([r["did"] for r in rows]).value_counts().to_dict()
    print(RULE)
    print("tally: " + ", ".join(f"{k}={v}" for k, v in did.items())
          + f"  ({len(rows)} cases)")


def _print_both(cases: list) -> None:
    """One row per case, both paths side by side, with a column marking divergence.

    "Diverges" is the outcome that matters: did the affected member end up FORECAST or
    not? The unbundled path drops it (skipped / logged); the bundled path forecasts it
    anyway behind the aggregate. Every ◆ row is a member the two paths treat differently
    on the same data.
    """
    u = {c: evaluate_case(c) for c in cases}
    b = {c: evaluate_case_bundled(c) for c in cases}
    wc = max(len(c) for c in cases)
    print(f"\nEdge-case evidence — both paths, affected member: {AFFECTED_MEMBER}\n{RULE}")
    print(f"{'case':<{wc}}  {'unbundled':<9}  {'bundled':<9}  diverges")
    print(RULE)
    n_div = 0
    for c in cases:
        u_fc = u[c]["did"] == "forecast"
        b_fc = b[c]["did"] == "forecast"
        diverges = u_fc != b_fc
        n_div += diverges
        mark = "◆ member forecast one way, dropped the other" if diverges else ""
        print(f"{c:<{wc}}  {u[c]['did']:<9}  {b[c]['did']:<9}  {mark}")
        print(f"{'':<{wc}}  unbundled: {u[c]['why']}")
        print(f"{'':<{wc}}  bundled:   {b[c]['why']}")
        print()
    print(RULE)
    print(f"tally: {n_div} of {len(cases)} cases diverge between the paths")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", default="unbundled",
                    choices=["unbundled", "bundled", "both"],
                    help="which forecasting path to exercise (bundled/both are plan 13)")
    ap.add_argument("--case", default=None, choices=CASE_NAMES,
                    help="run a single case instead of all")
    args = ap.parse_args()

    cases = [args.case] if args.case else CASE_NAMES
    if args.path == "both":
        _print_both(cases)
    elif args.path == "bundled":
        _print_single("bundled", [evaluate_case_bundled(c) for c in cases])
    else:
        _print_single("unbundled", [evaluate_case(c) for c in cases])


if __name__ == "__main__":
    main()
