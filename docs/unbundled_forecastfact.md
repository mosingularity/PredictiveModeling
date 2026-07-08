# Unbundled path & ForecastFact preview — implementation notes

Notes captured while implementing the unbundled (Ermelo, entity-keyed) display path
and the ForecastFact-shaped preview on `upgrade_system`. Companion to the
`databricks-e2e` validation plan.

## Summary

The **unbundled** path (entity-keyed LPU/SPU/PPU, Ermelo) is a **display-only** demo:
it loads entity-keyed data, fits per-entity models, and renders (1) a forecast/metrics
table and (2) a **ForecastFact-shaped preview** — and performs **zero DB writes**. The
**bundled** (customer/pod) path is unchanged and still writes `ForecastFact` +
`StatisticalPerformanceMetrics`.

## Assumptions

- **Unbundled = no DB writes** (DoD #2). Enforced structurally (no `jdbc_write` in the
  unbundled call graph) and by test (`test_forecast_fact_preview.py::
  test_unbundled_path_never_writes_to_db`).
- **`load_data()` is bundled-only** — it runs `load_and_prepare_data`, which sorts/casts
  `PodID`/`CustomerID`. It is **skipped on the unbundled path** (`if not unbundled:`); the
  unbundled path needs only `ufm_config` and fetches its own entity-keyed data.
- **`ForecastModel.__init__` no longer requires a loaded `processed_df`** — `None` is
  allowed (unbundled); `run_bundled` re-checks `None`/empty for the bundled path.
- **`EntityID → PodID`** in the ForecastFact preview is a **real POD for LPU**, but a
  **synthetic `Combo`/`CSA` composite for SPU/PPU**. The entity identity
  (`EntityID/EntityType/TariffType/TariffID`) is carried alongside so the mapping is visible.
- **The unbundled query** (`UNBUNDLED_FILTERS`) is hardcoded to
  `CustomerServiceArea = 'Ermelo'` (SPU + LPU scopes) and **ignores the UFMID**; the
  `ErmeloSource` widget overrides the CSA, the task only selects model/params/dates.
- **On-cluster `ENV` defaults to PROD** (`resolve_env`); DEV validation requires `ENV=DEV`
  (cluster Spark env var or a top cell). Confirm via `get_jdbc_options()[0]` → `…dev-01`.
- **Local fitting** runs in the `dbx-connect` env; `statsmodels` matches the cluster
  (0.14.0), but `pandas`/`sklearn`/`holidays` differ — minor numeric diffs possible.

## Schema

No change to the `ForecastFact` table itself. Two shapes meet in `to_forecast_fact`:

| | Columns |
|---|---|
| **ForecastFact write-shape** (`build_forecast_df`) | `PodID, UserForecastMethodID, CustomerID, ReportingMonth, <consumption-type columns>` — one row per ReportingMonth |
| **Unbundled entity contract** (`to_contract` / fixture) | `EntityID, TariffType, EntityType, TariffID, ReportingMonth, <consumption columns>` (+ `Scenario` in the fixture) |

`UnbundledResults.to_forecast_fact()` molds the entity contract into the write-shape via
the same `build_forecast_df` the bundled writer uses (EntityID→PodID), and carries
`EntityID/EntityType/TariffType/TariffID` alongside.

- **New env var:** `UNBUNDLED_FIXTURE_PATH` — unbundled-only fixture override.
  `get_unbundled_predictive_data` reads it first, then falls back to
  `PREDICTIVE_FIXTURE_PATH`. Kept separate so the unbundled fixture toggle never
  redirects the bundled `load_data()` at the entity-keyed parquet.

## Follow-up work

- **No ARIMA task** exists in DatabrickTasks 1–4 (RF=1, SARIMA=2, XGBoost=3/4). Add an
  ARIMA UFM/task for the ARIMA cell of the e2e matrix, or document it as not covered.
- **SPU/PPU `EntityID → PodID` is synthetic.** Decide the real `ForecastFact` key mapping
  before any unbundled write is ever contemplated.
- **Generation/wheeling entities** (e.g. `…GENWHE`) carry negative consumption and fit
  poorly under SARIMA (R² ≈ −1.3). Decide whether to filter or model them differently.
- **Live row-count proofs still pending:** DoD #2 (unbundled Δ=0) and DoD #3 (bundled
  identical to `dev`) need a real bundled run on DEV with before/after counts.
- **Library version skew** local vs cluster — pin the local `dbx-connect` env to the DBR
  15.4 versions if bit-identical local↔cluster parity is required.
- **Serverless cannot reach the DEV SQL Server** (no proxy/MSI). DB runs must use the
  Single-User cluster (`0629-005357-2jrm7wyx`).

## Tests (all mocked — no live DB)

- `tests/test_forecast_fact_preview.py` — successful formatting, missing/invalid fields,
  and the unbundled no-write contract (`jdbc_write` spy).
- `tests/test_bundled_loop.py` — bundled path writes exactly two tables (`jdbc_write` spy).
- `tests/test_unbundled_loop.py` — unbundled loop structure (one call per entity group).
