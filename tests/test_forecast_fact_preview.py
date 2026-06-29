"""
Tests for the unbundled ForecastFact-shaped preview
(``UnbundledResults.to_forecast_fact``) and the unbundled **no-write** contract.

All in-memory: successful formatting, missing/invalid fields, and persistence
behaviour are exercised with synthetic ``EntityPerformanceData`` and a mocked
``jdbc_write`` boundary — there is no live database and nothing is ever persisted.

Run from the project root:
    pytest tests/test_forecast_fact_preview.py -v
"""
import os
import sys
import types
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.performance import EntityPerformanceData, UnbundledResults

DATES = pd.to_datetime(["2026-04-01", "2026-05-01", "2026-06-01"])


def _entity(entity_id, entity_type, tariff_type, tariff_id, channels,
            dates=DATES, customer_id="", ufmid=370):
    """An EntityPerformanceData whose long performance_data_frame carries one row
    per consumption_type with a forecast Series (the shape to_forecast_fact molds)."""
    rows = [{"consumption_type": ct, "forecast": pd.Series(vals, index=dates),
             "RMSE": 1.0, "MAE": 1.0, "R2": 0.5} for ct, vals in channels.items()]
    return EntityPerformanceData(
        entity_id=entity_id, entity_type=entity_type, tariff_type=tariff_type,
        customer_id=customer_id, forecast_method_name="SARIMA",
        user_forecast_method_id=ufmid, performance_data_frame=pd.DataFrame(rows),
        tariff_id=tariff_id)


# ── 1. successful formatting ─────────────────────────────────────────────────────

def test_to_forecast_fact_write_shape():
    res = UnbundledResults("SARIMA", [
        _entity("0404.GENWHE", "POD", "LPU", "LANDR1",
                {"PeakConsumption": [1, 2, 3], "StandardConsumption": [4, 5, 6]}),
    ])
    ff = res.to_forecast_fact()
    for col in ("PodID", "UserForecastMethodID", "CustomerID", "ReportingMonth",
                "PeakConsumption", "StandardConsumption",
                "EntityID", "EntityType", "TariffType", "TariffID"):
        assert col in ff.columns, f"missing ForecastFact column {col!r}"
    assert len(ff) == 3                                  # one row per ReportingMonth
    assert (ff["PodID"] == "0404.GENWHE").all()         # EntityID -> PodID
    assert (ff["EntityID"] == "0404.GENWHE").all()
    assert (ff["UserForecastMethodID"] == 370).all()
    assert list(ff["PeakConsumption"]) == [1.0, 2.0, 3.0]
    assert list(ff.sort_values("ReportingMonth")["ReportingMonth"]) == list(DATES)


def test_to_forecast_fact_row_count_and_tariffs():
    res = UnbundledResults("SARIMA", [
        _entity("E1", "POD", "LPU", "T1", {"PeakConsumption": [1, 2, 3]}),
        _entity("E2", "Combo", "SPU", "T2", {"PeakConsumption": [7, 8, 9]}),
    ])
    ff = res.to_forecast_fact()
    assert len(ff) == 6                                  # 2 entities x 3 periods
    assert set(ff["TariffType"]) == {"LPU", "SPU"}
    assert dict(zip(ff["EntityID"], ff["PodID"])) == {"E1": "E1", "E2": "E2"}


# ── 2. missing / invalid fields ──────────────────────────────────────────────────

def test_to_forecast_fact_skips_invalid_entities():
    good = _entity("E1", "POD", "LPU", "T1", {"PeakConsumption": [1, 2, 3]})
    empty_pdf = EntityPerformanceData("E2", "POD", "LPU", "", "SARIMA", 370, pd.DataFrame(), 0)
    none_pdf = EntityPerformanceData("E3", "POD", "LPU", "", "SARIMA", 370, None, 0)
    no_ct_col = EntityPerformanceData("E4", "POD", "LPU", "", "SARIMA", 370,
                                      pd.DataFrame({"value": [1.0]}), 0)
    res = UnbundledResults("SARIMA", [good, empty_pdf, none_pdf, no_ct_col])
    ff = res.to_forecast_fact()
    assert set(ff["EntityID"]) == {"E1"}                 # invalid entities skipped, not fatal


def test_to_forecast_fact_empty_results_returns_empty_df():
    ff = UnbundledResults("SARIMA", []).to_forecast_fact()
    assert isinstance(ff, pd.DataFrame) and ff.empty


# ── 3. persistence behaviour: the unbundled path performs NO db write ────────────

def test_unbundled_path_never_writes_to_db():
    """run_unbundled + to_forecast_fact must never invoke the jdbc_write boundary —
    the unbundled Ermelo path is display-only (DoD #2). Verified with a spy, no DB."""
    from models.algorithms import _unbundled

    data = pd.DataFrame({
        "TariffType": ["LPU"] * 3, "EntityID": ["E1"] * 3, "EntityType": ["POD"] * 3,
        "CustomerID": ["C1"] * 3, "TariffID": [1] * 3,
        "ReportingMonth": DATES, "PeakConsumption": [10.0, 11.0, 12.0],
    })
    model = types.SimpleNamespace(
        dataset=types.SimpleNamespace(ufm_config=types.SimpleNamespace(
            forecast_method_name="SARIMA", user_forecast_method_id=370)),
        config=types.SimpleNamespace(log=False))

    def _stub_forecast(unit, ufm_config, m):
        return _entity(unit.entity_id, unit.entity_type, unit.tariff_type, "T1",
                       {"PeakConsumption": [1, 2, 3]})

    import db.utilities
    writes = []
    with patch.object(db.utilities, "jdbc_write", lambda *a, **k: writes.append(a)), \
         patch.object(_unbundled, "get_unbundled_predictive_data", return_value=data), \
         patch.object(_unbundled, "validate_series", return_value=(True, "ok")):
        res = _unbundled.run_unbundled(model, spark=None, forecast_for_entity=_stub_forecast)
        ff = res.to_forecast_fact()

    assert writes == []                                  # no DB writes on the unbundled path
    assert not ff.empty                                  # yet it DID produce write-ready rows
    assert (ff["PodID"] == "E1").all()
