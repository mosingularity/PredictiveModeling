"""Tests for db.unbundled_query — query assembly and contract mapping (no DB)."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.unbundled_query import (
    UNBUNDLED_FILTERS, build_query, to_contract,
)


# ── build_query: must be a single, wrappable SELECT ──────────────────────────────

def test_query_has_no_temp_tables_or_batch():
    sql = build_query(UNBUNDLED_FILTERS[0][1]).lower()
    assert "drop table" not in sql
    assert "into #" not in sql
    assert "#category" not in sql and "#csa" not in sql and "#tariff" not in sql


def test_query_has_no_order_by():
    # ORDER BY is invalid inside Spark's (SELECT * FROM (<query>) alias) wrapper.
    assert "order by" not in build_query(UNBUNDLED_FILTERS[0][1]).lower()


def test_query_inlines_lookup_values():
    sql = build_query(UNBUNDLED_FILTERS[0][1]).lower()
    assert "values" in sql
    assert "ermel" in sql  # the CSA lookup is inlined


def test_query_embeds_the_filter():
    where = UNBUNDLED_FILTERS[0][1]
    assert where in build_query(where)


def test_each_filter_builds():
    for _, where in UNBUNDLED_FILTERS:
        assert build_query(where).lower().startswith("select")


# ── to_contract: raw result → entity-keyed contract ──────────────────────────────

def _raw_row(**kw):
    base = dict(
        ID="POD_1", CustomerType="LPU", Tariff="LANDR1",
        ReportingMonth="2025-05-01", PeakConsumption=10.0,
        StandardConsumption=20.0, OffpeakConsumption=5.0,
    )
    base.update(kw)
    return base


def test_to_contract_maps_identifiers():
    out = to_contract(pd.DataFrame([_raw_row()]))
    assert out.loc[0, "EntityID"] == "POD_1"
    assert out.loc[0, "TariffType"] == "LPU"
    assert out.loc[0, "EntityType"] == "POD"     # derived from TariffType
    assert out.loc[0, "TariffID"] == "LANDR1"


def test_to_contract_entity_type_per_tariff():
    df = pd.DataFrame([_raw_row(CustomerType="LPU"), _raw_row(CustomerType="SPU")])
    out = to_contract(df)
    assert list(out["EntityType"]) == ["POD", "Combo"]


def test_to_contract_renames_offpeak_and_zero_fills_blocks():
    out = to_contract(pd.DataFrame([_raw_row(OffpeakConsumption=7.5)]))
    assert out.loc[0, "OffPeakConsumption"] == 7.5
    for col in ("Block1Consumption", "NonTOUConsumption"):
        assert out.loc[0, col] == 0.0


def test_to_contract_columns_match_validator_contract():
    out = to_contract(pd.DataFrame([_raw_row()]))
    expected = {
        "EntityID", "TariffType", "EntityType", "TariffID", "ReportingMonth",
        "PeakConsumption", "StandardConsumption", "OffPeakConsumption",
        "Block1Consumption", "Block2Consumption", "Block3Consumption",
        "Block4Consumption", "NonTOUConsumption",
    }
    assert set(out.columns) == expected
