"""
Tests for plan 01f — fixture edge-case entities.

Three layers:
  1. Unit tests: builder functions in unbundled-modelling/scripts/build_fixture.py
  2. Fixture content tests: assertions on the built parquet (skip if not built yet)
  3. Integration tests: forecast_for_podel_id logs the right error for each bad scenario;
     forecast_arima_unbundled loop routes every edge-case entity

Run from the project root:
    pytest tests/test_fixture_edge_cases.py -v
Build the fixture first (for Part 2):
    python unbundled-modelling/scripts/build_fixture.py
"""
import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from db.queries import ForecastConfig
from evaluation.performance import EntityPerformanceData, UnbundledResults
from validation.series import validate_series as _real_validate_series

# ── load build_fixture.py (directory name has a hyphen, can't be imported normally) ──
# unbundled-modelling/ is local-only planning scratch (gitignored): on a machine
# without it (cluster, CI, fresh clone) skip this module rather than error out.
_BF_PATH = PROJECT_ROOT / "unbundled-modelling" / "scripts" / "build_fixture.py"
if not _BF_PATH.exists():
    pytest.skip("unbundled-modelling scratch (edge-case builders) not present",
                allow_module_level=True)
_spec = importlib.util.spec_from_file_location("build_fixture", _BF_PATH)
_bf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bf)

build_short_series_rows   = _bf.build_short_series_rows
build_gapped_series_rows  = _bf.build_gapped_series_rows
build_all_zero_rows       = _bf.build_all_zero_rows
build_outlier_spike_rows  = _bf.build_outlier_spike_rows

FIXTURE_PATH      = PROJECT_ROOT / "data" / "fixtures" / "unbundled_predictive_input.parquet"
FIXTURE_AVAILABLE = FIXTURE_PATH.exists()


# ─── shared helpers ───────────────────────────────────────────────────────────

def _ufm_config() -> ForecastConfig:
    """Config with forecast window 2025-01 → 2025-06 (6 months)."""
    return ForecastConfig(
        forecast_method_id=1,
        forecast_method_name="ARIMA",
        model_parameters="",
        region="TEST",
        status="Active",
        user_forecast_method_id=99,
        start_date=pd.Timestamp("2025-01-01"),
        end_date=pd.Timestamp("2025-06-01"),
        databrick_task_id=0,
    )


def _model_stub():
    return types.SimpleNamespace(
        dataset=types.SimpleNamespace(ufm_config=_ufm_config()),
        config=types.SimpleNamespace(log=False),
    )


def _pod_frame(entity_id: str, dates: pd.DatetimeIndex, peak: list) -> pd.DataFrame:
    """DataFrame matching forecast_for_podel_id's expected format (PodID column, date index)."""
    return pd.DataFrame({"PodID": entity_id, "PeakConsumption": peak}, index=dates)


def _logged_error_types(mock_log: MagicMock) -> list[str]:
    return [c.kwargs["error_type"] for c in mock_log.call_args_list]


# ══════════════════════════════════════════════════════════════════════════════
# Part 1 — Unit tests: edge-case builder functions
# ══════════════════════════════════════════════════════════════════════════════

class TestShortSeriesBuilder:
    def test_row_count(self):
        assert len(build_short_series_rows()) == 6

    def test_entity_id(self):
        assert (build_short_series_rows()["EntityID"] == "POD_SHORT_SERIES").all()

    def test_scenario_label(self):
        assert (build_short_series_rows()["Scenario"] == "short_series").all()

    def test_tariff_type(self):
        assert (build_short_series_rows()["TariffType"] == "LPU").all()

    def test_date_range_starts_2024_07(self):
        assert build_short_series_rows()["ReportingMonth"].min() == pd.Timestamp("2024-07-01")

    def test_date_range_ends_2024_12(self):
        assert build_short_series_rows()["ReportingMonth"].max() == pd.Timestamp("2024-12-01")


class TestGappedSeriesBuilder:
    def test_row_count(self):
        assert len(build_gapped_series_rows()) == 20

    def test_entity_id(self):
        assert (build_gapped_series_rows()["EntityID"] == "POD_GAPPED_SERIES").all()

    def test_scenario_label(self):
        assert (build_gapped_series_rows()["Scenario"] == "gapped_series").all()

    def test_gap_months_absent(self):
        months = build_gapped_series_rows()["ReportingMonth"]
        gap = pd.date_range("2022-10-01", "2023-01-01", freq="MS")
        assert not months.isin(gap).any()

    def test_series_boundaries(self):
        df = build_gapped_series_rows()
        assert df["ReportingMonth"].min() == pd.Timestamp("2022-01-01")
        assert df["ReportingMonth"].max() == pd.Timestamp("2023-12-01")

    def test_gap_size_is_4_months(self):
        df = build_gapped_series_rows()
        full = pd.date_range("2022-01-01", "2023-12-01", freq="MS")
        assert len(full) - len(df) == 4


class TestAllZeroBuilder:
    def test_row_count(self):
        assert len(build_all_zero_rows()) == 24

    def test_entity_id(self):
        assert (build_all_zero_rows()["EntityID"] == "COMBO_ZERO_CONSUMPTION").all()

    def test_scenario_label(self):
        assert (build_all_zero_rows()["Scenario"] == "all_zero").all()

    def test_tariff_type_spu(self):
        assert (build_all_zero_rows()["TariffType"] == "SPU").all()

    def test_all_consumption_columns_zero(self):
        df = build_all_zero_rows()
        consumption_cols = [c for c in df.columns if "Consumption" in c]
        assert (df[consumption_cols] == 0).all().all()


class TestOutlierSpikeBuilder:
    def test_row_count(self):
        assert len(build_outlier_spike_rows()) == 36

    def test_entity_id(self):
        assert (build_outlier_spike_rows()["EntityID"] == "POD_OUTLIER_SPIKE").all()

    def test_scenario_label(self):
        assert (build_outlier_spike_rows()["Scenario"] == "outlier_spike").all()

    def test_spike_value_at_index_20(self):
        assert build_outlier_spike_rows().iloc[20]["PeakConsumption"] == 9999.0

    def test_non_spike_months_are_200(self):
        df = build_outlier_spike_rows()
        assert (df[df["PeakConsumption"] != 9999.0]["PeakConsumption"] == 200.0).all()

    def test_series_ends_at_2024_12(self):
        df = build_outlier_spike_rows()
        assert df["ReportingMonth"].max() == pd.Timestamp("2024-12-01")


# ══════════════════════════════════════════════════════════════════════════════
# Part 2 — Fixture content tests (require the parquet to be built first)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def fixture_df():
    if not FIXTURE_AVAILABLE:
        pytest.skip(
            "Fixture not built — run: python unbundled-modelling/scripts/build_fixture.py"
        )
    return pd.read_parquet(FIXTURE_PATH)


@pytest.mark.skipif(not FIXTURE_AVAILABLE, reason="Fixture parquet not built")
class TestFixtureContent:
    def test_scenario_column_present(self, fixture_df):
        assert "Scenario" in fixture_df.columns

    def test_scenario_no_nulls(self, fixture_df):
        assert fixture_df["Scenario"].isna().sum() == 0

    def test_all_required_scenarios_present(self, fixture_df):
        required = {"happy_path", "short_series", "gapped_series", "all_zero", "outlier_spike"}
        assert required <= set(fixture_df["Scenario"].unique())

    def test_short_series_row_count(self, fixture_df):
        short = fixture_df[fixture_df["Scenario"] == "short_series"]
        assert len(short) == 6

    def test_short_series_entity_id(self, fixture_df):
        short = fixture_df[fixture_df["Scenario"] == "short_series"]
        assert (short["EntityID"] == "POD_SHORT_SERIES").all()

    def test_gapped_series_row_count(self, fixture_df):
        gapped = fixture_df[fixture_df["Scenario"] == "gapped_series"]
        assert len(gapped) == 20

    def test_gapped_series_entity_id(self, fixture_df):
        gapped = fixture_df[fixture_df["Scenario"] == "gapped_series"]
        assert (gapped["EntityID"] == "POD_GAPPED_SERIES").all()

    def test_all_zero_row_count(self, fixture_df):
        zeros = fixture_df[fixture_df["Scenario"] == "all_zero"]
        assert len(zeros) == 24

    def test_all_zero_consumption_values(self, fixture_df):
        zeros = fixture_df[fixture_df["Scenario"] == "all_zero"]
        consumption_cols = [c for c in zeros.columns if "Consumption" in c]
        assert (zeros[consumption_cols] == 0).all().all()

    def test_outlier_spike_row_count(self, fixture_df):
        spike = fixture_df[fixture_df["Scenario"] == "outlier_spike"]
        assert len(spike) == 36

    def test_outlier_spike_peak_max(self, fixture_df):
        spike = fixture_df[fixture_df["Scenario"] == "outlier_spike"]
        assert spike["PeakConsumption"].max() == 9999.0

    def test_happy_path_rows_present(self, fixture_df):
        assert len(fixture_df[fixture_df["Scenario"] == "happy_path"]) > 0

    def test_all_tariff_types_present(self, fixture_df):
        assert {"LPU", "SPU", "PPU"} <= set(fixture_df["TariffType"].unique())

    def test_entity_id_no_nulls(self, fixture_df):
        assert fixture_df["EntityID"].isna().sum() == 0

    def test_reporting_month_is_datetime(self, fixture_df):
        assert pd.api.types.is_datetime64_any_dtype(fixture_df["ReportingMonth"])


# ══════════════════════════════════════════════════════════════════════════════
# Part 3 — Error-logging unit tests via forecast_for_podel_id
#
# These tests call forecast_for_podel_id directly with edge-case DataFrames
# that have a PodID column (the format the function expects), patch
# report_validation_error, and assert the right error_type is logged.
# ══════════════════════════════════════════════════════════════════════════════

from models.algorithms.autoarima import forecast_for_podel_id


class TestShortSeriesLogsError:
    """6-month series (< 12) → SplitConfigurationError before ARIMA is attempted."""

    def _run(self) -> MagicMock:
        dates = pd.date_range("2024-07-01", periods=6, freq="MS")
        # Use varying values so the InvalidSeries (all-same) check doesn't fire first.
        peak = [190.0 + i * 10 for i in range(6)]
        df = _pod_frame("POD_SHORT_SERIES", dates, peak)
        mock_log = MagicMock()
        with patch("models.algorithms.autoarima.report_validation_error", mock_log):
            forecast_for_podel_id(
                df=df,
                order=(1, 1, 1),
                customer_id="SYNTH_EDGE_001",
                pod_id="POD_SHORT_SERIES",
                consumption_types=["PeakConsumption"],
                ufm_config=_ufm_config(),
                forecast_model=_model_stub(),
            )
        return mock_log

    def test_error_is_logged(self):
        assert self._run().called

    def test_error_type_is_split_configuration(self):
        assert "SplitConfigurationError" in _logged_error_types(self._run())

    def test_no_arima_fit_attempted(self):
        mock_log = MagicMock()
        with patch("models.algorithms.autoarima.report_validation_error", mock_log), \
             patch("models.algorithms.autoarima.fit_time_series_model") as mock_fit:
            dates = pd.date_range("2024-07-01", periods=6, freq="MS")
            df = _pod_frame("POD_SHORT_SERIES", dates, [190.0 + i * 10 for i in range(6)])
            forecast_for_podel_id(
                df=df, order=(1, 1, 1), customer_id="SYNTH_EDGE_001",
                pod_id="POD_SHORT_SERIES", consumption_types=["PeakConsumption"],
                ufm_config=_ufm_config(), forecast_model=_model_stub(),
            )
        mock_fit.assert_not_called()


class TestAllZeroLogsError:
    """All-zero series → InvalidSeries before ARIMA is attempted."""

    def _run(self) -> MagicMock:
        dates = pd.date_range("2022-01-01", periods=24, freq="MS")
        df = _pod_frame("COMBO_ZERO_CONSUMPTION", dates, [0.0] * 24)
        mock_log = MagicMock()
        with patch("models.algorithms.autoarima.report_validation_error", mock_log):
            forecast_for_podel_id(
                df=df,
                order=(1, 1, 1),
                customer_id="SYNTH_EDGE_003",
                pod_id="COMBO_ZERO_CONSUMPTION",
                consumption_types=["PeakConsumption"],
                ufm_config=_ufm_config(),
                forecast_model=_model_stub(),
            )
        return mock_log

    def test_error_is_logged(self):
        assert self._run().called

    def test_error_type_is_invalid_series(self):
        assert "InvalidSeries" in _logged_error_types(self._run())

    def test_no_arima_fit_attempted(self):
        mock_log = MagicMock()
        with patch("models.algorithms.autoarima.report_validation_error", mock_log), \
             patch("models.algorithms.autoarima.fit_time_series_model") as mock_fit:
            dates = pd.date_range("2022-01-01", periods=24, freq="MS")
            df = _pod_frame("COMBO_ZERO_CONSUMPTION", dates, [0.0] * 24)
            forecast_for_podel_id(
                df=df, order=(1, 1, 1), customer_id="SYNTH_EDGE_003",
                pod_id="COMBO_ZERO_CONSUMPTION", consumption_types=["PeakConsumption"],
                ufm_config=_ufm_config(), forecast_model=_model_stub(),
            )
        mock_fit.assert_not_called()


class TestGappedSeriesLogsError:
    """Series ending 2023-12 with forecast window starting 2025-01 → ForecastGapTooLarge."""

    def _run(self) -> MagicMock:
        full = pd.date_range("2022-01-01", "2023-12-01", freq="MS")
        gap = pd.date_range("2022-10-01", "2023-01-01", freq="MS")
        dates = full[~full.isin(gap)]  # 20 months, last = 2023-12-01
        # Vary values so InvalidSeries check is bypassed.
        peak = [200.0 + i * 5 for i in range(len(dates))]
        df = _pod_frame("POD_GAPPED_SERIES", dates, peak)
        mock_log = MagicMock()
        with patch("models.algorithms.autoarima.report_validation_error", mock_log):
            forecast_for_podel_id(
                df=df,
                order=(1, 1, 1),
                customer_id="SYNTH_EDGE_002",
                pod_id="POD_GAPPED_SERIES",
                consumption_types=["PeakConsumption"],
                ufm_config=_ufm_config(),  # start_date=2025-01-01, gap > 12 months
                forecast_model=_model_stub(),
            )
        return mock_log

    def test_error_is_logged(self):
        assert self._run().called

    def test_error_type_is_gap_too_large(self):
        assert "ForecastGapTooLarge" in _logged_error_types(self._run())

    def test_no_arima_fit_attempted(self):
        mock_log = MagicMock()
        with patch("models.algorithms.autoarima.report_validation_error", mock_log), \
             patch("models.algorithms.autoarima.fit_time_series_model") as mock_fit:
            full = pd.date_range("2022-01-01", "2023-12-01", freq="MS")
            gap = pd.date_range("2022-10-01", "2023-01-01", freq="MS")
            dates = full[~full.isin(gap)]
            df = _pod_frame("POD_GAPPED_SERIES", dates, [200.0 + i * 5 for i in range(len(dates))])
            forecast_for_podel_id(
                df=df, order=(1, 1, 1), customer_id="SYNTH_EDGE_002",
                pod_id="POD_GAPPED_SERIES", consumption_types=["PeakConsumption"],
                ufm_config=_ufm_config(), forecast_model=_model_stub(),
            )
        mock_fit.assert_not_called()


class TestOutlierSpikeNoPreflightError:
    """Spike entity has 36 months ending 2024-12 → passes all pre-flight checks."""

    def test_no_split_or_invalid_series_error(self):
        dates = pd.date_range("2022-01-01", periods=36, freq="MS")
        peak = [200.0] * 36
        peak[20] = 9999.0
        df = _pod_frame("POD_OUTLIER_SPIKE", dates, peak)

        forecast_horizon = pd.date_range("2025-01-01", periods=6, freq="MS")
        mock_fit_instance = MagicMock()
        mock_fit_instance.get_forecast.return_value.predicted_mean = pd.Series(
            [210.0, 212.0, 214.0, 216.0, 218.0, 220.0], index=forecast_horizon
        )
        mock_fit_instance.predict.return_value = pd.Series(
            [205.0] * 6, index=forecast_horizon
        )

        mock_log = MagicMock()
        with patch("models.algorithms.autoarima.report_validation_error", mock_log), \
             patch("models.algorithms.autoarima.fit_time_series_model", return_value=mock_fit_instance):
            forecast_for_podel_id(
                df=df,
                order=(1, 1, 1),
                customer_id="SYNTH_EDGE_004",
                pod_id="POD_OUTLIER_SPIKE",
                consumption_types=["PeakConsumption"],
                ufm_config=_ufm_config(),
                forecast_model=_model_stub(),
            )

        logged = _logged_error_types(mock_log)
        assert "SplitConfigurationError" not in logged
        assert "InvalidSeries" not in logged
        assert "ForecastGapTooLarge" not in logged

    def test_arima_fit_is_called(self):
        dates = pd.date_range("2022-01-01", periods=36, freq="MS")
        peak = [200.0] * 36
        peak[20] = 9999.0
        df = _pod_frame("POD_OUTLIER_SPIKE", dates, peak)

        forecast_horizon = pd.date_range("2025-01-01", periods=6, freq="MS")
        mock_fit_instance = MagicMock()
        mock_fit_instance.get_forecast.return_value.predicted_mean = pd.Series(
            [210.0, 212.0, 214.0, 216.0, 218.0, 220.0], index=forecast_horizon
        )
        mock_fit_instance.predict.return_value = pd.Series([205.0] * 6, index=forecast_horizon)

        with patch("models.algorithms.autoarima.report_validation_error"), \
             patch("models.algorithms.autoarima.fit_time_series_model",
                   return_value=mock_fit_instance) as mock_fit:
            forecast_for_podel_id(
                df=df,
                order=(1, 1, 1),
                customer_id="SYNTH_EDGE_004",
                pod_id="POD_OUTLIER_SPIKE",
                consumption_types=["PeakConsumption"],
                ufm_config=_ufm_config(),
                forecast_model=_model_stub(),
            )

        # Fitted twice now: once on the full series (forecast + in-sample fitted
        # line) and once on the training portion for the out-of-sample backtest
        # that the metrics are scored on. The point of this test — that a valid
        # outlier-spike series reaches the fit and isn't skipped — still holds.
        assert mock_fit.call_count == 2


# ══════════════════════════════════════════════════════════════════════════════
# Part 4 — Loop integration: forecast_arima_unbundled routes all edge entities
#
# forecast_for_entity is mocked to avoid the old bundled-path PodID column
# requirement. These tests verify the loop's groupby routing, not model fitting.
# ══════════════════════════════════════════════════════════════════════════════

from models.algorithms.autoarima import forecast_arima_unbundled

_EDGE_ENTITY_IDS = {
    "POD_SHORT_SERIES",
    "POD_GAPPED_SERIES",
    "COMBO_ZERO_CONSUMPTION",
    "POD_OUTLIER_SPIKE",
}


def _build_edge_fixture() -> pd.DataFrame:
    """In-memory fixture with all 4 edge-case entities (no file I/O)."""
    df = pd.concat(
        [build_short_series_rows(), build_gapped_series_rows(),
         build_all_zero_rows(), build_outlier_spike_rows()],
        ignore_index=True,
    )
    df["TariffID"]    = df["TariffID"].astype(str)
    df["CustomerID"]  = df["CustomerID"].astype(str)
    # The loop now groups by PodID; the entity-keyed builders predate that, so
    # mirror EntityID until the builders are rewritten to the PodID contract.
    df["PodID"] = df["EntityID"]
    return df


_EDGE_FIXTURE = _build_edge_fixture()


def _entity_perf_stub(unit, *args, **kwargs) -> EntityPerformanceData:
    return EntityPerformanceData(
        entity_id=unit.entity_id,
        entity_type=unit.entity_type,
        tariff_type=unit.tariff_type,
        customer_id=unit.customer_id,
        forecast_method_name="ARIMA",
        user_forecast_method_id=99,
        performance_data_frame=pd.DataFrame({"value": [1.0]}),
    )


# Pre-flight skip-design (plan 10 / unbundled-01f): the loop validates every entity
# and skips short, gapped, and all-zero series *before* routing to forecast_for_entity.
# Only POD_OUTLIER_SPIKE — a long, populated series whose only quirk is an outlier —
# passes validation and is forecast.
_VALID_EDGE_IDS   = {"POD_OUTLIER_SPIKE"}
_SKIPPED_EDGE_IDS = {"POD_SHORT_SERIES", "POD_GAPPED_SERIES", "COMBO_ZERO_CONSUMPTION"}


@pytest.fixture(scope="module")
def loop_run():
    """Run forecast_arima_unbundled once and share results across tests in this class.

    Captures both which units the loop *constructs* (every entity, via a
    validate_series spy that delegates to the real validator) and which it *routes*
    to forecast_for_entity (only those that pass pre-flight). This lets the type
    tests assert on all entities while the routing tests assert the skip behaviour.
    """
    route_spy = MagicMock(side_effect=_entity_perf_stub)
    constructed = []

    def _validate_capture(unit, method):
        constructed.append(unit)
        return _real_validate_series(unit, method)

    with patch("models.algorithms._unbundled.get_unbundled_predictive_data", return_value=_EDGE_FIXTURE), \
         patch("models.algorithms.autoarima.forecast_for_entity", route_spy), \
         patch("models.algorithms._unbundled.validate_series", side_effect=_validate_capture):
        result = forecast_arima_unbundled(_model_stub(), spark=None)
    return result, route_spy, constructed


class TestLoopRoutesEdgeEntities:
    def test_returns_unbundled_results_type(self, loop_run):
        result, _, _ = loop_run
        assert isinstance(result, UnbundledResults)

    def test_all_edge_entities_validated(self, loop_run):
        # Every entity is constructed and validated, even the ones later skipped.
        _, _, constructed = loop_run
        assert {u.entity_id for u in constructed} == _EDGE_ENTITY_IDS

    def test_only_valid_entity_routed(self, loop_run):
        _, spy, _ = loop_run
        assert spy.call_count == len(_VALID_EDGE_IDS)

    def test_correct_entity_ids_routed(self, loop_run):
        _, spy, _ = loop_run
        routed = {c.args[0].entity_id for c in spy.call_args_list}
        assert routed == _VALID_EDGE_IDS

    def test_only_valid_entity_in_results(self, loop_run):
        result, _, _ = loop_run
        assert len(result.entity_performance) == len(_VALID_EDGE_IDS)

    def test_result_entity_ids_match(self, loop_run):
        result, _, _ = loop_run
        result_ids = {e.entity_id for e in result.entity_performance}
        assert result_ids == _VALID_EDGE_IDS

    def test_short_series_entity_skipped(self, loop_run):
        _, spy, _ = loop_run
        ids = {c.args[0].entity_id for c in spy.call_args_list}
        assert "POD_SHORT_SERIES" not in ids

    def test_gapped_series_entity_skipped(self, loop_run):
        _, spy, _ = loop_run
        ids = {c.args[0].entity_id for c in spy.call_args_list}
        assert "POD_GAPPED_SERIES" not in ids

    def test_zero_consumption_entity_skipped(self, loop_run):
        _, spy, _ = loop_run
        ids = {c.args[0].entity_id for c in spy.call_args_list}
        assert "COMBO_ZERO_CONSUMPTION" not in ids

    def test_outlier_spike_entity_routed(self, loop_run):
        _, spy, _ = loop_run
        ids = {c.args[0].entity_id for c in spy.call_args_list}
        assert "POD_OUTLIER_SPIKE" in ids

    def test_tariff_types_correctly_assigned(self, loop_run):
        # Type assignment is checked on the constructed units (all entities),
        # independent of whether they were later skipped by pre-flight.
        _, _, constructed = loop_run
        units = {u.entity_id: u for u in constructed}
        assert units["POD_SHORT_SERIES"].tariff_type  == "LPU"
        assert units["POD_GAPPED_SERIES"].tariff_type == "LPU"
        assert units["COMBO_ZERO_CONSUMPTION"].tariff_type == "SPU"
        assert units["POD_OUTLIER_SPIKE"].tariff_type == "LPU"

    def test_entity_types_no_longer_derived(self, loop_run):
        # The PodID contract drops the LPU/SPU-derived entity-type routing;
        # every unit carries the neutral placeholder.
        _, _, constructed = loop_run
        assert {u.entity_type for u in constructed} == {""}
