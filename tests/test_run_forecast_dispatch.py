"""Dispatch tests for notebook_bootstrap.run_forecast (plan 11).

The UFM decides the iteration, not the operator: ``BundledInd = 1`` routes the data through
the bundle path (:func:`models.algorithms.bundled.run_bundled`), ``0`` or ``NULL`` runs the
per-entity loop. There is no widget, env var or CLI flag left to override it, so these tests
drive the routing entirely from ``ufm_config.bundled``.

The NULL case carries the most weight: every UFM predating bundled modelling has
``BundledInd = NULL`` (tasks 1 to 4 in DEV), and all of them must keep running unbundled.

The legacy ``ForecastPipeline`` (retired in plan 01) must stay gone.
"""
import importlib
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.queries import row_to_config
from notebook_bootstrap import run_forecast


def _stub_dataset(bundled):
    """A minimal ForecastDataset stand-in: processed_df=None so ForecastModel init
    passes, plus the ufm_config run_forecast/run_bundled read from."""
    return types.SimpleNamespace(
        processed_df=None,
        ufm_config=types.SimpleNamespace(
            forecast_method_name="SARIMA", user_forecast_method_id=422,
            model_parameters="", bundled=bundled),
    )


def _route(bundled):
    """Run the dispatch with spies on both branches; return the winning branch's result.

    The losing branch raises rather than returning a sentinel, so a wrong route fails loudly
    instead of quietly producing the other mode's numbers.
    """
    def spy(name):
        def _fn(model, spark):
            assert model is not None
            return f"{name}_RESULT"
        return _fn

    def boom(name):
        def _fn(model, spark):
            raise AssertionError(f"{name} must not run for bundled={bundled}")
        return _fn

    forecasters = ({"unbundled": boom("unbundled"), "bundled": spy("BUNDLE")} if bundled
                   else {"unbundled": spy("UNBUNDLED"), "bundled": boom("bundled")})
    return run_forecast(_stub_dataset(bundled), spark=None, config=object(),
                        forecasters=forecasters)


def test_bundled_true_runs_bundle():
    """BundledInd = 1 → the bundle path."""
    assert _route(True) == "BUNDLE_RESULT"


def test_bundled_false_runs_entity_loop():
    """BundledInd = 0 → the per-entity loop."""
    assert _route(False) == "UNBUNDLED_RESULT"


# ── row_to_config coercion: what the database actually hands us ──────────────────

def _row(bundled_ind):
    """A get_user_forecast_data row with BundledInd set to the given raw value."""
    row = {
        "ForecastMethodID": 1, "Method": "ARIMA", "Parameters": "(3,3,3)",
        "Region": "LOCAL", "Status": "In Queue", "UserForecastMethodID": 421,
        "StartDate": None, "EndDate": None, "DatabrickID": 6,
    }
    if bundled_ind is not _MISSING:
        row["BundledInd"] = bundled_ind
    return row


_MISSING = object()


@pytest.mark.parametrize("raw,expected", [
    (True, True),
    (1, True),
    (False, False),
    (0, False),
    (None, False),        # NULL — tasks 1 to 4 in DEV
    (_MISSING, False),    # column absent entirely (older fixtures)
])
def test_row_to_config_coerces_bundled_ind(raw, expected):
    """True/1 → bundled; False/0/NULL/absent → unbundled, always a real bool."""
    config = row_to_config(_row(raw))
    assert config.bundled is expected


def test_null_bundled_ind_routes_unbundled():
    """The NULL case end to end: a legacy UFM row dispatches to the entity loop.

    This is the regression that matters most — a NULL misread as bundled would silently
    change the mode of every forecast predating this feature.
    """
    config = row_to_config(_row(None))
    dataset = types.SimpleNamespace(
        processed_df=None,
        ufm_config=types.SimpleNamespace(
            forecast_method_name="ARIMA",
            user_forecast_method_id=config.user_forecast_method_id,
            model_parameters="", bundled=config.bundled),
    )

    def boom(model, spark):
        raise AssertionError("a NULL BundledInd must never reach the bundle path")

    out = run_forecast(dataset, spark=None, config=object(),
                       forecasters={"unbundled": lambda m, s: "UNBUNDLED_RESULT",
                                    "bundled": boom})
    assert out == "UNBUNDLED_RESULT"


def test_forecast_pipeline_not_imported():
    """plan 01 deleted programs/pipeline.py; it must not be importable."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("programs.pipeline")
