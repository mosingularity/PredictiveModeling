import subprocess
import sys
from pathlib import Path

import pytest
import pandas as pd


FIXTURE_PATH = "data/fixtures/unbundled_predictive_input.parquet"
EXPECTED_TARIFF_TYPES = {"LPU", "SPU", "PPU"}
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_no_pyspark_or_forecast_dataset_on_import():
    # Verify importing fixture_loader pulls in neither pyspark nor ForecastDataset.
    # This must run in a fresh interpreter: other tests in the suite legitimately
    # load pyspark, so inspecting *this* process's sys.modules is unreliable
    # (ordering-dependent). A subprocess gives a clean, deterministic module state.
    code = (
        "import sys\n"
        "import unbundled.data.fixture_loader  # noqa: F401\n"
        "bad = [m for m in sys.modules if m.startswith('pyspark') or 'ForecastDataset' in m]\n"
        "assert not bad, f'forbidden modules imported by fixture_loader: {bad}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "fixture_loader import pulled in a forbidden module (pyspark/ForecastDataset):\n"
        f"{result.stderr}"
    )


def test_load_fixture_shape_and_tariff_types(monkeypatch):
    monkeypatch.setenv("PREDICTIVE_FIXTURE_PATH", FIXTURE_PATH)

    from unbundled.data.fixture_loader import load_fixture

    df = load_fixture()

    assert isinstance(df, pd.DataFrame), "load_fixture must return a pd.DataFrame"
    assert df.shape == (218, 15), f"Expected (218, 15), got {df.shape}"
    assert set(df["TariffType"].unique()) == EXPECTED_TARIFF_TYPES, (
        f"Expected tariff types {EXPECTED_TARIFF_TYPES}, "
        f"got {set(df['TariffType'].unique())}"
    )


def test_load_fixture_raises_when_env_unset(monkeypatch):
    monkeypatch.delenv("PREDICTIVE_FIXTURE_PATH", raising=False)

    from unbundled.data.fixture_loader import load_fixture

    with pytest.raises(FileNotFoundError, match="PREDICTIVE_FIXTURE_PATH"):
        load_fixture()


def test_load_fixture_raises_when_file_missing(monkeypatch):
    monkeypatch.setenv("PREDICTIVE_FIXTURE_PATH", "does/not/exist.parquet")

    from unbundled.data.fixture_loader import load_fixture

    with pytest.raises(FileNotFoundError, match="does/not/exist.parquet"):
        load_fixture()
