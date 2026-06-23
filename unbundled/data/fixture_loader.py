import os
import pandas as pd


def load_fixture() -> pd.DataFrame:
    path = os.environ.get("PREDICTIVE_FIXTURE_PATH")
    if not path:
        raise FileNotFoundError(
            "PREDICTIVE_FIXTURE_PATH environment variable is not set. "
            "Point it to the fixture CSV/Parquet file before running."
        )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fixture file not found at PREDICTIVE_FIXTURE_PATH={path!r}. "
            "Check that the path is correct and the file exists."
        )
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)
