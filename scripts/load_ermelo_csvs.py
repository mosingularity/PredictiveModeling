import sys
import pandas as pd

LPU_PATH = "data/raw/ermelo_lpu.csv"
SPU_PATH = "data/raw/ermelo_spu.csv"
OUT_PATH = "data/fixtures/unbundled_real_ermelo.parquet"

# Header-less SSMS exports — column positions inferred from data inspection.
# Cols 0-12: metadata; col 13: date; cols 14-16: TOU consumption;
# col 17: TOU total (skip — derived sum); cols 18-20: non-consumption (skip).
CSV_COLS = {
    0: "EntityID",
    1: "TariffType",
    5: "_TariffIDRaw",
    13: "ReportingMonth",
    14: "PeakConsumption",
    15: "StandardConsumption",
    16: "OffPeakConsumption",
}

ENTITY_TYPE_MAP = {"LPU": "POD", "SPU": "Combo"}


def _read_raw(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, encoding="utf-8-sig")
    return raw[list(CSV_COLS.keys())].rename(columns=CSV_COLS)


def _build_contract(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["EntityID"] = df["EntityID"].astype(str)
    out["TariffType"] = df["TariffType"].astype(str)
    # LPU col5 is NaN; extract TariffID from the suffix of EntityID (e.g. "123.GENWHE" → "GENWHE").
    # SPU col5 is the tariff code (e.g. "LANDR1").
    lpu_mask = out["TariffType"] == "LPU"
    out["TariffID"] = df["_TariffIDRaw"].astype(str)
    out.loc[lpu_mask, "TariffID"] = df.loc[lpu_mask, "EntityID"].str.split(".").str[-1]
    out["EntityType"] = out["TariffType"].map(ENTITY_TYPE_MAP)
    out["ReportingMonth"] = pd.to_datetime(df["ReportingMonth"])
    for col in ("PeakConsumption", "StandardConsumption", "OffPeakConsumption"):
        out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    out["Block1Consumption"] = 0.0
    out["Block2Consumption"] = 0.0
    out["Block3Consumption"] = 0.0
    out["Block4Consumption"] = 0.0
    # SPU Landrate 1 is a flat tariff; total kWh sits in StandardConsumption in the SQL export.
    out["NonTOUConsumption"] = out["StandardConsumption"].where(~lpu_mask, other=0.0)
    out["Scenario"] = "happy_path"
    return out


def main() -> None:
    lpu_raw = _read_raw(LPU_PATH)
    spu_raw = _read_raw(SPU_PATH)

    lpu_df = _build_contract(lpu_raw)
    spu_df = _build_contract(spu_raw)
    print(f"Loaded {len(lpu_df)} LPU rows, {len(spu_df)} SPU rows")

    combined = pd.concat([lpu_df, spu_df], ignore_index=True)
    combined["ReportingMonth"] = combined["ReportingMonth"].astype("datetime64[ns]")

    assert {"LPU", "SPU"}.issubset(set(combined["TariffType"])), (
        f"Missing TariffType values; found: {set(combined['TariffType'])}"
    )

    combined.to_parquet(OUT_PATH, index=False)
    combined.to_csv(OUT_PATH.replace(".parquet", ".csv"), index=False)
    print(f"Combined: {combined.shape}")
    print(f"TariffType counts: {combined['TariffType'].value_counts().to_dict()}")
    print(f"Written to {OUT_PATH}")


if __name__ == "__main__":
    main()
