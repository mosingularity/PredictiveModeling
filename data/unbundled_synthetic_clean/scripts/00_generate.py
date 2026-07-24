"""
00_generate.py — Clean synthetic twin of unbundled_real_ermelo.csv

Goal: a dataset with the IDENTICAL schema and the same kind of entities as the
real ermelo fixture (LPU PODs with a TOU split, SPU Combos with NonTOU only),
monthly over the same 24-month window — but generated from a KNOWN process so
every value is positive (no negatives), with clean trend + winter-peaking
seasonality + small multiplicative noise. No degenerate inactive POD.

Because the columns match ermelo exactly, this file runs through the same
bundled/unbundled SARIMA pipeline with no code change. The clean twin is the
layer that DEMONSTRATES the variance/coherence claims (the truth is known);
ermelo is the layer that shows the method survives real-world mess.

The data-generating process (the "truth" a reviewer can check):

    total_i(t) = level_i
                 * (1 + growth_i * t/12)                 # linear-ish annual trend
                 * (1 + amp_i * cos(2*pi*(t - 2)/12))    # winter peak (~Jul, t=2)
                 * exp( N(0, sigma_i) )                   # small lognormal noise -> always > 0

    LPU: total split into Peak/Standard/OffPeak by a fixed TOU ratio; NonTOU = 0.
    SPU: total goes entirely to Standard = NonTOU;  Peak = OffPeak = 0.
    Block1-4 = 0 for everyone (no PPU rows), matching the ermelo fixture.

Reproducible: fixed RNG seed. Re-run to regenerate identically.
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_CSV = HERE.parents[1] / "fixtures" / "unbundled_synthetic_clean.csv"
SEED = 20260629
rng = np.random.default_rng(SEED)

MONTHS = pd.date_range("2024-05-01", periods=24, freq="MS")  # same window as ermelo
COLS = ["EntityID", "TariffType", "TariffID", "EntityType", "ReportingMonth",
        "PeakConsumption", "StandardConsumption", "OffPeakConsumption",
        "Block1Consumption", "Block2Consumption", "Block3Consumption",
        "Block4Consumption", "NonTOUConsumption", "Scenario"]

# --- the known DGP params (documented truth) --------------------------------
# LPU PODs: TOU, high volume. tou_split = (Peak, Standard, OffPeak), sums to 1.
LPU = [
    dict(eid="1000000001.MEGAFLEX", tariff="MEGAFLEX", level=8.0e6, growth=0.12, amp=0.18, sigma=0.04, split=(0.50, 0.34, 0.16)),
    dict(eid="1000000002.MEGAFLEX", tariff="MEGAFLEX", level=5.0e6, growth=0.05, amp=0.12, sigma=0.04, split=(0.48, 0.36, 0.16)),
    dict(eid="1000000003.GENWHE",   tariff="GENWHE",   level=2.5e6, growth=0.08, amp=0.15, sigma=0.05, split=(0.52, 0.33, 0.15)),
]
# SPU Combos: NonTOU only, lower volume.
SPU = [
    dict(eid="RESSYN01LANDR1V500ZX", tariff="LANDR1", level=1.8e6, growth=0.06, amp=0.20, sigma=0.05),
    dict(eid="RESSYN02LANDR1V500ZX", tariff="LANDR1", level=0.9e6, growth=0.04, amp=0.25, sigma=0.06),
    dict(eid="RESSYN03LANDR1V500ZX", tariff="LANDR1", level=0.3e6, growth=0.10, amp=0.30, sigma=0.07),
]

t = np.arange(24)
season = np.cos(2 * np.pi * (t - 2) / 12)          # peaks at t=2 (Jul 2024), 12-mo period


def totals(p):
    trend = 1 + p["growth"] * t / 12
    noise = np.exp(rng.normal(0, p["sigma"], size=24))
    return p["level"] * trend * (1 + p["amp"] * season) * noise   # strictly > 0


rows = []
for p in LPU:
    tot = totals(p)
    pk, st, op = p["split"]
    for m, v in zip(MONTHS, tot):
        rows.append([p["eid"], "LPU", p["tariff"], "POD", m,
                     round(v * pk), round(v * st, 1), round(v * op),
                     0.0, 0.0, 0.0, 0.0, 0.0, "synthetic_clean"])
for p in SPU:
    tot = totals(p)
    for m, v in zip(MONTHS, tot):
        rows.append([p["eid"], "SPU", p["tariff"], "Combo", m,
                     0, round(v, 1), 0,
                     0.0, 0.0, 0.0, 0.0, round(v, 1), "synthetic_clean"])

df = pd.DataFrame(rows, columns=COLS)
df["ReportingMonth"] = df["ReportingMonth"].dt.strftime("%Y-%m-%d")

# --- sanity: clean twin must be all-positive & schema-identical to ermelo ----
val_cols = ["PeakConsumption", "StandardConsumption", "OffPeakConsumption",
            "Block1Consumption", "Block2Consumption", "Block3Consumption",
            "Block4Consumption", "NonTOUConsumption"]
assert (df[val_cols].values >= 0).all(), "negatives leaked into the clean twin!"
ermelo_cols = pd.read_csv(OUT_CSV.parent / "unbundled_real_ermelo.csv", nrows=0).columns.tolist()
assert df.columns.tolist() == ermelo_cols, "schema drift vs ermelo!"

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print(f"wrote {OUT_CSV}  ({len(df)} rows, {df.EntityID.nunique()} entities)")
print("min value across all consumption cols:", df[val_cols].values.min(), "(>= 0 ✓)")
print("schema matches ermelo ✓")
