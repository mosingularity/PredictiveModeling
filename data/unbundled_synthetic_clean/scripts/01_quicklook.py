"""01_quicklook.py — confirm the clean twin mirrors ermelo's structure, cleanly."""
import sys
from pathlib import Path
import pandas as pd, matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "fixtures" / "unbundled_synthetic_clean.csv"
FIG = HERE.parent / "figures"; FIG.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE.parents[2] / "demos" / "bundled_vs_unbundled"))   # shared deck palette
from deck_style import LPU, SPU, MILLIONS as M, apply_rcparams                  # noqa: E402

apply_rcparams()

df = pd.read_csv(SRC, parse_dates=["ReportingMonth"])
df["tou_total"] = df.PeakConsumption + df.StandardConsumption + df.OffPeakConsumption
ents = df[["EntityID", "TariffType"]].drop_duplicates().sort_values(["TariffType", "EntityID"])

fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), constrained_layout=True)
for ax, (_, r) in zip(axes.ravel(), ents.iterrows()):
    s = df[df.EntityID == r.EntityID].sort_values("ReportingMonth")
    c = LPU if r.TariffType == "LPU" else SPU
    ax.plot(s.ReportingMonth, s.tou_total, marker="o", ms=3, color=c)
    ax.set_ylim(bottom=0)
    ax.set_title(f"{r.EntityID} ({r.TariffType})", fontsize=9)
    ax.yaxis.set_major_formatter(M); ax.tick_params(axis="x", labelrotation=45, labelsize=7)
fig.suptitle("Clean synthetic twin — monthly total per entity\n"
             "all positive · winter-peaking seasonality · linear trend · slate=LPU POD teal=SPU Combo",
             fontsize=12, fontweight="bold")
fig.savefig(FIG / "00_quicklook.png", bbox_inches="tight"); plt.close(fig)
print("wrote", FIG / "00_quicklook.png")
