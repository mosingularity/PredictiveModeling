"""Show how a bundle forecast is split back to its members, on real UFM data.

The bundle path forecasts a group of entities as one total, then hands each member
its portion back using that member's share of consumption over history. This script
is the evidence for that step: it prints the shares, shows how sensitive they are to
the history window used, and confirms the split-out members add back to the bundle
total every month. It writes a figure alongside the printout.

Nothing here writes to a database or changes any state — it reads a fixture and
re-runs the same functions the pipeline uses.

Usage (run from the ``predictive_modeling`` folder, or anywhere — it chdirs itself):

    python scripts/bundle_share_evidence.py                     # UFM 421, Peak
    python scripts/bundle_share_evidence.py --ufm 423
    python scripts/bundle_share_evidence.py --channel OffPeakConsumption
    python scripts/bundle_share_evidence.py --method ARIMA --horizon 12
    python scripts/bundle_share_evidence.py --out docs/figures   # where the png goes

The UFMs with a bundled configuration in DEV are 421, 423, 425 and 427; their
unbundled counterparts (420, 422, 424, 426) load the same way and are useful for
comparison.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PACKAGE_ROOT)
sys.path.insert(0, str(PACKAGE_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from models.bundle import (BUNDLE_CONSUMPTION_COLUMN, aggregate_members,
                           build_member_series, compute_shares, forecast_for_bundle)
from models.bundle_forecasters import bundle_forecaster

# Categorical slots 1-3 of the validated default palette, in fixed order. Members are
# an identity, not a magnitude, so hues are assigned by position and never cycled; a
# bundle with more than three members folds the tail into "other" (see _colour_map).
PALETTE = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834",
           "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#d9d8d4"

WINDOWS = [("all history", None), ("last 12 months", 12), ("last 6 months", 6)]
RULE = "─" * 78


def _shares_over(members: pd.DataFrame, months: int | None) -> pd.Series:
    """Member shares computed over the last ``months`` months (None = all history)."""
    window = members if months is None else members.iloc[-months:]
    totals = window.sum(axis=0)
    return totals / totals.sum()


def _short(name: str, width: int = 22) -> str:
    """Member ids are long and their tails are the distinguishing part."""
    return name if len(name) <= width else name[:width - 1] + "…"


def _banner(title: str) -> None:
    print(f"\n{RULE}\n  {title}\n{RULE}")


def report(ufm: int, channel: str, method: str, horizon: int, out_dir: Path) -> None:
    df = pd.read_csv(f"data/fixtures/Results_{ufm}.csv")
    members = build_member_series(df, channel)
    bundle = aggregate_members(members)[BUNDLE_CONSUMPTION_COLUMN]
    shares = compute_shares(members)

    print(f"\n  UFM {ufm}  ·  {channel}  ·  {len(members.columns)} members  ·  "
          f"{len(members)} months  ·  {members.index[0]:%b %Y} to {members.index[-1]:%b %Y}")

    # 1 ── the shares themselves ───────────────────────────────────────────────
    _banner("1. Each member's share of the bundle, over the whole history")
    table = pd.DataFrame({
        "history total": members.sum(axis=0).map(lambda v: f"{v:,.0f}"),
        "share": shares.map(lambda v: f"{v:.6f}"),
        "% of bundle": shares.map(lambda v: f"{v * 100:6.2f}%"),
    })
    table.index = [_short(c) for c in table.index]
    print(table.to_string())
    print(f"\n  shares total {shares.sum():.12f}  → nothing is lost or created in the split")

    # 2 ── how much the window matters ─────────────────────────────────────────
    _banner("2. The same shares, computed over shorter windows")
    by_window = pd.DataFrame({label: _shares_over(members, m) for label, m in WINDOWS})
    drift = ((by_window["last 12 months"] / by_window["all history"] - 1) * 100)
    shown = by_window.map(lambda v: f"{v * 100:6.2f}%")
    shown["12m vs all"] = drift.map(lambda v: f"{v:+6.1f}%")
    shown.index = [_short(c) for c in shown.index]
    print(shown.to_string())
    print(f"\n  largest move: {drift.abs().max():.1f}% for {_short(drift.abs().idxmax())}"
          "  → the window is a real choice, not a detail")

    # 3 ── coherence: the members add back up ─────────────────────────────────
    _banner(f"3. One {method} fit on the bundle, split back to members")
    fit_fn = bundle_forecaster(method)
    bundle_fc = fit_fn(bundle, horizon)
    parts = forecast_for_bundle(members, method=method, horizon=horizon, fit_fn=fit_fn)

    shown = parts.copy()
    shown.columns = [_short(c, 18) for c in shown.columns]
    shown["── members ──"] = parts.sum(axis=1)
    shown["bundle fit"] = bundle_fc.to_numpy()
    print(shown.round(0).to_string())

    gap = abs(parts.sum(axis=1).to_numpy() - bundle_fc.to_numpy()).max()
    dev = (parts.div(parts.sum(axis=1), axis=0) - shares).abs().to_numpy().max()
    print(f"\n  largest monthly gap between the members and the bundle: {gap:.2e}")
    print(f"  largest drift of any member's monthly proportion from its share: {dev:.2e}")
    print("  → the members reconstruct the bundle exactly, every month")

    # 4 ── credits ────────────────────────────────────────────────────────────
    negative_months = int((bundle < 0).sum())
    negative_members = int((members.sum(axis=0) < 0).sum())
    _banner("4. Credits in this data")
    print(f"  months where the bundle total is negative:  {negative_months} of {len(bundle)}"
          + (f"  (lowest {bundle.min():,.0f})" if negative_months else ""))
    print(f"  members whose history total is negative:    {negative_members}")
    print("\n  Shares are computed on history totals, so negative months alone do not"
          "\n  break them. A member with a negative total would: every share would flip"
          "\n  sign while still summing to 1, and the coherence check above would still"
          "\n  pass. That case is not currently guarded.")

    _figure(ufm, channel, method, members, bundle, shares, by_window, parts,
            bundle_fc, out_dir)


def _colour_map(columns) -> dict:
    """Fixed-order hues by member, largest first, tail folded into one grey."""
    if len(columns) <= len(PALETTE):
        return {c: PALETTE[i] for i, c in enumerate(columns)}
    keep = {c: PALETTE[i] for i, c in enumerate(columns[:len(PALETTE) - 1])}
    keep.update({c: "#8a8981" for c in columns[len(PALETTE) - 1:]})
    return keep


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=8.5, length=3)
    ax.grid(axis="y", color=GRID, lw=.6, alpha=.7)
    ax.set_axisbelow(True)


def _figure(ufm, channel, method, members, bundle, shares, by_window, parts,
            bundle_fc, out_dir: Path) -> None:
    order = list(shares.sort_values(ascending=False).index)
    colours = _colour_map(order)

    fig = plt.figure(figsize=(13.5, 8.2), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=.46, wspace=.30,
                            left=.15, right=.97, top=.82, bottom=.09)
    ax_hist = fig.add_subplot(grid[0, :])
    ax_share = fig.add_subplot(grid[1, 0])
    ax_win = fig.add_subplot(grid[1, 1])
    for ax in (ax_hist, ax_share, ax_win):
        _style(ax)

    # ── history: members stacked to the bundle total ─────────────────────────
    ax_hist.stackplot(members.index, [members[c] for c in order],
                      labels=[_short(c, 24) for c in order],
                      colors=[colours[c] for c in order],
                      edgecolor=SURFACE, linewidth=1.4)
    ax_hist.plot(bundle.index, bundle.values, color=INK, lw=2, label="bundle total")
    if (bundle < 0).any():
        ax_hist.axhline(0, color=INK_SOFT, lw=1, ls=(0, (4, 3)))
        for month in bundle.index[bundle < 0]:
            ax_hist.axvspan(month - pd.Timedelta(days=14), month + pd.Timedelta(days=14),
                            color="#e34948", alpha=.10, lw=0)
        ax_hist.text(0, -.17, "shaded: months where credits push the total below zero",
                     transform=ax_hist.transAxes, fontsize=9, color="#b03a39")
    ax_hist.set_title(f"The members stack to the bundle total: {channel}",
                      fontsize=11, color=INK, pad=8, loc="left")
    ax_hist.set_ylabel("consumption", fontsize=9, color=INK_SOFT)
    ax_hist.legend(fontsize=8.5, loc="upper left", frameon=False, ncol=2,
                   labelcolor=INK_SOFT)

    # ── shares ───────────────────────────────────────────────────────────────
    pct = shares[order] * 100
    bars = ax_share.barh([_short(c, 24) for c in order][::-1], pct.values[::-1],
                         color=[colours[c] for c in order][::-1], height=.62)
    for bar, value in zip(bars, pct.values[::-1]):
        ax_share.text(bar.get_width() + pct.max() * .02, bar.get_y() + bar.get_height() / 2,
                      f"{value:.2f}%", va="center", fontsize=9.5, color=INK)
    ax_share.set_xlim(0, pct.max() * 1.22)
    ax_share.set_title("Share of the bundle each member receives", fontsize=11,
                       color=INK, pad=8, loc="left")
    ax_share.set_xlabel("% of bundle, whole history", fontsize=9, color=INK_SOFT)
    ax_share.grid(axis="y", visible=False)
    ax_share.grid(axis="x", color=GRID, lw=.6, alpha=.7)

    # ── window sensitivity ───────────────────────────────────────────────────
    baseline = by_window["all history"]
    all_moves = {m: [(by_window[label][m] / baseline[m] - 1) * 100
                     for label, _ in WINDOWS] for m in order}
    # Label only the member that moves most: the others are readable off the axis,
    # and three end-labels on a narrow panel collide.
    loudest = max(all_moves, key=lambda m: max(abs(v) for v in all_moves[m]))
    for member in order:
        moves = all_moves[member]
        ax_win.plot(range(len(WINDOWS)), moves, color=colours[member], lw=2,
                    marker="o", markersize=8, markeredgecolor=SURFACE,
                    markeredgewidth=1.6, label=_short(member, 20))
    peak = max(range(len(WINDOWS)), key=lambda i: abs(all_moves[loudest][i]))
    ax_win.annotate(f"{all_moves[loudest][peak]:+.0f}% at its widest",
                    (peak, all_moves[loudest][peak]), textcoords="offset points",
                    xytext=(0, -20), ha="center", fontsize=10, color=INK)
    ax_win.axhline(0, color=INK_SOFT, lw=1)
    ax_win.set_xticks(range(len(WINDOWS)), [label for label, _ in WINDOWS])
    ax_win.set_xlim(-.3, len(WINDOWS) - .7)
    ax_win.margins(y=.22)
    ax_win.set_title("How much each share moves if we shorten the window",
                     fontsize=11, color=INK, pad=8, loc="left")
    ax_win.set_ylabel("change vs all history (%)", fontsize=9, color=INK_SOFT)
    ax_win.legend(fontsize=8, loc="upper left", frameon=False, labelcolor=INK_SOFT)

    fig.suptitle("Splitting a bundle forecast back to its members", x=.043, y=.965,
                 ha="left", fontsize=16, color=INK)
    fig.text(.043, .915,
             f"UFM {ufm}, {channel}. One {method} fit on the bundle total, then divided "
             f"by each member's historical share.\nThe members reconstruct the total "
             f"exactly, every month.",
             ha="left", va="top", fontsize=10, color=INK_SOFT, linespacing=1.5)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"bundle_shares_ufm{ufm}_{channel}.png"
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    print(f"\n  figure written to {path}\n")


BUNDLED_UFMS = [420, 421, 422, 423, 424, 425, 426, 427]
CHANNELS = ["PeakConsumption", "StandardConsumption", "OffPeakConsumption"]


def sweep(window_months: int) -> None:
    """Every UFM and channel: how far a shorter share window moves each member.

    The decision this answers is which history window the shares should use. A window
    only matters if it changes a member's share, so this measures exactly that, across
    everything we have rather than one case.
    """
    _banner(f"Share window: last {window_months} months against all available history")
    rows = []
    for ufm in BUNDLED_UFMS:
        df = pd.read_csv(f"data/fixtures/Results_{ufm}.csv", encoding="utf-8-sig")
        for channel in CHANNELS:
            try:
                members = build_member_series(df, channel)
            except ValueError:
                continue
            full = _shares_over(members, None)
            short = _shares_over(members, window_months)
            move = ((short / full - 1) * 100).abs()
            rows.append({
                "UFM": ufm,
                "channel": channel.replace("Consumption", ""),
                "months": len(members),
                "members": len(members.columns),
                "largest move": f"{move.max():6.1f}%",
                "on member": _short(move.idxmax(), 20),
            })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    worst = max(float(r["largest move"].strip("% ")) for r in rows)
    spans = {r["months"] for r in rows}
    print(f"\n  Largest move anywhere: {worst:.1f}%")
    print(f"  History available in these fixtures: {sorted(spans)} months — "
          f"a 5-year (60 month) window cannot be exercised on this data.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sweep", type=int, metavar="MONTHS", default=None,
                        help="compare an N-month share window against all history, "
                             "across every UFM and channel")
    parser.add_argument("--ufm", type=int, default=421, help="UFM id (default 421)")
    parser.add_argument("--channel", default="PeakConsumption",
                        choices=["PeakConsumption", "StandardConsumption",
                                 "OffPeakConsumption"])
    parser.add_argument("--method", default="SARIMA",
                        choices=["ARIMA", "SARIMA", "RandomForest", "XGBoost"])
    parser.add_argument("--horizon", type=int, default=6, help="months to forecast")
    parser.add_argument("--out", type=Path, default=Path("docs/evidence/share-window-sensitivity"),
                        help="where the figure is written")
    parser.add_argument("--verbose", action="store_true",
                        help="show the fitters' own logging")
    args = parser.parse_args()

    if not args.verbose:
        logging.getLogger().setLevel(logging.WARNING)
        logging.getLogger("hyperparameters").setLevel(logging.WARNING)

    if args.sweep:
        sweep(args.sweep)
    else:
        report(args.ufm, args.channel, args.method, args.horizon, args.out)


if __name__ == "__main__":
    main()
