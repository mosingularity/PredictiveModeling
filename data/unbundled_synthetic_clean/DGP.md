# Clean synthetic twin — data-generating process (the known truth)

**File**: `data/fixtures/unbundled_synthetic_clean.csv` (peer of `unbundled_real_ermelo.csv`)
**Generator**: `scripts/00_generate.py` · **Seed**: 20260629 · re-run to regenerate identically.

## Why this exists

The real ermelo data can't *demonstrate* the statistical claims (does bundling lower
aggregate forecast variance? does reconciliation hold?) because its variance is dominated
by data-quality artifacts — negatives, zeros, an inactive POD, 5-orders-of-magnitude scale
spread. This file has the **same schema and entity structure** as ermelo but is generated
from a process we control, so the theoretical answer is computable and the method can be
shown to recover it. The clean twin proves the mechanism; ermelo proves robustness.

## Schema

Identical to `unbundled_real_ermelo.csv` (asserted in the generator). 24 months,
2024-05-01 → 2026-04-01 monthly, 6 entities, `Scenario = synthetic_clean`.
Block1-4 all zero (no PPU rows), matching ermelo.

## The process

For entity *i*, month index *t* = 0…23:

```
total_i(t) = level_i
             * (1 + growth_i * t/12)              # linear annual trend
             * (1 + amp_i * cos(2π(t-2)/12))      # winter peak at t=2 (Jul), period 12
             * exp( ε ),   ε ~ N(0, sigma_i)       # small lognormal noise → strictly > 0
```

- **LPU (POD, TOU)**: `total` split into Peak/Standard/OffPeak by a fixed ratio; NonTOU = 0.
- **SPU (Combo, NonTOU)**: `total` → Standard = NonTOU; Peak = OffPeak = 0.

All values are guaranteed positive (lognormal noise can't cross zero) — the generator
asserts `min >= 0` before writing.

## Known parameters

| EntityID | Type | level | growth/yr | seas amp | noise σ | TOU split (Pk/St/Op) |
|---|---|---|---|---|---|---|
| 1000000001.MEGAFLEX | LPU POD | 8.0M | 12% | 0.18 | 0.04 | 0.50 / 0.34 / 0.16 |
| 1000000002.MEGAFLEX | LPU POD | 5.0M | 5%  | 0.12 | 0.04 | 0.48 / 0.36 / 0.16 |
| 1000000003.GENWHE   | LPU POD | 2.5M | 8%  | 0.15 | 0.05 | 0.52 / 0.33 / 0.15 |
| RESSYN01LANDR1V500ZX | SPU Combo | 1.8M | 6%  | 0.20 | 0.05 | — (NonTOU) |
| RESSYN02LANDR1V500ZX | SPU Combo | 0.9M | 4%  | 0.25 | 0.06 | — (NonTOU) |
| RESSYN03LANDR1V500ZX | SPU Combo | 0.3M | 10% | 0.30 | 0.07 | — (NonTOU) |

## What it deliberately leaves OUT (vs ermelo)

No negatives, no all-zero/inactive POD, no duplicate rows per month, no missing months.
Those live in the ermelo layer on purpose — that's where "survives reality" is demonstrated.

## How the deck uses it

- Aggregate the members → bundle `B(t) = Σ total_i(t)`.
- **Variance claim**: `Var(B) = 1ᵀΣ1` — the cross-entity covariance term is *why* the bundled
  path can beat the unbundled one; with a known Σ this is checkable, not asserted.
- **Coherence claim**: disaggregate `B̂` by known historical shares → reconciliation residual
  closes to ~0 by construction.
