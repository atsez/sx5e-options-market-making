# Results: sx5e_20261012

Vol dynamics source: history 2024-10-11 to 2026-10-12, 506 days

## Surface fit

| T (yrs) | F | DF | quotes | RMSE (vol pts) | inside bid-ask | arb-free | penalised refit |
|---|---|---|---|---|---|---|---|
| 0.107 | 6210.63 | 0.99705 | 78 | 0.049 | 98.7% | True | False |
| 0.184 | 6214.44 | 0.99491 | 85 | 0.054 | 100.0% | True | True |
| 0.260 | 6227.78 | 0.99285 | 92 | 0.059 | 100.0% | True | True |
| 0.433 | 6251.30 | 0.98756 | 100 | 0.051 | 98.0% | True | True |
| 0.682 | 6197.58 | 0.98025 | 86 | 0.101 | 100.0% | True | True |

## Base case, mean of 5 seeds x 10,000 paths

| metric | Symmetric | Vega-skewed | Skewed + adaptive width |
|---|---|---|---|
| mean P&L (EUR) | 1,188 | 1,186 | 1,325 |
| std P&L (EUR) | 3,277 | 1,329 | 1,310 |
| 5th pct P&L (EUR) | -4,105 | -998 | -823 |
| paths profitable | 68.8% | 82.4% | 85.5% |
| mean max drawdown (EUR) | 3,441 | 1,193 | 1,155 |
| 95th pct max drawdown (EUR) | 8,494 | 2,402 | 2,357 |
| fills per day | 72.1 | 72.4 | 62.6 |
| edge per fill (EUR) | 5.48 | 5.39 | 6.53 |
| desk markout per informed fill (EUR) | -10.88 | -10.86 | -11.37 |
| desk markout per uninformed fill (EUR) | 0.00 | 0.02 | 0.01 |
| mean abs net vega (EUR/vol pt) | 719 | 262 | 261 |
| 95th pct max abs vega | 2,943 | 1,125 | 1,115 |
| attr edge | 1,975 | 1,951 | 2,045 |
| attr delta (latency) | -169 | -165 | -167 |
| attr gamma | 21 | 10 | 10 |
| attr theta | -32 | -11 | -12 |
| attr vega | -152 | -160 | -153 |
| attr vanna | -31 | -17 | -17 |
| attr volga | 15 | 7 | 7 |
| attr residual | 0.04 | 0.04 | 0.04 |
| hedging cost | 438 | 429 | 387 |
| max abs reconciliation error (EUR) | 2.2e-09 | 1.3e-09 | 1.5e-09 |
| seed-to-seed std of mean P&L | 38.8 | 16.6 | 14.4 |

## Headline comparisons

- Vega skew vs symmetric: P&L std -59%, mean max drawdown -65%, mean abs vega -64%, mean P&L -0.2%
- Adaptive width vs symmetric: mean P&L +11.5%, fills -13%, edge per fill +19%, Sharpe 2.6 -> 7.2
- Latency cost: each informed fill loses EUR 10.88 after hedging, against EUR 5.48 edge per fill

## Stress tests (seed 0)

| scenario | policy | mean P&L | 5th pct P&L | 95th pct max DD |
|---|---|---|---|---|
| Base | Symmetric | 1,213 | -4,119 | 8,557 |
| Base | Vega-skewed | 1,191 | -1,003 | 2,393 |
| Base | Skewed + adaptive width | 1,326 | -804 | 2,328 |
| Vol +5 pts | Symmetric | 1,297 | -6,581 | 13,317 |
| Vol +5 pts | Vega-skewed | 1,216 | -2,353 | 4,175 |
| Vol +5 pts | Skewed + adaptive width | 1,310 | -2,158 | 4,217 |
| Gap -5% | Symmetric | 1,019 | -8,562 | 12,755 |
| Gap -5% | Vega-skewed | 900 | -4,223 | 5,883 |
| Gap -5% | Skewed + adaptive width | 958 | -3,925 | 5,669 |
| Informed x2 | Symmetric | 960 | -4,801 | 9,271 |
| Informed x2 | Vega-skewed | 946 | -1,298 | 2,628 |
| Informed x2 | Skewed + adaptive width | 1,141 | -1,152 | 2,578 |

## Kill conditions (mean P&L crosses zero)

- informed share of flow, Symmetric: break-even at 44.21
- informed share of flow, Vega-skewed: break-even at 44.86
- informed share of flow, Skewed + adaptive width: no break-even in tested range (P&L 1,477 -> 360)
- futures half-spread (index pts), Symmetric: break-even at 1.88
- futures half-spread (index pts), Vega-skewed: break-even at 1.89
- futures half-spread (index pts), Skewed + adaptive width: break-even at 2.21
- hedge latency (minutes), Symmetric: no break-even in tested range (P&L 1,386 -> 894)
- hedge latency (minutes), Vega-skewed: no break-even in tested range (P&L 1,359 -> 876)
- hedge latency (minutes), Skewed + adaptive width: no break-even in tested range (P&L 1,471 -> 1,037)

## Assumption sensitivity (mean P&L, seed 0)

- flow decay multiplier k 0.5: Symmetric 1,213, Vega-skewed 1,186, Skewed + adaptive width 1,464
- flow decay multiplier k 2.0: Symmetric 1,213, Vega-skewed 1,200, Skewed + adaptive width 1,075
- market share 0.05: Symmetric 649, Vega-skewed 621, Skewed + adaptive width 701
- market share 0.2: Symmetric 2,356, Vega-skewed 2,333, Skewed + adaptive width 2,593
