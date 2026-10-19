# Euro Stoxx 50 options: volatility surface and market-making simulator

A small version of what an options market-making desk runs: a C++ pricing engine, an arbitrage-free
volatility surface fitted to a real Eurex option chain, a market-making simulation with inventory risk,
adverse selection and hedging latency, and a desk-style P&L attribution that reconciles to total P&L.

## Headline results (Eurex OESX closing chain, 12 Oct 2026)

- **Surface.** Raw SVI fitted to 441 cleaned out-of-the-money quotes across five expiries. The fit sits
  inside the bid-ask on 98-100% of strikes per expiry, and every expiry passes butterfly and calendar
  arbitrage checks.
- **Inventory control.** Skewing quotes against net vega inventory cut P&L volatility 59% and mean
  maximum drawdown 65% against symmetric quoting, for a 0.2% lower mean P&L (5 seeds x 10,000 paths).
- **Adverse selection.** Each informed fill lost EUR 10.88 after hedging, against EUR 5.48 of edge per fill.
- **Attribution.** P&L split into edge, delta, gamma, theta, vega, vanna, volga and hedging cost
  reconciles to an independent cash-plus-positions total within EUR 2.2e-9 per path.
- **Pricing engine.** Analytic Greeks match finite differences to 1.9e-7 relative error, put-call parity
  holds to 2.8e-14, and the engine prices about 20 million options with full Greeks per second.

Full numbers are in [`results/results_sx5e_20261012.md`](results/results_sx5e_20261012.md), and the
pricing-engine checks are in [`validate_output.txt`](validate_output.txt).

## Layout

| File | What it does |
|---|---|
| `bs.hpp` | Black-Scholes-Merton prices, Greeks (delta, gamma, vega, theta, rho, vanna, volga), implied vol, Monte Carlo check |
| `validate.cpp` | Parity, finite-difference Greeks (Richardson), implied-vol round trip, Monte Carlo, throughput |
| `surface.py` | Chain cleaning, forward and discount factor from put-call parity, implied vols, raw SVI fit, butterfly and calendar arbitrage checks |
| `mmsim.cpp` | Market-making simulator: three quoting policies on common random numbers, delta hedging in whole futures, step-by-step attribution (OpenMP) |
| `pipeline.py` | Chain to surface to book to simulation; base case, stress tests, kill-condition sweeps, assumption sensitivity |
| `report.py` | Charts and `results.md`, every number read from the run files |
| `synthetic_chain.py` | Known-answer SSVI chain used to test the pipeline without market data |
| `tests/` | Surface and simulator invariant tests |

## Reproduce

Requires a C++17 compiler with OpenMP and Python 3 (`pip install -r requirements.txt`).

```
./run_all.sh                                        # synthetic test run, no data needed
./run_all.sh chain.csv <spot> history.csv real      # run on a real chain
```

`chain.csv` columns: `T` (years to expiry), `K`, `type` (C/P), `bid`, `ask`, `volume`.
`history.csv` columns: `date`, `spot` (SX5E close), `iv` (VSTOXX or 30-day ATM implied, vol points).

The real chain and history were pulled from a Bloomberg Terminal and are not redistributed here.
The synthetic chain runs the full pipeline end to end.

## Model in plain English

- **Surface.** For each expiry, call minus put across strikes is a straight line in strike whose
  slope is minus the discount factor and whose intercept gives the forward, so no rate or dividend
  input is needed. Out-of-the-money mids are inverted to implied vols and fitted with raw SVI,
  weighted by how tight each quote is. A fit is rejected if it implies a negative probability
  density (butterfly arbitrage) or if total variance falls with maturity (calendar arbitrage), and
  is re-fitted with a penalty.
- **Market.** The index follows a lognormal path whose volatility is the surface's ATM level plus a
  mean-reverting shock, correlated negatively with returns, minus a variance risk premium. Option
  fair values move with the index on a sticky-strike surface plus that parallel vol shift.
- **Flow.** Uninformed orders hit the desk's quotes at a rate that halves for each extra market
  half-spread of distance (Avellaneda-Stoikov intensity). Informed orders see fair value 15 minutes
  ahead and trade only when that value is outside the quotes.
- **Policies.** (1) symmetric quotes at market width; (2) quotes shifted against net vega inventory,
  reaching one half-spread of skew at the vega limit, with a hard limit; (3) policy 2 plus wider
  quotes when short-window realised vol runs above implied or recent fills mark out badly.
- **Hedging.** Net delta is hedged in whole futures with a lag (default one 5-minute step), paying
  the futures half-spread.
- **Attribution.** Each step's P&L is split into edge at fill, delta, gamma, theta, vega, vanna,
  volga, residual and hedging cost, and checked against an independent cash-plus-positions total.

## Limits

Market share of listed volume, flow decay, informed share and look-ahead, hedge latency and the
vega limit are assumptions, stated in `inputs.json` and swept in the results. Absolute P&L and
Sharpe are therefore model outputs, not forecasts. The robust results are the comparisons between
policies on identical paths and the kill conditions.
