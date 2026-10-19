"""End-to-end pipeline: chain -> surface -> book selection and flow calibration ->
market-making simulation for three policies -> attribution, stress tests, kill conditions.

    python3 pipeline.py --synthetic                      # test run on the synthetic chain
    python3 pipeline.py --chain chain.csv --spot 5512.3 --history history.csv

Every number reported comes from files written into results/<run name>/.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

from surface import build_surface, save_surface, SVI

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "mmsim")

# ------------------------------------------------------------------ assumptions (stated, swept)
ASSUME = {
    "market_share": 0.10,      # share of listed daily volume the simulated desk could win at market width
    "k_mult": 1.0,             # flow at mid = 2x flow at market width when k_mult = 1
    "informed_frac": 0.10,     # share of baseline arrivals that are informed
    "horizon_steps": 3,        # informed look-ahead: 3 x 5 minutes = 15 minutes
    "hedge_lag": 1,            # fills hedged one step (5 minutes) later
    "vega_limit": 2000.0,      # EUR per vol point
    "steps_per_day": 78,       # 5-minute steps
    "days": 5,
    "n_paths": 10000,
    "fut_half_spread": 0.5,    # index points (one-tick-wide futures market)
}
# vol dynamics: replaced by calibrate_history() when real history is supplied
SYNTHETIC_DYNAMICS = {"kappa": 4.0, "eta": 0.20, "rho_sv": -0.70, "vrp": 0.025, "source": "SYNTHETIC placeholder"}


def calibrate_history(hist: pd.DataFrame) -> dict:
    """hist columns: date, spot (SX5E), iv (V2X or 30d ATM implied, in vol points)."""
    h = hist.dropna().sort_values("date")
    ret = np.log(h["spot"]).diff()
    rv20 = ret.rolling(20).std() * np.sqrt(252)
    iv = h["iv"] / 100.0
    fwd_rv = rv20.shift(-20)                              # realised over the next 20 days
    vrp = float(np.nanmean(iv - fwd_rv))
    div = iv.diff()
    eta = float(np.nanstd(div) * np.sqrt(252))
    rho = float(pd.concat([ret, div], axis=1).dropna().corr().iloc[0, 1])
    b = np.polyfit(iv.iloc[:-1].to_numpy(), iv.iloc[1:].to_numpy(), 1)[0]   # AR(1)
    kappa = float(-np.log(max(min(b, 0.9999), 1e-6)) * 252)
    return {"kappa": kappa, "eta": eta, "rho_sv": rho, "vrp": vrp,
            "source": f"history {h['date'].iloc[0]} to {h['date'].iloc[-1]}, {len(h)} days"}


def select_book(surf: dict, chain: pd.DataFrame, market_share: float) -> list[dict]:
    """1-month and 3-month expiries; 25-delta put, ATM call, 25-delta call on each."""
    exps = surf["expiries"]
    picks = []
    for target in (21 / 252, 63 / 252):
        e = min(exps, key=lambda e: abs(e["T"] - target))
        svi = SVI(**e["svi"])
        qd = pd.DataFrame(e["_quotes"])
        qd["k"] = np.log(qd["K"] / e["F"])
        vol = svi.vol(qd["k"].to_numpy(), e["T"])
        sqT = np.sqrt(e["T"])
        d1 = (-qd["k"] + 0.5 * vol**2 * e["T"]) / (vol * sqT)
        qd["delta_c"] = np.exp(-e["q"] * e["T"]) * norm.cdf(d1)
        qd["delta_p"] = qd["delta_c"] - np.exp(-e["q"] * e["T"])
        calls, puts = qd[qd["type"] == "C"], qd[qd["type"] == "P"]
        legs = [("P", puts.iloc[(puts["delta_p"] + 0.25).abs().argmin()]),
                ("C", calls.iloc[(calls["k"]).abs().argmin()]),
                ("C", calls.iloc[(calls["delta_c"] - 0.25).abs().argmin()])]
        sl = chain[np.isclose(chain["T"], e["T"])]
        for typ, row in legs:
            vol_k = float(svi.vol(np.log(row["K"] / e["F"]), e["T"]))
            half = float((row["iv_ask"] - row["iv_bid"]) / 2 * 100)
            vrow = sl[(sl["K"] == row["K"]) & (sl["type"] == typ)]
            volume = float(vrow["volume"].iloc[0]) if len(vrow) and "volume" in vrow else 0.0
            picks.append({"type": typ, "K": float(row["K"]), "T": e["T"], "r": e["r"], "q": e["q"],
                          "vol": vol_k, "half_vol_pts": half, "volume": volume,
                          "lambda0": max(market_share * volume / 2, 1.0)})
    return picks


def write_config(path, S0, atm_ref, book, dyn, a, policy, seed, extra=None):
    lines = [f"S0 {S0}", f"atm_ref {atm_ref}", f"policy {policy}", f"seed {seed}",
             f"steps_per_day {a['steps_per_day']}", f"days {a['days']}", f"n_paths {a['n_paths']}",
             f"kappa {dyn['kappa']}", f"eta {dyn['eta']}", f"rho_sv {dyn['rho_sv']}", f"vrp {dyn['vrp']}",
             f"k_mult {a['k_mult']}", f"informed_frac {a['informed_frac']}", f"horizon {a['horizon_steps']}",
             f"vega_limit {a['vega_limit']}", f"fut_half_spread {a['fut_half_spread']}", f"hedge_lag {a['hedge_lag']}"]
    for k, v in (extra or {}).items():
        lines.append(f"{k} {v}")
    for b in book:
        lines.append(f"inst {b['type']} {b['K']} {b['T']} {b['r']} {b['q']} {b['vol']} {b['half_vol_pts']} {b['lambda0']}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def summarise(df: pd.DataFrame, days: int) -> dict:
    t = df["total"]
    comps = ["edge", "delta", "gamma", "theta", "vega", "vanna", "volga", "residual", "cost"]
    out = {
        "mean_pnl": t.mean(), "std_pnl": t.std(), "sharpe_ann": t.mean() / t.std() * np.sqrt(252 / days),
        "p5_pnl": t.quantile(0.05), "pct_profitable": (t > 0).mean() * 100,
        "mean_max_dd": df["max_dd"].mean(), "p95_max_dd": df["max_dd"].quantile(0.95),
        "fills_per_day": df["fills"].mean() / days, "informed_fill_share": df["fills_inf"].sum() / df["fills"].sum() * 100,
        "markout_per_informed_fill": df["markout_inf"].sum() / max(df["fills_inf"].sum(), 1),
        "markout_per_uninformed_fill": df["markout_unf"].sum() / max((df["fills"] - df["fills_inf"]).sum(), 1),
        "desk_markout_per_informed_fill": df["hmarkout_inf"].sum() / max(df["fills_inf"].sum(), 1),
        "desk_markout_per_uninformed_fill": df["hmarkout_unf"].sum() / max((df["fills"] - df["fills_inf"]).sum(), 1),
        "edge_per_fill": df["edge"].sum() / df["fills"].sum(),
        "mean_abs_vega": df["mean_abs_vega"].mean(), "p95_max_abs_vega": df["max_abs_vega"].quantile(0.95),
        "max_abs_recon_err": df["recon_err"].abs().max(),
    }
    for c in comps:
        out["attr_" + c] = df[c].mean()
    return {k: float(v) for k, v in out.items()}


def run(cfg_dir, name, S0, atm_ref, book, dyn, a, policy, seed=20261012, extra=None) -> dict:
    cfg = os.path.join(cfg_dir, f"{name}.cfg")
    out = os.path.join(cfg_dir, f"{name}.csv")
    write_config(cfg, S0, atm_ref, book, dyn, a, policy, seed, extra)
    subprocess.run([SIM, cfg, out], check=True)
    s = summarise(pd.read_csv(out), a["days"])
    s.update({"run": name, "policy": policy, "seed": seed, **(extra or {})})
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--chain")
    ap.add_argument("--spot", type=float)
    ap.add_argument("--history")
    ap.add_argument("--name", default=None)
    ap.add_argument("--quick", action="store_true", help="fewer paths, for development")
    args = ap.parse_args()

    if args.synthetic:
        from synthetic_chain import make_chain, S0
        chain, _ = make_chain()
        spot, name = S0, args.name or "synthetic"
    else:
        chain = pd.read_csv(args.chain)
        spot, name = args.spot, args.name or "real"
    outdir = os.path.join(HERE, "results", name)
    os.makedirs(outdir, exist_ok=True)
    a = dict(ASSUME)
    if args.quick:
        a["n_paths"] = 2000

    t0 = time.time()
    surf = build_surface(chain, spot)
    save_surface(surf, os.path.join(outdir, "surface.json"))
    json.dump([e["_quotes"] for e in surf["expiries"]], open(os.path.join(outdir, "quotes.json"), "w"))
    dyn = calibrate_history(pd.read_csv(args.history)) if args.history else dict(SYNTHETIC_DYNAMICS)
    book = select_book(surf, chain, a["market_share"])
    atm_ref = [b for b in book if b["type"] == "C"][0]["vol"]
    json.dump({"assumptions": a, "dynamics": dyn, "book": book, "atm_ref": atm_ref},
              open(os.path.join(outdir, "inputs.json"), "w"), indent=2)

    rows = []
    N = a["steps_per_day"] * a["days"]
    # base case, three policies, five seeds for seed-to-seed variation
    for pol in (1, 2, 3):
        for s in range(5):
            rows.append(run(outdir, f"base_p{pol}_s{s}", spot, atm_ref, book, dyn, a, pol, seed=1000 + s))
    # stress tests (seed 0)
    stresses = {"vol_shock": {"stress_step": N // 2, "vol_shock": 0.05},
                "gap_down": {"stress_step": N // 2, "gap_move": -0.05},
                "informed_x2": {"informed_frac": 2 * a["informed_frac"]}}
    for sname, extra in stresses.items():
        for pol in (1, 2, 3):
            rows.append(run(outdir, f"{sname}_p{pol}", spot, atm_ref, book, dyn, a, pol, seed=1000, extra=extra))
    # kill-condition sweeps
    for phi in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
        for pol in (1, 2, 3):
            rows.append(run(outdir, f"sweep_phi{phi}_p{pol}", spot, atm_ref, book, dyn, a, pol, seed=1000,
                            extra={"informed_frac": phi}))
    for hs in (0.5, 1.0, 2.0, 4.0, 8.0):
        for pol in (1, 2, 3):
            rows.append(run(outdir, f"sweep_cost{hs}_p{pol}", spot, atm_ref, book, dyn, a, pol, seed=1000,
                            extra={"fut_half_spread": hs}))
    for lag in (0, 1, 2, 3, 6):
        for pol in (1, 2, 3):
            rows.append(run(outdir, f"sweep_lag{lag}_p{pol}", spot, atm_ref, book, dyn, a, pol, seed=1000,
                            extra={"hedge_lag": lag}))
    # assumption sensitivity (flow decay and market share)
    for km in (0.5, 2.0):
        for pol in (1, 2, 3):
            rows.append(run(outdir, f"sens_k{km}_p{pol}", spot, atm_ref, book, dyn, a, pol, seed=1000,
                            extra={"k_mult": km}))
    for ms in (0.05, 0.20):
        b2 = [dict(b, lambda0=max(ms * b["volume"] / 2, 1.0)) for b in book]
        for pol in (1, 2, 3):
            rows.append(run(outdir, f"sens_share{ms}_p{pol}", spot, atm_ref, b2, dyn, a, pol, seed=1000))

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(outdir, "summary.csv"), index=False)
    print(f"done in {time.time() - t0:.0f}s, {len(res)} runs -> {outdir}/summary.csv")


if __name__ == "__main__":
    main()
