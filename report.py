"""Build charts and a numbers sheet from results/<run>/. Every figure in results.md is
read from summary.csv, surface.json or inputs.json; nothing is typed by hand.

    python3 report.py synthetic
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from surface import SVI

POL = {1: ("Symmetric", "#2a78d6"), 2: ("Vega-skewed", "#eb6834"), 3: ("Skewed + adaptive width", "#1baf7a")}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "legend.frameon": False})


def zero_cross(x, y):
    """Linear interpolation of where y first crosses zero; None if it never does."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    for i in range(len(x) - 1):
        if y[i] > 0 >= y[i + 1]:
            return float(x[i] + (x[i + 1] - x[i]) * y[i] / (y[i] - y[i + 1]))
    return None


def main(run):
    d = os.path.join("results", run)
    r = pd.read_csv(os.path.join(d, "summary.csv"))
    surf = json.load(open(os.path.join(d, "surface.json")))
    inp = json.load(open(os.path.join(d, "inputs.json")))
    days = inp["assumptions"]["days"]

    # ---------------- chart 1: surface fit, small multiples
    exps = surf["expiries"]
    fig, axes = plt.subplots(1, len(exps), figsize=(2.6 * len(exps), 2.6), sharey=True)
    quotes = json.load(open(os.path.join(d, "quotes.json"))) if os.path.exists(os.path.join(d, "quotes.json")) else None
    for ax, e, qi in zip(np.atleast_1d(axes), exps, (quotes or [None] * len(exps))):
        svi = SVI(**e["svi"])
        kk = np.linspace(-0.3, 0.15, 200)
        ax.plot(kk, svi.vol(kk, e["T"]) * 100, color=POL[1][1], lw=2, label="SVI fit")
        if qi:
            k = np.log(np.asarray(qi["K"]) / e["F"])
            ax.vlines(k, np.asarray(qi["iv_bid"]) * 100, np.asarray(qi["iv_ask"]) * 100, color=INK2, lw=1, label="bid-ask IV")
        ax.set_title(f"{e['T'] * 12:.0f}M  RMSE {e['rmse_vol_pts']:.2f} vol pts", fontsize=8.5, color=INK)
        ax.set_xlabel("log-moneyness ln(K/F)")
    np.atleast_1d(axes)[0].set_ylabel("implied vol (%)")
    np.atleast_1d(axes)[0].legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "chart_surface.png"), dpi=160)
    plt.close(fig)

    base = r[r["run"].str.startswith("base")]
    bm = base.groupby("policy").mean(numeric_only=True)

    # ---------------- chart 2: P&L distribution per policy (seed 0 base runs)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for p, (name, col) in POL.items():
        pp = pd.read_csv(os.path.join(d, f"base_p{p}_s0.csv"))["total"]
        ax.hist(pp, bins=120, histtype="step", lw=2, color=col, density=True, label=name)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlabel(f"P&L per {days}-day run (EUR)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.set_title("Same edge, far less variance once quotes lean against inventory", fontsize=9.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "chart_pnl_distribution.png"), dpi=160)
    plt.close(fig)

    # ---------------- chart 3: kill-condition sweeps
    sweeps = [("sweep_phi", "informed_frac", "informed share of flow", 100, "%"),
              ("sweep_cost", "fut_half_spread", "futures half-spread (index pts)", 1, ""),
              ("sweep_lag", "hedge_lag", "hedge latency (minutes)", 5, "")]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9), sharey=True)
    for ax, (pre, col, lab, scale, _) in zip(axes, sweeps):
        s = r[r["run"].str.startswith(pre)]
        for p, (name, c) in POL.items():
            sp = s[s["policy"] == p].sort_values(col)
            ax.plot(sp[col] * scale, sp["mean_pnl"], color=c, lw=2, marker="o", ms=4, label=name)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xlabel(lab)
    axes[0].set_ylabel(f"mean P&L per {days}-day run (EUR)")
    axes[0].legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "chart_kill_sweeps.png"), dpi=160)
    plt.close(fig)

    # ---------------- chart 4: tail loss under stress
    scen = [("base_p{p}_s0", "Base"), ("vol_shock_p{p}", "Vol +5 pts"), ("gap_down_p{p}", "Gap -5%"), ("informed_x2_p{p}", "Informed x2")]
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    w = 0.26
    for j, (p, (name, c)) in enumerate(POL.items()):
        vals = [float(r.loc[r["run"] == pat.format(p=p), "p5_pnl"].iloc[0]) for pat, _ in scen]
        ax.bar(np.arange(len(scen)) + (j - 1) * w, vals, width=w - 0.02, color=c, label=name)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(np.arange(len(scen)), [s for _, s in scen])
    ax.set_ylabel("5th percentile P&L (EUR)")
    ax.legend(fontsize=7.5, ncol=3, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "chart_stress_tail.png"), dpi=160)
    plt.close(fig)

    # ---------------- numbers sheet
    L = [f"# Results: {run}", "", f"Vol dynamics source: {inp['dynamics']['source']}", ""]
    L += ["## Surface fit", "", "| T (yrs) | F | DF | quotes | RMSE (vol pts) | inside bid-ask | arb-free | penalised refit |", "|---|---|---|---|---|---|---|---|"]
    for e in exps:
        L.append(f"| {e['T']:.3f} | {e['F']:.2f} | {e['DF']:.5f} | {e['n_quotes']} | {e['rmse_vol_pts']:.3f} | {e['pct_inside_bid_ask']:.1f}% | {e['arb_free']} | {e['penalised_refit']} |")
    L += ["", "## Base case, mean of 5 seeds x 10,000 paths", "",
          "| metric | " + " | ".join(POL[p][0] for p in POL) + " |", "|---|---|---|---|"]
    rows = [("mean P&L (EUR)", "mean_pnl", "{:,.0f}"), ("std P&L (EUR)", "std_pnl", "{:,.0f}"),
            ("5th pct P&L (EUR)", "p5_pnl", "{:,.0f}"), ("paths profitable", "pct_profitable", "{:.1f}%"),
            ("mean max drawdown (EUR)", "mean_max_dd", "{:,.0f}"), ("95th pct max drawdown (EUR)", "p95_max_dd", "{:,.0f}"),
            ("fills per day", "fills_per_day", "{:.1f}"), ("edge per fill (EUR)", "edge_per_fill", "{:.2f}"),
            ("desk markout per informed fill (EUR)", "desk_markout_per_informed_fill", "{:.2f}"),
            ("desk markout per uninformed fill (EUR)", "desk_markout_per_uninformed_fill", "{:.2f}"),
            ("mean abs net vega (EUR/vol pt)", "mean_abs_vega", "{:,.0f}"), ("95th pct max abs vega", "p95_max_abs_vega", "{:,.0f}"),
            ("attr edge", "attr_edge", "{:,.0f}"), ("attr delta (latency)", "attr_delta", "{:,.0f}"),
            ("attr gamma", "attr_gamma", "{:,.0f}"), ("attr theta", "attr_theta", "{:,.0f}"), ("attr vega", "attr_vega", "{:,.0f}"),
            ("attr vanna", "attr_vanna", "{:,.0f}"), ("attr volga", "attr_volga", "{:,.0f}"),
            ("attr residual", "attr_residual", "{:,.2f}"), ("hedging cost", "attr_cost", "{:,.0f}"),
            ("max abs reconciliation error (EUR)", "max_abs_recon_err", "{:.1e}")]
    for lab, col, f in rows:
        L.append(f"| {lab} | " + " | ".join(f.format(bm.loc[p, col]) for p in POL) + " |")
    L.append("| seed-to-seed std of mean P&L | " + " | ".join(f"{base[base.policy == p]['mean_pnl'].std():.1f}" for p in POL) + " |")
    p1, p2, p3 = bm.loc[1], bm.loc[2], bm.loc[3]
    L += ["", "## Headline comparisons", "",
          f"- Vega skew vs symmetric: P&L std {100 * (p2.std_pnl / p1.std_pnl - 1):+.0f}%, mean max drawdown {100 * (p2.mean_max_dd / p1.mean_max_dd - 1):+.0f}%, mean abs vega {100 * (p2.mean_abs_vega / p1.mean_abs_vega - 1):+.0f}%, mean P&L {100 * (p2.mean_pnl / p1.mean_pnl - 1):+.1f}%",
          f"- Adaptive width vs symmetric: mean P&L {100 * (p3.mean_pnl / p1.mean_pnl - 1):+.1f}%, fills {100 * (p3.fills_per_day / p1.fills_per_day - 1):+.0f}%, edge per fill {100 * (p3.edge_per_fill / p1.edge_per_fill - 1):+.0f}%, Sharpe {p1.sharpe_ann:.1f} -> {p3.sharpe_ann:.1f}",
          f"- Latency cost: each informed fill loses EUR {-p1.desk_markout_per_informed_fill:.2f} after hedging, against EUR {p1.edge_per_fill:.2f} edge per fill"]
    L += ["", "## Stress tests (seed 0)", "", "| scenario | policy | mean P&L | 5th pct P&L | 95th pct max DD |", "|---|---|---|---|---|"]
    for pat, sname in scen:
        for p in POL:
            x = r[r["run"] == pat.format(p=p)].iloc[0]
            L.append(f"| {sname} | {POL[p][0]} | {x.mean_pnl:,.0f} | {x.p5_pnl:,.0f} | {x.p95_max_dd:,.0f} |")
    L += ["", "## Kill conditions (mean P&L crosses zero)", ""]
    for pre, col, lab, scale, _ in sweeps:
        s = r[r["run"].str.startswith(pre)]
        for p in POL:
            sp = s[s["policy"] == p].sort_values(col)
            z = zero_cross(sp[col], sp["mean_pnl"])
            L.append(f"- {lab}, {POL[p][0]}: " + (f"break-even at {z * scale:.2f}" if z is not None else
                     f"no break-even in tested range (P&L {sp['mean_pnl'].iloc[0]:,.0f} -> {sp['mean_pnl'].iloc[-1]:,.0f})"))
    L += ["", "## Assumption sensitivity (mean P&L, seed 0)", ""]
    for pre, lab in [("sens_k", "flow decay multiplier k"), ("sens_share", "market share")]:
        s = r[r["run"].str.startswith(pre)].copy()
        s["v"] = s["run"].str.extract(r"(?:k|share)([\d.]+)_p")[0]
        for v, grp in s.groupby("v"):
            L.append(f"- {lab} {v}: " + ", ".join(f"{POL[p][0]} {grp[grp.policy == p]['mean_pnl'].iloc[0]:,.0f}" for p in POL))
    open(os.path.join(d, "results.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "synthetic")
