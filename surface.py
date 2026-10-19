"""Volatility surface construction from a listed option chain.

Pipeline, per expiry
  1. Clean quotes (drop crossed / zero bids, cap relative spread).
  2. Back out forward F and discount factor DF from put-call parity across strikes:
        C - P = DF * F - DF * K   ->  linear regression of (C - P) on K.
  3. Implied vols (bid / mid / ask) from out-of-the-money options only.
  4. Fit raw SVI total variance  w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2)),
     k = ln(K / F), weighted by the inverse bid-ask width in vol.
  5. No-arbitrage checks:
        butterfly  Gatheral's density condition g(k) >= 0 on a wide k grid
        calendar   total variance non-decreasing in maturity at every k
     Violations are removed by re-fitting with a penalty term.

All prices are per index point (multiply by the contract multiplier outside this module).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import norm

# ---------------------------------------------------------------- Black-Scholes (vectorised)

def bs_price(is_call, S, K, T, r, q, vol):
    vol = np.maximum(vol, 1e-12)
    sqT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * sqT)
    d2 = d1 - vol * sqT
    call = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return np.where(is_call, call, put)


def bs_vega(S, K, T, r, q, vol):
    sqT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * sqT)
    return S * np.exp(-q * T) * norm.pdf(d1) * sqT


def implied_vol(is_call, price, S, K, T, r, q, tol=1e-10, max_iter=100):
    """Vectorised Newton-Raphson with a bisection safeguard. NaN where no vol fits."""
    is_call, price, S, K, T, r, q = np.broadcast_arrays(is_call, price, S, K, T, r, q)
    lo = np.full(price.shape, 1e-4)
    hi = np.full(price.shape, 4.0)
    p_lo = bs_price(is_call, S, K, T, r, q, lo)
    p_hi = bs_price(is_call, S, K, T, r, q, hi)
    valid = (price > p_lo) & (price < p_hi)
    v = np.full(price.shape, 0.25)
    for _ in range(max_iter):
        p = bs_price(is_call, S, K, T, r, q, v)
        diff = p - price
        if np.all(np.abs(diff[valid]) < tol):
            break
        hi = np.where(diff > 0, v, hi)
        lo = np.where(diff <= 0, v, lo)
        vega = bs_vega(S, K, T, r, q, v)
        step = np.where(vega > 1e-12, diff / np.maximum(vega, 1e-12), 0.0)
        nxt = v - step
        bad = ~((nxt > lo) & (nxt < hi)) | (vega <= 1e-12)
        v = np.where(bad, 0.5 * (lo + hi), nxt)
    return np.where(valid, v, np.nan)

# ---------------------------------------------------------------- SVI

@dataclass
class SVI:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def w(self, k):
        x = k - self.m
        return self.a + self.b * (self.rho * x + np.sqrt(x * x + self.sigma**2))

    def dw(self, k):
        x = k - self.m
        return self.b * (self.rho + x / np.sqrt(x * x + self.sigma**2))

    def d2w(self, k):
        x = k - self.m
        return self.b * self.sigma**2 / (x * x + self.sigma**2) ** 1.5

    def g(self, k):
        """Gatheral's butterfly-arbitrage density function; must be >= 0."""
        w, w1, w2 = self.w(k), self.dw(k), self.d2w(k)
        return (1 - k * w1 / (2 * w)) ** 2 - (w1**2 / 4) * (1 / w + 0.25) + w2 / 2

    def vol(self, k, T):
        return np.sqrt(np.maximum(self.w(k), 1e-12) / T)


K_GRID = np.linspace(-1.5, 1.0, 501)   # wide log-moneyness grid for arbitrage checks


def _residuals(p, k, w_mkt, wts, prev: SVI | None, pen):
    s = SVI(*p)
    res = wts * (s.w(k) - w_mkt)
    extra = []
    if pen > 0:
        # small safety margins so the constraint holds strictly, not just to rounding
        g = s.g(K_GRID)
        extra.append(pen * np.minimum(g - 1e-4, 0.0))
        wg = s.w(K_GRID)
        extra.append(pen * np.minimum(wg - 1e-6, 0.0))
        if prev is not None:
            extra.append(pen * np.minimum(wg - prev.w(K_GRID) - 1e-6, 0.0))
    return np.concatenate([res] + extra) if extra else res


def fit_svi(k, w_mkt, wts, prev: SVI | None = None) -> tuple[SVI, dict]:
    """Fit raw SVI; re-fit with an arbitrage penalty if the free fit violates butterfly
    or calendar conditions. Several starting points guard against local minima."""
    lb = [-1.0, 1e-6, -0.999, -1.0, 1e-4]
    ub = [1.0, 5.0, 0.999, 1.0, 2.0]
    atm_w = float(np.interp(0.0, k, w_mkt)) if k.min() < 0 < k.max() else float(np.median(w_mkt))
    starts = [[atm_w * 0.5, 0.1, -0.5, 0.0, 0.1],
              [atm_w * 0.8, 0.05, -0.7, 0.05, 0.05],
              [atm_w * 0.2, 0.3, -0.3, -0.05, 0.3]]

    def solve(pen):
        best = None
        for x0 in starts:
            r = least_squares(_residuals, x0, bounds=(lb, ub), args=(k, w_mkt, wts, prev, pen),
                              method="trf", x_scale="jac", max_nfev=20000)
            if best is None or r.cost < best.cost:
                best = r
        return SVI(*best.x)

    s = solve(0.0)
    info = arb_report(s, prev)
    info["penalised_refit"] = False
    for pen in (1e3, 1e5):
        if info["arb_free"]:
            break
        s = solve(pen)
        info = arb_report(s, prev)
        info["penalised_refit"] = True
    return s, info


def arb_report(s: SVI, prev: SVI | None) -> dict:
    g_min = float(np.min(s.g(K_GRID)))
    w_min = float(np.min(s.w(K_GRID)))
    cal_min = float(np.min(s.w(K_GRID) - prev.w(K_GRID))) if prev is not None else None
    ok = g_min >= -1e-9 and w_min > 0 and (cal_min is None or cal_min >= -1e-9)
    return {"butterfly_g_min": g_min, "w_min": w_min, "calendar_dw_min": cal_min, "arb_free": bool(ok)}

# ---------------------------------------------------------------- chain processing

def clean_chain(df: pd.DataFrame, max_rel_spread=0.5) -> pd.DataFrame:
    """Expect columns: T (years), K, type ('C'/'P'), bid, ask; optional volume, oi."""
    d = df.copy()
    d = d[(d["bid"] > 0) & (d["ask"] > d["bid"])]
    mid = 0.5 * (d["bid"] + d["ask"])
    d = d[(d["ask"] - d["bid"]) / mid <= max_rel_spread]
    d["mid"] = 0.5 * (d["bid"] + d["ask"])
    return d


def parity_forward(sl: pd.DataFrame, S_guess: float) -> tuple[float, float, int]:
    """Regress (C - P) on K across strikes quoted on both sides, weighted to the money."""
    c = sl[sl["type"] == "C"].set_index("K")["mid"]
    p = sl[sl["type"] == "P"].set_index("K")["mid"]
    both = c.index.intersection(p.index)
    K = np.asarray(both, dtype=float)
    y = (c.loc[both] - p.loc[both]).to_numpy()
    w = np.exp(-((K / S_guess - 1) / 0.05) ** 2)          # weight near-the-money pairs
    A = np.vstack([np.ones_like(K), K]).T * np.sqrt(w)[:, None]
    coef, *_ = np.linalg.lstsq(A, y * np.sqrt(w), rcond=None)
    DF = -coef[1]
    F = coef[0] / DF
    return float(F), float(DF), int(len(K))


def build_surface(chain: pd.DataFrame, S: float) -> dict:
    chain = clean_chain(chain)
    out = {"S": S, "expiries": []}
    prev = None
    for T, sl in sorted(chain.groupby("T"), key=lambda x: x[0]):
        F, DF, n_pairs = parity_forward(sl, S)
        r = -np.log(DF) / T
        q = r - np.log(F / S) / T
        otm = sl[((sl["type"] == "C") & (sl["K"] >= F)) | ((sl["type"] == "P") & (sl["K"] < F))].copy()
        is_call = (otm["type"] == "C").to_numpy()
        K = otm["K"].to_numpy(float)
        for col in ["bid", "mid", "ask"]:
            otm["iv_" + col] = implied_vol(is_call, otm[col].to_numpy(), S, K, T, r, q)
        otm = otm.dropna(subset=["iv_mid"])
        k = np.log(otm["K"].to_numpy(float) / F)
        w_mkt = otm["iv_mid"].to_numpy() ** 2 * T
        iv_spread = (otm["iv_ask"] - otm["iv_bid"]).to_numpy()
        iv_spread = np.where(np.isfinite(iv_spread) & (iv_spread > 0), iv_spread, np.nanmedian(iv_spread))
        wts = 1.0 / np.maximum(iv_spread, 0.002)
        svi, arb = fit_svi(k, w_mkt, wts, prev)
        fit_vol = svi.vol(k, T)
        err = fit_vol - otm["iv_mid"].to_numpy()
        lo = otm["iv_bid"].to_numpy()
        hi = otm["iv_ask"].to_numpy()
        inside = np.where(np.isfinite(lo), (fit_vol >= lo - 1e-12) & (fit_vol <= hi + 1e-12), np.abs(err) < 1e-9)
        out["expiries"].append({
            "T": float(T), "F": F, "DF": float(DF), "r": float(r), "q": float(q), "parity_pairs": n_pairs,
            "n_quotes": int(len(otm)), "svi": asdict(svi),
            "rmse_vol_pts": float(np.sqrt(np.mean(err**2)) * 100),
            "max_abs_err_vol_pts": float(np.max(np.abs(err)) * 100),
            "pct_inside_bid_ask": float(np.mean(inside) * 100),
            "median_half_spread_vol_pts": float(np.nanmedian(iv_spread) * 50),
            **arb,
        })
        out["expiries"][-1]["_quotes"] = otm[["K", "type", "bid", "ask", "mid", "iv_bid", "iv_mid", "iv_ask"]].to_dict("list")
        prev = svi
    return out


def save_surface(surf: dict, path: str):
    clean = {"S": surf["S"], "expiries": [{k: v for k, v in e.items() if not k.startswith("_")} for e in surf["expiries"]]}
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
