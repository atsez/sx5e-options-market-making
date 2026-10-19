"""Synthetic Euro Stoxx 50-style option chain, used to test the pipeline end to end.
The true surface is SSVI (Gatheral-Jacquier), which is free of static arbitrage by
construction and whose slices are exact raw-SVI curves, so the fitter has a known answer
to recover. It also lets the repo run without proprietary market data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from surface import bs_price, SVI

S0, R, Q = 5500.0, 0.022, 0.028
EXPIRIES = np.array([21, 42, 63, 126, 189]) / 252.0      # ~1, 2, 3, 6, 9 months
RHO, ETA, GAMMA = -0.70, 1.20, 0.45                        # SSVI skew parameters


def atm_vol(T):
    return 0.18 - 0.03 * np.exp(-T / 0.25)                 # upward-sloping ATM term structure


def ssvi_slice(T) -> SVI:
    theta = atm_vol(T) ** 2 * T
    phi = ETA / (theta**GAMMA * (1 + theta) ** (1 - GAMMA))
    return SVI(a=theta / 2 * (1 - RHO**2), b=theta * phi / 2, rho=RHO,
               m=-RHO / phi, sigma=np.sqrt(1 - RHO**2) / phi)


def make_chain(seed=7, noise_vol_pts=0.03, tick=0.1) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    rows, truth = [], {}
    for T in EXPIRIES:
        F = S0 * np.exp((R - Q) * T)
        svi = ssvi_slice(T)
        truth[float(T)] = {"F": F, "DF": float(np.exp(-R * T)), "svi": svi}
        for K in np.arange(np.floor(0.75 * F / 25) * 25, 1.15 * F, 25.0):
            k = np.log(K / F)
            vol = svi.vol(k, T) + rng.normal(0, noise_vol_pts / 100)
            half = (0.08 + 0.30 * abs(k)) / 100 * np.sqrt(0.25 / T) ** 0.5
            vol_bid, vol_ask = max(vol - half, 0.01), vol + half
            for typ in ("C", "P"):
                is_call = typ == "C"
                bid = np.floor(bs_price(is_call, S0, K, T, R, Q, vol_bid) / tick) * tick
                ask = np.ceil(bs_price(is_call, S0, K, T, R, Q, vol_ask) / tick) * tick
                if ask < tick:
                    continue
                vol_mid = 0.5 * (vol_bid + vol_ask)
                volume = int(rng.poisson(400 * np.exp(-abs(k) / 0.05) * np.exp(-T)) )
                rows.append({"T": float(T), "K": float(K), "type": typ, "bid": float(bid),
                             "ask": float(ask), "volume": volume, "true_vol": float(vol)})
    return pd.DataFrame(rows), truth
