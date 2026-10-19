"""Pipeline tests on the synthetic chain (known answer) and on the C++ grid (cross-check)."""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from surface import bs_price, implied_vol, build_surface, SVI, arb_report, K_GRID   # noqa: E402
from synthetic_chain import make_chain, S0                                            # noqa: E402


def test_python_bs_matches_cpp_grid():
    grid = pd.read_csv(os.path.join(os.path.dirname(HERE), "grid.csv"))
    is_call = (grid["type"] == "C").to_numpy()
    py = bs_price(is_call, 100.0, grid["K"].to_numpy(), grid["T"].to_numpy(), 0.04, 0.01, grid["vol"].to_numpy())
    assert np.max(np.abs(py - grid["price"].to_numpy())) < 1e-9


def test_iv_round_trip():
    K = np.linspace(4000, 7000, 61)
    T, vol = 0.25, 0.2
    p = bs_price(True, 5500.0, K, T, 0.02, 0.03, vol)
    iv = implied_vol(True, p, 5500.0, K, T, 0.02, 0.03)
    ok = np.isfinite(iv)
    assert ok.sum() > 50
    assert np.max(np.abs(iv[ok] - vol)) < 1e-6


def test_known_bad_svi_flagged():
    # strongly negative a with steep wings produces a negative-density region
    bad = SVI(a=-0.05, b=0.6, rho=-0.95, m=0.0, sigma=0.05)
    assert not arb_report(bad, None)["arb_free"]


def test_surface_recovers_synthetic_truth():
    chain, truth = make_chain()
    surf = build_surface(chain, S0)
    assert len(surf["expiries"]) == len(truth)
    for e in surf["expiries"]:
        t = truth[e["T"]]
        assert abs(e["F"] / t["F"] - 1) < 2e-4, (e["T"], e["F"], t["F"])
        assert abs(e["DF"] / t["DF"] - 1) < 2e-3
        assert e["arb_free"]
        # fitted vols within noise of the true surface across the quoted range
        svi = SVI(**e["svi"])
        k = np.log(np.asarray(e["_quotes"]["K"]) / e["F"])
        dev = np.abs(svi.vol(k, e["T"]) - t["svi"].vol(k, e["T"]))
        assert np.median(dev) * 100 < 0.10, np.median(dev) * 100
        assert e["pct_inside_bid_ask"] > 90


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
