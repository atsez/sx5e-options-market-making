"""Simulator invariants: attribution reconciles, uninformed flow carries no information,
no informed fills when the informed share is zero, and policies share market paths."""
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import pipeline as P                                   # noqa: E402
from surface import build_surface                      # noqa: E402
from synthetic_chain import make_chain, S0             # noqa: E402

CHAIN, _ = make_chain()
SURF = build_surface(CHAIN, S0)
BOOK = P.select_book(SURF, CHAIN, P.ASSUME["market_share"])
ATM = [b for b in BOOK if b["type"] == "C"][0]["vol"]


def sim(policy, **extra):
    a = dict(P.ASSUME, n_paths=1000)
    with tempfile.TemporaryDirectory() as d:
        cfg, out = os.path.join(d, "t.cfg"), os.path.join(d, "t.csv")
        P.write_config(cfg, S0, ATM, BOOK, P.SYNTHETIC_DYNAMICS, a, policy, 7, extra)
        subprocess.run([os.path.join(ROOT, "mmsim"), cfg, out], check=True)
        return pd.read_csv(out)


def test_attribution_reconciles():
    for pol in (1, 2, 3):
        assert sim(pol)["recon_err"].abs().max() < 1e-6


def test_uninformed_flow_has_no_markout():
    df = sim(1)
    per_fill = df["hmarkout_unf"].sum() / (df["fills"] - df["fills_inf"]).sum()
    assert abs(per_fill) < 0.1 * (df["edge"].sum() / df["fills"].sum())


def test_no_informed_fills_when_share_is_zero():
    assert sim(2, informed_frac=0.0)["fills_inf"].sum() == 0


def test_common_random_numbers():
    # with no inventory control and zero widening, policies 1 and 3 differ only through
    # widening; setting alpha = beta = 0 must reproduce policy 1 exactly
    a = sim(1)
    b = sim(3, alpha=0.0, beta=0.0, vega_limit=1e12, hard_limit_mult=1e6)
    assert np.allclose(a["total"], b["total"], atol=1e-6)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
