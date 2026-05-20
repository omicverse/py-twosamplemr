"""Smoke tests for py-twosamplemr — every public estimator / diagnostic
runs and returns sane, finite output on the bundled example data.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import pytwosamplemr as mr
from pytwosamplemr.datasets import (
    chdlodds,
    chdloddsse,
    example_inputs,
    ldl_chd_input,
    ldlc,
    ldlcse,
)

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def obj():
    return ldl_chd_input()


def test_mr_input_construction():
    o = mr.mr_input(bx=ldlc, bxse=ldlcse, by=chdlodds, byse=chdloddsse)
    assert o.nsnps == 28
    assert len(o.snps) == 28
    assert o.snps[0] == "snp_1"
    assert len(o) == 28


def test_mr_input_validation():
    with pytest.raises(ValueError):
        mr.mr_input(bx=[1, 2], bxse=[1], by=[1, 2], byse=[1, 2])


def test_ivw(obj):
    r = mr.mr_ivw(obj)
    assert np.isfinite(r.estimate) and np.isfinite(r.se)
    assert r.ci_lower < r.estimate < r.ci_upper
    assert 0 <= r.pvalue <= 1
    assert r.model == "random"
    # fixed effects
    rf = mr.mr_ivw(obj, model="fixed")
    assert rf.model == "fixed"
    assert abs(rf.estimate - r.estimate) < 1e-9
    assert rf.se < r.se


def test_ivw_penalized(obj):
    r = mr.mr_ivw(obj, penalized=True)
    assert r.penalized and np.isfinite(r.estimate)


def test_robust_ivw_egger(obj):
    # robust variants run and stay finite; robust down-weighting must not
    # blow the estimate far from the standard fit on outlier-free data
    ri = mr.mr_ivw(obj, robust=True)
    re = mr.mr_egger(obj, robust=True)
    assert ri.robust and np.isfinite(ri.estimate) and np.isfinite(ri.se)
    assert re.robust and np.isfinite(re.estimate) and np.isfinite(re.intercept)
    assert abs(ri.estimate - mr.mr_ivw(obj).estimate) < 0.5 * abs(
        mr.mr_ivw(obj).estimate)


def test_robust_rejects_correlated():
    o = example_inputs()["ldl"]
    n = o.nsnps
    oc = mr.mr_input(bx=o.betaX, bxse=o.betaXse, by=o.betaY, byse=o.betaYse,
                     correlation=np.eye(n))
    with pytest.raises(ValueError, match="correlated"):
        mr.mr_ivw(oc, robust=True)


def test_ivw_correlated():
    o = example_inputs()["ldl"]
    n = o.nsnps
    rng = np.random.default_rng(0)
    A = rng.normal(size=(n, n))
    corr = np.eye(n) * 0.999 + 0.001 * (A @ A.T) / n
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    oc = mr.mr_input(bx=o.betaX, bxse=o.betaXse, by=o.betaY, byse=o.betaYse,
                     correlation=corr)
    r = mr.mr_ivw(oc)
    assert r.correlation and np.isfinite(r.estimate)


def test_egger(obj):
    r = mr.mr_egger(obj)
    assert np.isfinite(r.estimate)
    assert np.isfinite(r.intercept)
    assert 0 <= r.i_sq <= 1
    assert 0 <= r.intercept_pvalue <= 1


def test_egger_too_few_snps():
    o = mr.mr_input(bx=[0.1, 0.2], bxse=[0.01, 0.02],
                    by=[0.2, 0.3], byse=[0.02, 0.03])
    with pytest.raises(ValueError):
        mr.mr_egger(o)


@pytest.mark.parametrize("weighting", ["simple", "weighted", "penalized"])
def test_median(obj, weighting):
    r = mr.mr_median(obj, weighting=weighting, iterations=500)
    assert np.isfinite(r.estimate) and r.se > 0
    assert r.weighting == weighting


def test_mbe(obj):
    r = mr.mr_mbe(obj, iterations=300)
    assert np.isfinite(r.estimate) and r.se > 0


def test_maxlik(obj):
    r = mr.mr_maxlik(obj)
    assert np.isfinite(r.estimate) and r.se > 0


def test_divw(obj):
    r = mr.mr_divw(obj)
    assert np.isfinite(r.estimate) and r.se > 0
    rn = mr.mr_divw(obj, over_dispersion=False)
    assert np.isfinite(rn.estimate)


def test_conmix(obj):
    r = mr.mr_conmix(obj)
    assert np.isfinite(r.estimate)
    assert len(r.valid) > 0
    assert r.ci_lower <= r.estimate <= r.ci_upper


def test_lasso(obj):
    r = mr.mr_lasso(obj)
    assert np.isfinite(r.estimate) and r.se > 0
    assert r.n_valid >= 2


def test_cml(obj):
    r = mr.mr_cml(obj, n=17723, ma=True, dp=False)
    assert np.isfinite(r.estimate) and r.se > 0
    rb = mr.mr_cml(obj, n=17723, ma=False, dp=False)
    assert np.isfinite(rb.estimate)


def test_cml_dp(obj):
    r = mr.mr_cml(obj, n=17723, ma=True, dp=True, num_pert=30)
    assert np.isfinite(r.estimate)
    assert np.isfinite(r.gof1_pvalue)


def test_allmethods(obj):
    for method in ["all", "main", "ivw", "median", "egger"]:
        df = mr.mr_allmethods(obj, method=method, iterations=300)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "Estimate" in df.columns


def test_heterogeneity(obj):
    df = mr.mr_heterogeneity(obj)
    assert len(df) == 2
    assert (df["Q"] > 0).all()


def test_pleiotropy(obj):
    df = mr.mr_pleiotropy_test(obj)
    assert len(df) == 1
    assert "egger_intercept" in df.columns


def test_singlesnp(obj):
    df = mr.mr_singlesnp(obj)
    assert len(df) == obj.nsnps + 2
    assert df.iloc[-2]["SNP"].startswith("All - Inverse")


def test_leaveoneout(obj):
    df = mr.mr_leaveoneout(obj)
    assert len(df) == obj.nsnps + 1
    assert df.iloc[-1]["SNP"] == "All"


def test_steiger():
    res = mr.mr_steiger(
        p_exp=[1e-20, 1e-15, 1e-10],
        p_out=[0.4, 0.5, 0.6],
        n_exp=10000, n_out=10000,
    )
    assert res["correct_causal_direction"] is True
    assert 0 <= res["steiger_test"] <= 1


def test_harmonise():
    exposure = pd.DataFrame({
        "SNP": ["rs1", "rs2", "rs3", "rs4"],
        "beta": [0.1, 0.2, -0.15, 0.05],
        "se": [0.01, 0.02, 0.015, 0.008],
        "effect_allele": ["A", "C", "G", "A"],
        "other_allele": ["G", "T", "A", "T"],
        "eaf": [0.3, 0.4, 0.6, 0.2],
    })
    outcome = pd.DataFrame({
        "SNP": ["rs1", "rs2", "rs3", "rs4"],
        "beta": [0.05, -0.1, 0.08, 0.02],
        "se": [0.01, 0.012, 0.011, 0.009],
        # rs2 flipped alleles; rs4 palindromic A/T
        "effect_allele": ["A", "T", "G", "A"],
        "other_allele": ["G", "C", "A", "T"],
        "eaf": [0.3, 0.6, 0.6, 0.2],
    })
    h = mr.harmonise_data(exposure, outcome)
    assert len(h) == 4
    # rs2 had alleles swapped -> outcome beta sign flipped
    rs2 = h[h["SNP"] == "rs2"].iloc[0]
    assert rs2["beta.outcome"] > 0
    assert "mr_keep" in h.columns


def test_directionality_test():
    exposure = pd.DataFrame({
        "SNP": ["rs1", "rs2", "rs3"],
        "beta": [0.3, 0.25, 0.28],
        "se": [0.01, 0.01, 0.01],
        "effect_allele": ["A", "C", "G"],
        "other_allele": ["G", "T", "A"],
        "eaf": [0.5, 0.5, 0.5],
    })
    outcome = pd.DataFrame({
        "SNP": ["rs1", "rs2", "rs3"],
        "beta": [0.02, 0.01, 0.03],
        "se": [0.01, 0.01, 0.01],
        "effect_allele": ["A", "C", "G"],
        "other_allele": ["G", "T", "A"],
        "eaf": [0.5, 0.5, 0.5],
    })
    h = mr.harmonise_data(exposure, outcome)
    d = mr.directionality_test(h, n_exp=5000, n_out=5000)
    assert d.iloc[0]["correct_causal_direction"]


def test_plots(obj):
    import matplotlib
    matplotlib.use("Agg")
    for fn in [mr.mr_scatter, mr.mr_forest, mr.mr_funnel, mr.mr_loo]:
        ax = fn(obj)
        assert ax is not None


def test_repr(obj):
    r = mr.mr_ivw(obj)
    s = repr(r)
    assert "IVW" in s and "estimate" in s
