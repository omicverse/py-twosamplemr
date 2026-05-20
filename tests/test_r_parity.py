"""R-parity tests — py-twosamplemr vs MendelianRandomization 0.10.0.

The R driver (:file:`r_reference_driver.R`) runs every estimator on the
*MendelianRandomization* package's own bundled lipid GWAS example data
(the ``ldlc`` / ``chdlodds`` vectors), and py-twosamplemr runs on the
identical input, so the two sides are directly comparable.

Closed-form estimators (IVW, MR-Egger, debiased IVW, contamination
mixture, maximum likelihood, MR-Lasso post-selection, cML-BIC /
cML-MA-BIC) are asserted to agree to ``rtol < 1e-5``.  Bootstrap-based
estimators (weighted / penalized median, mode-based estimate) cannot be
bit-exact across RNGs: their *estimates* still match closed-form to high
precision (the median location does not depend on the bootstrap), while
the bootstrap *SE* is checked to within ~12%.

Tests skip gracefully when the CMAP R env or MendelianRandomization is
unavailable.
"""
from __future__ import annotations

import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pytwosamplemr as mr
from pytwosamplemr.datasets import ldl_chd_input

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
R_DRIVER = HERE / "r_reference_driver.R"
CONDA_SH = "/home/users/steorra/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV = "/scratch/users/steorra/env/CMAP"


def _r_available() -> bool:
    if not R_DRIVER.exists():
        return False
    try:
        out = subprocess.run(
            ["bash", "-lc",
             f"source {CONDA_SH} && conda activate {CONDA_ENV} "
             "&& Rscript -e 'library(MendelianRandomization); cat(\"OK\")'"],
            capture_output=True, text=True, timeout=180, check=False,
        )
        return out.returncode == 0 and "OK" in out.stdout
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _r_available(),
    reason="CMAP R env or MendelianRandomization not installed.",
)


@pytest.fixture(scope="module")
def r_reference(tmp_path_factory):
    """Run the R driver once and return its TSV as a DataFrame."""
    out = tmp_path_factory.mktemp("rparity") / "r_reference.tsv"
    res = subprocess.run(
        ["bash", "-lc",
         f"source {CONDA_SH} && conda activate {CONDA_ENV} "
         f"&& Rscript {R_DRIVER} {out}"],
        capture_output=True, text=True, timeout=600, check=False,
    )
    assert res.returncode == 0, f"R driver failed:\n{res.stderr}"
    df = pd.read_csv(out, sep="\t")
    return df.set_index("method")


@pytest.fixture(scope="module")
def obj():
    return ldl_chd_input()


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


# --------------------------------------------------------------------------
# closed-form estimators — assert tight agreement
# --------------------------------------------------------------------------
def test_ivw_random(r_reference, obj):
    ref = r_reference.loc["ivw_random"]
    r = mr.mr_ivw(obj)
    assert _rel(r.estimate, ref["estimate"]) < 1e-6
    assert _rel(r.se, ref["se"]) < 1e-6
    assert _rel(r.pvalue, ref["pvalue"]) < 1e-5
    assert _rel(r.heter_stat, ref["extra"]) < 1e-6  # Cochran's Q


def test_ivw_fixed(r_reference, obj):
    ref = r_reference.loc["ivw_fixed"]
    r = mr.mr_ivw(obj, model="fixed")
    assert _rel(r.estimate, ref["estimate"]) < 1e-6
    assert _rel(r.se, ref["se"]) < 1e-6


def test_ivw_penalized(r_reference, obj):
    ref = r_reference.loc["ivw_penalized"]
    r = mr.mr_ivw(obj, penalized=True)
    assert _rel(r.estimate, ref["estimate"]) < 1e-6
    assert _rel(r.se, ref["se"]) < 1e-6


def test_egger(r_reference, obj):
    ref = r_reference.loc["egger"]
    r = mr.mr_egger(obj)
    assert _rel(r.estimate, ref["estimate"]) < 1e-6
    assert _rel(r.se, ref["se"]) < 1e-6
    assert _rel(r.causal_pvalue, ref["pvalue"]) < 1e-5
    assert _rel(r.i_sq, ref["extra"]) < 1e-6  # I-squared_GX


def test_egger_intercept(r_reference, obj):
    ref = r_reference.loc["egger_intercept"]
    r = mr.mr_egger(obj)
    assert _rel(r.intercept, ref["estimate"]) < 1e-6
    assert _rel(r.intercept_se, ref["se"]) < 1e-6
    assert _rel(r.pleio_pvalue, ref["pvalue"]) < 1e-5


def test_maxlik(r_reference, obj):
    """Maximum likelihood — bit-exact thanks to the R Nelder-Mead port."""
    ref = r_reference.loc["maxlik"]
    r = mr.mr_maxlik(obj)
    assert _rel(r.estimate, ref["estimate"]) < 1e-6
    assert _rel(r.se, ref["se"]) < 1e-6
    assert _rel(r.pvalue, ref["pvalue"]) < 1e-5
    assert _rel(r.heter_stat, ref["extra"]) < 1e-6


def test_divw(r_reference, obj):
    ref = r_reference.loc["divw"]
    r = mr.mr_divw(obj)
    assert _rel(r.estimate, ref["estimate"]) < 1e-6
    assert _rel(r.se, ref["se"]) < 1e-6
    assert _rel(r.condition, ref["extra"]) < 1e-6


def test_conmix(r_reference, obj):
    ref = r_reference.loc["conmix"]
    r = mr.mr_conmix(obj)
    assert _rel(r.estimate, ref["estimate"]) < 1e-6
    assert _rel(r.pvalue, ref["pvalue"]) < 1e-4
    assert _rel(r.ci_lower, ref["ci_lower"]) < 1e-4
    assert _rel(r.ci_upper, ref["ci_upper"]) < 1e-4


def test_lasso(r_reference, obj):
    """MR-Lasso — the post-selection IVW estimate is bit-exact even though
    glmnet's lambda grid differs (the het-stopping rule selects the same
    set of valid instruments)."""
    ref = r_reference.loc["lasso"]
    r = mr.mr_lasso(obj)
    assert _rel(r.estimate, ref["estimate"]) < 1e-5
    assert _rel(r.se, ref["se"]) < 1e-5
    assert r.n_valid == int(ref["extra"])


def test_cml_ma_bic(r_reference, obj):
    """cML-MA-BIC without data perturbation — deterministic, bit-exact."""
    ref = r_reference.loc["cml_ma_bic"]
    r = mr.mr_cml(obj, n=17723, ma=True, dp=False)
    assert _rel(r.estimate, ref["estimate"]) < 1e-6
    assert _rel(r.se, ref["se"]) < 1e-6


def test_cml_bic(r_reference, obj):
    """cML-BIC without data perturbation — deterministic, bit-exact."""
    ref = r_reference.loc["cml_bic"]
    r = mr.mr_cml(obj, n=17723, ma=False, dp=False)
    assert _rel(r.estimate, ref["estimate"]) < 1e-6
    assert _rel(r.se, ref["se"]) < 1e-6


# --------------------------------------------------------------------------
# median estimators — point estimate exact, bootstrap SE approximate
# --------------------------------------------------------------------------
def test_median_simple(r_reference, obj):
    ref = r_reference.loc["median_simple"]
    r = mr.mr_median(obj, weighting="simple")
    # the weighted-median location does not depend on the bootstrap RNG
    assert _rel(r.estimate, ref["estimate"]) < 1e-6
    # bootstrap SE: agree within ~12% (different RNG)
    assert _rel(r.se, ref["se"]) < 0.12


def test_median_weighted(r_reference, obj):
    ref = r_reference.loc["median_weighted"]
    r = mr.mr_median(obj, weighting="weighted")
    assert _rel(r.estimate, ref["estimate"]) < 1e-6
    assert _rel(r.se, ref["se"]) < 0.12


def test_median_penalized(r_reference, obj):
    ref = r_reference.loc["median_penalized"]
    r = mr.mr_median(obj, weighting="penalized")
    assert _rel(r.estimate, ref["estimate"]) < 1e-6
    assert _rel(r.se, ref["se"]) < 0.12


# --------------------------------------------------------------------------
# mode-based estimate — estimate exact, bootstrap SE approximate
# --------------------------------------------------------------------------
def test_mbe(r_reference, obj):
    ref = r_reference.loc["mbe"]
    r = mr.mr_mbe(obj)
    # the MBE point estimate uses no RNG -> bit-exact
    assert _rel(r.estimate, ref["estimate"]) < 1e-6
    # bootstrap SE (MAD of bootstrap replicates): within ~12%
    assert _rel(r.se, ref["se"]) < 0.12


def test_parity_summary(r_reference, obj, capsys):
    """Print a per-method R-vs-Python comparison table."""
    rows = []
    closed = {
        "ivw_random": mr.mr_ivw(obj).estimate,
        "ivw_fixed": mr.mr_ivw(obj, model="fixed").estimate,
        "egger": mr.mr_egger(obj).estimate,
        "maxlik": mr.mr_maxlik(obj).estimate,
        "divw": mr.mr_divw(obj).estimate,
        "conmix": mr.mr_conmix(obj).estimate,
        "lasso": mr.mr_lasso(obj).estimate,
        "cml_ma_bic": mr.mr_cml(obj, n=17723, ma=True, dp=False).estimate,
    }
    for m, py_est in closed.items():
        r_est = r_reference.loc[m]["estimate"]
        rows.append((m, r_est, py_est, _rel(py_est, r_est)))
    with capsys.disabled():
        print("\n  method            R-estimate   Py-estimate   rel-diff")
        for m, r, p, d in rows:
            print(f"  {m:16s} {r:11.6f} {p:13.6f}  {d:.2e}")
    assert all(d < 1e-5 for _, _, _, d in rows)
