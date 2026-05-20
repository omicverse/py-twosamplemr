"""pytwosamplemr: pure-Python two-sample Mendelian randomization.

A standalone, dependency-light port of the statistical-methods core of
the R/CRAN package *MendelianRandomization* (Yavorska & Burgess,
*Int. J. Epidemiol.* 2017) together with the harmonisation /
diagnostics workflow of *TwoSampleMR* (Hemani et al., *eLife* 2018).

Numerical parity with the R reference (*MendelianRandomization* 0.10.0)
is the design goal: closed-form estimators (IVW, MR-Egger, debiased
IVW, contamination mixture, maximum likelihood, MR-Lasso post-selection,
cML-BIC) reproduce R bit-for-bit, including a faithful port of R's
Nelder-Mead optimiser used by the maximum-likelihood estimator.
Bootstrap-based estimators (weighted median, mode-based estimate, cML
with data perturbation) agree to within Monte-Carlo error.

Core data structure
-------------------
* :class:`MRInput` / :func:`mr_input` — the harmonised SNP-exposure /
  SNP-outcome summary-statistics container.

Estimators
----------
* :func:`mr_ivw`     — inverse-variance weighted (fixed / random,
  penalized, correlated SNPs, Cochran's Q).
* :func:`mr_egger`   — MR-Egger regression (causal estimate +
  directional-pleiotropy intercept + I-squared_GX).
* :func:`mr_median`  — simple / weighted / penalized weighted median.
* :func:`mr_mbe`     — mode-based estimate.
* :func:`mr_maxlik`  — maximum-likelihood estimator.
* :func:`mr_divw`    — debiased IVW.
* :func:`mr_conmix`  — contamination mixture.
* :func:`mr_lasso`   — MR-Lasso post-selection estimator.
* :func:`mr_cml`     — constrained maximum likelihood (cML-MA / cML-BIC).
* :func:`mr_allmethods` — run the standard panel and tabulate.

Diagnostics / workflow
----------------------
* :func:`harmonise_data`      — align SNP-exposure / SNP-outcome alleles.
* :func:`mr_heterogeneity`    — Cochran's Q per method.
* :func:`mr_pleiotropy_test`  — MR-Egger intercept test.
* :func:`mr_steiger` / :func:`directionality_test` — Steiger
  directionality test.
* :func:`mr_singlesnp`        — per-SNP Wald ratios.
* :func:`mr_leaveoneout`      — leave-one-out analysis.

Plots
-----
* :func:`mr_plot` / :func:`mr_scatter`, :func:`mr_forest`,
  :func:`mr_funnel`, :func:`mr_loo`.

Quick-start
-----------
>>> import pytwosamplemr as mr
>>> from pytwosamplemr.datasets import ldl_chd_input
>>> obj = ldl_chd_input()
>>> mr.mr_ivw(obj)
IVW: estimate=2.8342 se=0.5298 95% CI [1.7958, 3.8726] p=8.815e-08  (28 SNPs)
>>> mr.mr_egger(obj).intercept
-0.011460665651289616
>>> mr.mr_allmethods(obj, method="main")
"""
from __future__ import annotations

from .allmethods import mr_allmethods
from .core import MRInput, ci_normal, ci_t, egger_bounds, mr_input
from .estimators import (
    mr_cml,
    mr_conmix,
    mr_divw,
    mr_egger,
    mr_ivw,
    mr_lasso,
    mr_maxlik,
    mr_mbe,
    mr_median,
    weighted_median,
    weighted_median_boot_se,
)
from .harmonise import (
    directionality_test,
    harmonise_data,
    mr_heterogeneity,
    mr_leaveoneout,
    mr_pleiotropy_test,
    mr_singlesnp,
    mr_steiger,
)
from .plotting import mr_forest, mr_funnel, mr_loo, mr_plot, mr_scatter
from .results import (
    CMLResult,
    ConMixResult,
    DIVWResult,
    EggerResult,
    IVWResult,
    LassoResult,
    MaxLikResult,
    MBEResult,
    MedianResult,
    MRResult,
)

__version__ = "0.1.0"

__all__ = [
    # data structure
    "MRInput",
    "mr_input",
    # estimators
    "mr_ivw",
    "mr_egger",
    "mr_median",
    "mr_mbe",
    "mr_maxlik",
    "mr_divw",
    "mr_conmix",
    "mr_lasso",
    "mr_cml",
    "mr_allmethods",
    "weighted_median",
    "weighted_median_boot_se",
    # diagnostics / workflow
    "harmonise_data",
    "mr_heterogeneity",
    "mr_pleiotropy_test",
    "mr_steiger",
    "directionality_test",
    "mr_singlesnp",
    "mr_leaveoneout",
    # plots
    "mr_plot",
    "mr_scatter",
    "mr_forest",
    "mr_funnel",
    "mr_loo",
    # result containers
    "MRResult",
    "IVWResult",
    "EggerResult",
    "MedianResult",
    "MBEResult",
    "MaxLikResult",
    "DIVWResult",
    "ConMixResult",
    "LassoResult",
    "CMLResult",
    # helpers
    "ci_normal",
    "ci_t",
    "egger_bounds",
]
