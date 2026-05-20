"""``mr_allmethods`` — run the standard MR estimator panel and tabulate.

Faithful port of ``MendelianRandomization::mr_allmethods``: assembles a
tidy results table covering the simple / weighted / penalized medians,
IVW (+ penalized) and MR-Egger (estimate + intercept).
"""
from __future__ import annotations

import pandas as pd

from .core import MRInput
from .estimators import mr_egger, mr_ivw, mr_median

__all__ = ["mr_allmethods"]

_HEADINGS = ["Method", "Estimate", "Std Error", "CILower", "CIUpper", "P-value"]


def _row(method, res, use_intercept=False):
    if use_intercept:
        return [
            method, res.intercept, res.intercept_se,
            res.intercept_ci_lower, res.intercept_ci_upper,
            res.intercept_pvalue,
        ]
    return [
        method, res.estimate, res.se, res.ci_lower, res.ci_upper,
        res.pvalue if not hasattr(res, "causal_pvalue")
        else res.causal_pvalue,
    ]


def mr_allmethods(
    obj: MRInput,
    method: str = "all",
    iterations: int = 10000,
    seed: int = 314159265,
) -> pd.DataFrame:
    """Run a panel of MR estimators and return a tidy table.

    Parameters
    ----------
    obj : MRInput
        Harmonised summary statistics.
    method : {'all', 'main', 'ivw', 'median', 'egger'}
        Which subset of estimators to run.  Matches the R argument.
    iterations, seed : int
        Bootstrap settings forwarded to the median estimators.

    Returns
    -------
    pandas.DataFrame
        Columns: ``Method, Estimate, Std Error, CILower, CIUpper,
        P-value``.  The robust IVW / Egger rows of R's ``method='all'``
        are omitted because they depend on R's ``lmrob``.
    """
    if method not in ("all", "ivw", "median", "egger", "main"):
        raise ValueError("method must be one of: all, ivw, median, egger, main.")

    rows = []
    if method in ("all", "median", "main"):
        sm = mr_median(obj, weighting="simple", iterations=iterations, seed=seed)
        wm = mr_median(obj, weighting="weighted", iterations=iterations, seed=seed)
        rows.append(_row("Simple median", sm))
        rows.append(_row("Weighted median", wm))
        if method in ("all", "median"):
            pm = mr_median(obj, weighting="penalized",
                           iterations=iterations, seed=seed)
            rows.append(_row("Penalized weighted median", pm))
    if method in ("all", "ivw", "main"):
        ivw = mr_ivw(obj, model="default")
        rows.append(_row("IVW", ivw))
        if method in ("all", "ivw"):
            piv = mr_ivw(obj, model="default", penalized=True)
            rows.append(_row("Penalized IVW", piv))
    if method in ("all", "egger", "main"):
        eg = mr_egger(obj)
        rows.append(_row("MR-Egger", eg))
        rows.append(_row("(intercept)", eg, use_intercept=True))
        if method in ("all", "egger"):
            peg = mr_egger(obj, penalized=True)
            rows.append(_row("Penalized MR-Egger", peg))
            rows.append(_row("(intercept)", peg, use_intercept=True))

    return pd.DataFrame(rows, columns=_HEADINGS)
