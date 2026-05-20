"""Weighted-least-squares helpers replicating R's ``lm`` summary slots.

The *MendelianRandomization* estimators lean heavily on
``summary(lm(By ~ Bx, weights = w))``.  We replicate exactly the three
slots the package consumes: the coefficient estimate, its standard error
(``coef[, 2]``) and the residual standard error (``sigma``).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["wls_no_intercept", "wls_with_intercept"]


def wls_no_intercept(y, x, w) -> Tuple[float, float, float]:
    """Weighted ``lm(y ~ x - 1)``.

    Returns ``(coef, coef_se, sigma)`` where ``sigma`` is the residual
    standard error ``sqrt(RSS_w / df)`` and ``coef_se`` is the standard
    error R reports in ``summary(...)$coef[1, 2]``.
    """
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    n = len(y)
    sw_xx = np.sum(w * x * x)
    coef = np.sum(w * x * y) / sw_xx
    resid = y - coef * x
    df = n - 1
    rss = np.sum(w * resid ** 2)
    sigma = np.sqrt(rss / df) if df > 0 else float("nan")
    coef_se = sigma / np.sqrt(sw_xx)
    return float(coef), float(coef_se), float(sigma)


def wls_with_intercept(y, x, w):
    """Weighted ``lm(y ~ x)`` with intercept.

    Returns a dict with ``intercept``, ``intercept_se``, ``slope``,
    ``slope_se``, ``sigma`` and the two-sided p-values (t-distribution
    with ``n - 2`` df) for both coefficients, mirroring the columns of
    ``summary(lm(...))$coef``.
    """
    from scipy import stats

    y = np.asarray(y, float)
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    W = np.diag(w)
    XtWX = X.T @ W @ X
    XtWX_inv = np.linalg.inv(XtWX)
    beta = XtWX_inv @ X.T @ W @ y
    resid = y - X @ beta
    df = n - 2
    rss = np.sum(w * resid ** 2)
    sigma = np.sqrt(rss / df) if df > 0 else float("nan")
    cov = XtWX_inv * sigma ** 2
    se = np.sqrt(np.diag(cov))
    tvals = beta / se
    pvals = 2.0 * stats.t.sf(np.abs(tvals), df=df)
    return {
        "intercept": float(beta[0]),
        "intercept_se": float(se[0]),
        "intercept_p": float(pvals[0]),
        "slope": float(beta[1]),
        "slope_se": float(se[1]),
        "slope_p": float(pvals[1]),
        "sigma": float(sigma),
        "resid": resid,
    }
