"""Weighted-least-squares helpers replicating R's ``lm`` summary slots.

The *MendelianRandomization* estimators lean heavily on
``summary(lm(By ~ Bx, weights = w))``.  We replicate exactly the three
slots the package consumes: the coefficient estimate, its standard error
(``coef[, 2]``) and the residual standard error (``sigma``).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = [
    "wls_no_intercept",
    "wls_with_intercept",
    "rlm_no_intercept",
    "rlm_with_intercept",
]


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


# --------------------------------------------------------------------------
# Robust variants — functional analogs of R's robustbase::lmrob
# --------------------------------------------------------------------------
# R's mr_ivw(robust=TRUE) / mr_egger(robust=TRUE) call ``lmrob`` (an MM
# estimator: S-estimate initialisation + Tukey-biweight M-step). statsmodels
# has no MM estimator, so these helpers use its M-estimator ``RLM`` with the
# Tukey-biweight norm instead. Prior IVW weights ``w`` are folded in with the
# usual sqrt-weight transform (regress sqrt(w)*y on sqrt(w)*x), so the robust
# down-weighting acts on the precision-weighted residuals — the same intent
# as ``lmrob(..., weights = w)``. Estimates are robust-equivalent to R but
# NOT bit-exact (M-estimator vs MM-estimator). Return contracts match
# ``wls_no_intercept`` / ``wls_with_intercept`` exactly.


def rlm_no_intercept(y, x, w) -> Tuple[float, float, float]:
    """Robust weighted ``lmrob(y ~ x - 1, weights = w)`` analog.

    Returns ``(coef, coef_se, sigma)`` like :func:`wls_no_intercept`, where
    ``sigma`` is the robust residual scale and ``coef_se`` the RLM standard
    error (which embeds ``sigma``, mirroring R's ``summary$coef[1, 2]``).
    """
    import statsmodels.api as sm

    y = np.asarray(y, float)
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    sw = np.sqrt(w)
    fit = sm.RLM(y * sw, (x * sw)[:, None],
                 M=sm.robust.norms.TukeyBiweight()).fit()
    return float(fit.params[0]), float(fit.bse[0]), float(fit.scale)


def rlm_with_intercept(y, x, w):
    """Robust weighted ``lmrob(y ~ x, weights = w)`` analog.

    Returns the same dict as :func:`wls_with_intercept` (``intercept`` /
    ``slope`` / their SEs and p-values / ``sigma`` / ``resid``).
    """
    from scipy import stats
    import statsmodels.api as sm

    y = np.asarray(y, float)
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    n = len(y)
    sw = np.sqrt(w)
    X = np.column_stack([np.ones(n), x])
    fit = sm.RLM(y * sw, X * sw[:, None],
                 M=sm.robust.norms.TukeyBiweight()).fit()
    beta = fit.params
    se = fit.bse
    sigma = float(fit.scale)
    df = n - 2
    tvals = beta / se
    pvals = 2.0 * stats.t.sf(np.abs(tvals), df=df) if df > 0 else np.full(2, np.nan)
    return {
        "intercept": float(beta[0]),
        "intercept_se": float(se[0]),
        "intercept_p": float(pvals[0]),
        "slope": float(beta[1]),
        "slope_se": float(se[1]),
        "slope_p": float(pvals[1]),
        "sigma": sigma,
        "resid": y - X @ beta,
    }
