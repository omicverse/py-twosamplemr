"""Two-sample Mendelian-randomization estimators.

Faithful pure-Python ports of the *MendelianRandomization* (CRAN 0.10.0)
estimator suite: IVW, MR-Egger, median (simple / weighted / penalized),
mode-based (MBE), maximum-likelihood, debiased-IVW, contamination
mixture, MR-Lasso and constrained maximum likelihood (cML).

Every routine reproduces the corresponding R S4 method line-for-line so
that closed-form estimators are bit-exact and bootstrap-based ones agree
to within Monte-Carlo error.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ._nmmin import nmmin
from ._wls import wls_no_intercept, wls_with_intercept
from .core import MRInput, ci_normal, ci_t, egger_bounds
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
)

__all__ = [
    "mr_ivw",
    "mr_egger",
    "mr_median",
    "mr_mbe",
    "mr_maxlik",
    "mr_divw",
    "mr_conmix",
    "mr_lasso",
    "mr_cml",
    "weighted_median",
    "weighted_median_boot_se",
]

_RDEFAULT_SEED = 314159265


# --------------------------------------------------------------------------
# IVW
# --------------------------------------------------------------------------
def mr_ivw(
    obj: MRInput,
    model: str = "default",
    robust: bool = False,
    penalized: bool = False,
    weights: str = "simple",
    psi: float = 0.0,
    correl: bool = False,
    distribution: str = "normal",
    alpha: float = 0.05,
) -> IVWResult:
    """Inverse-variance weighted estimator (port of ``mr_ivw``).

    Supports fixed / random effects, the ``penalized`` weighting scheme,
    and correlated SNPs via ``obj.correlation``.  The ``robust`` variant
    (which requires R's ``lmrob``) is not ported; passing ``robust=True``
    raises :class:`NotImplementedError`.
    """
    Bx = obj.betaX
    By = obj.betaY
    Bxse = obj.betaXse
    Byse = obj.betaYse
    rho = obj.correlation
    nsnps = len(Bx)

    if model == "default":
        model = "fixed" if nsnps < 4 else "random"
    if rho is not None:
        correl = True
    if robust:
        raise NotImplementedError(
            "robust IVW relies on R's robustbase::lmrob and is not ported."
        )
    if model not in ("random", "fixed"):
        raise ValueError("model must be one of: default, random, fixed.")
    if distribution not in ("normal", "t-dist"):
        raise ValueError("distribution must be one of: normal, t-dist.")

    if correl and rho is not None:
        omega = np.outer(Byse, Byse) * rho
        omega_inv = np.linalg.inv(omega)
        A = Bx @ omega_inv @ Bx
        theta = float((Bx @ omega_inv @ By) / A)
        rse_vec = By - theta * Bx
        if model == "random":
            se = np.sqrt(1.0 / A) * max(
                np.sqrt(rse_vec @ omega_inv @ rse_vec / (nsnps - 1)), 1.0
            )
        else:
            se = np.sqrt(1.0 / A)
        rse = float(np.sqrt(rse_vec @ omega_inv @ rse_vec / (nsnps - 1)))
        heter = (nsnps - 1) * rse ** 2
        heter_p = stats.chi2.sf(heter, df=nsnps - 1)
        chol = np.linalg.cholesky(rho).T  # R chol() is upper-triangular
        fstat = float(np.sum(((Bx / Bxse) @ np.linalg.inv(chol)) ** 2) / nsnps)
        correlation_flag = True
    else:
        if nsnps == 1:
            theta = float(By[0] / Bx[0])
            if weights == "delta":
                se = float(np.sqrt(
                    Byse[0] ** 2 / Bx[0] ** 2
                    + By[0] ** 2 * Bxse[0] ** 2 / Bx[0] ** 4
                    - 2 * psi * By[0] * Bxse[0] * Byse[0] / Bx[0] ** 3
                ))
            else:
                se = float(abs(Byse[0] / Bx[0]))
            rse = 1.0
            heter = float("nan")
            heter_p = float("nan")
            fstat = float((Bx[0] / Bxse[0]) ** 2)
            correlation_flag = False
        else:
            if weights == "simple":
                w = Byse ** -2
            elif weights == "delta":
                w = (Byse ** 2 + By ** 2 * Bxse ** 2 / Bx ** 2
                     - 2 * psi * By * Bxse * Byse / Bx) ** -1
            else:
                raise ValueError("weights must be one of: simple, delta.")
            if penalized:
                pw = _penalised_weights(Bx, Bxse, By, Byse) if weights == "simple" \
                    else _penalised_weights_delta(Bx, Bxse, By, Byse, psi)
                if weights == "simple":
                    w = Byse ** -2 * np.minimum(1.0, pw * 100)
                else:
                    w = ((Byse ** 2 + By ** 2 * Bxse ** 2 / Bx ** 2
                          - 2 * psi * By * Bxse * Byse / Bx) ** -1
                         * np.minimum(1.0, pw * 20))
            coef, coef_se, sigma = wls_no_intercept(By, Bx, w)
            theta = coef
            if model == "random":
                se = coef_se / min(sigma, 1.0)
            else:
                se = coef_se / sigma
            rse = sigma
            if penalized:
                heter = float("nan")
                heter_p = float("nan")
            else:
                heter = (nsnps - 1) * rse ** 2
                heter_p = stats.chi2.sf(heter, df=nsnps - 1)
            fstat = float(np.sum((Bx / Bxse) ** 2) / nsnps)
            correlation_flag = False

    if distribution == "normal":
        ci_l = ci_normal("l", theta, se, alpha)
        ci_u = ci_normal("u", theta, se, alpha)
        pval = 2.0 * stats.norm.cdf(-abs(theta / se))
    else:
        df = nsnps - 1
        ci_l = ci_t("l", theta, se, df, alpha)
        ci_u = ci_t("u", theta, se, df, alpha)
        pval = 2.0 * stats.t.cdf(-abs(theta / se), df=df)

    return IVWResult(
        method="IVW",
        exposure=obj.exposure,
        outcome=obj.outcome,
        estimate=float(theta),
        se=float(se),
        ci_lower=float(ci_l),
        ci_upper=float(ci_u),
        pvalue=float(pval),
        nsnps=nsnps,
        alpha=alpha,
        model=model,
        robust=robust,
        penalized=penalized,
        correlation=correlation_flag,
        rse=float(rse),
        heter_stat=float(heter),
        heter_pvalue=float(heter_p),
        fstat=float(fstat),
    )


def _penalised_weights(Bx, Bxse, By, Byse):
    theta = np.median(By / Bx)
    return stats.chi2.sf((Bx ** 2 / Byse ** 2) * (By / Bx - theta) ** 2, df=1)


def _penalised_weights_delta(Bx, Bxse, By, Byse, psi):
    theta = np.median(By / Bx)
    denom = (Byse ** 2 / Bx ** 2 + By ** 2 * Bxse ** 2 / Bx ** 4
             - 2 * psi * By * Bxse * Byse / Bx ** 3) ** -1
    return stats.chi2.sf(denom * (By / Bx - theta) ** 2, df=1)


# --------------------------------------------------------------------------
# MR-Egger
# --------------------------------------------------------------------------
def mr_egger(
    obj: MRInput,
    robust: bool = False,
    penalized: bool = False,
    correl: bool = False,
    distribution: str = "normal",
    alpha: float = 0.05,
) -> EggerResult:
    """MR-Egger regression (port of ``mr_egger``).

    Returns the causal estimate together with the directional-pleiotropy
    intercept and I-squared_GX.  Supports the correlated-SNP variant; the
    ``robust`` variant (R's ``lmrob``) is not ported.
    """
    if len(obj.betaX) < 3:
        raise ValueError("Method requires data on >2 variants.")
    if distribution not in ("normal", "t-dist"):
        raise ValueError("distribution must be one of: normal, t-dist.")
    if robust:
        raise NotImplementedError(
            "robust MR-Egger relies on R's robustbase::lmrob and is not ported."
        )

    By = np.sign(obj.betaX) * obj.betaY
    Bx = np.abs(obj.betaX)
    Bxse = obj.betaXse
    Byse = obj.betaYse
    rho = obj.correlation
    nsnps = len(Bx)
    if rho is not None:
        correl = True

    if correl and rho is not None:
        rho_e = rho * np.outer(np.sign(obj.betaX), np.sign(obj.betaX))
        omega = np.outer(Byse, Byse) * rho_e
        omega_inv = np.linalg.inv(omega)
        X = np.column_stack([np.ones(nsnps), Bx])
        XtOX = X.T @ omega_inv @ X
        XtOX_inv = np.linalg.inv(XtOX)
        theta_vals = XtOX_inv @ X.T @ omega_inv @ By
        theta_inter = float(theta_vals[0])
        theta_e = float(theta_vals[1])
        rse_vec = By - theta_inter - theta_e * Bx
        rse = float(np.sqrt(rse_vec @ omega_inv @ rse_vec / (nsnps - 2)))
        theta_e_se = np.sqrt(XtOX_inv[1, 1]) * max(1.0, rse)
        theta_inter_se = np.sqrt(XtOX_inv[0, 0]) * max(1.0, rse)
        i_sq = float("nan")
    else:
        w = Byse ** -2
        if penalized:
            base = wls_with_intercept(By, Bx, w)
            pen_w = stats.chi2.sf(
                (1.0 / Byse ** 2)
                * (By - base["intercept"] - base["slope"] * Bx) ** 2,
                df=1,
            )
            r_w = Byse ** -2 * np.minimum(1.0, pen_w * 100)
            fit = wls_with_intercept(By, Bx, r_w)
        else:
            fit = wls_with_intercept(By, Bx, w)
        sigma = fit["sigma"]
        theta_e = fit["slope"]
        theta_inter = fit["intercept"]
        theta_e_se = fit["slope_se"] / min(sigma, 1.0)
        theta_inter_se = fit["intercept_se"] / min(sigma, 1.0)
        rse = sigma
        Q = np.sum(
            (Bxse / Byse) ** -2
            * (Bx / Byse - np.average(Bx / Byse, weights=(Bxse / Byse) ** -2)) ** 2
        )
        i_sq = float(max(0.0, (Q - (nsnps - 1)) / Q))

    df = nsnps - 2
    ci_l = egger_bounds("l", distribution, theta_e, theta_e_se, df, rse, alpha)
    ci_u = egger_bounds("u", distribution, theta_e, theta_e_se, df, rse, alpha)
    ci_l_int = egger_bounds("l", distribution, theta_inter, theta_inter_se,
                            df, rse, alpha)
    ci_u_int = egger_bounds("u", distribution, theta_inter, theta_inter_se,
                            df, rse, alpha)

    if distribution == "normal":
        pleio_p = 2.0 * (1.0 - stats.norm.cdf(abs(theta_inter / theta_inter_se)))
        causal_p = 2.0 * (1.0 - stats.norm.cdf(abs(theta_e / theta_e_se)))
    else:
        def _tp(stat):
            if rse < 1:
                return max(
                    2.0 * (1.0 - stats.norm.cdf(abs(stat))),
                    2.0 * (1.0 - stats.t.cdf(abs(stat / rse), df=df)),
                )
            return 2.0 * (1.0 - stats.t.cdf(abs(stat), df=df))
        pleio_p = _tp(theta_inter / theta_inter_se)
        causal_p = _tp(theta_e / theta_e_se)

    heter = (nsnps - 2) * rse ** 2
    heter_p = stats.chi2.sf(heter, df=nsnps - 2)

    return EggerResult(
        method="MR-Egger",
        exposure=obj.exposure,
        outcome=obj.outcome,
        estimate=float(theta_e),
        se=float(theta_e_se),
        ci_lower=float(ci_l),
        ci_upper=float(ci_u),
        pvalue=float(causal_p),
        nsnps=nsnps,
        alpha=alpha,
        intercept=float(theta_inter),
        intercept_se=float(theta_inter_se),
        intercept_ci_lower=float(ci_l_int),
        intercept_ci_upper=float(ci_u_int),
        intercept_pvalue=float(pleio_p),
        causal_pvalue=float(causal_p),
        pleio_pvalue=float(pleio_p),
        model="random",
        robust=robust,
        penalized=penalized,
        rse=float(rse),
        heter_stat=float(heter),
        heter_pvalue=float(heter_p),
        i_sq=float(i_sq),
    )


# --------------------------------------------------------------------------
# Median-based estimators
# --------------------------------------------------------------------------
def weighted_median(theta, weights) -> float:
    """Weighted-median estimator (port of ``weighted.median``)."""
    theta = np.asarray(theta, float)
    weights = np.asarray(weights, float)
    order = np.argsort(theta, kind="stable")
    theta_s = theta[order]
    w_s = weights[order]
    cs = np.cumsum(w_s) - 0.5 * w_s
    cs = cs / np.sum(w_s)
    k = int(np.sum(cs < 0.5))
    ratio = (0.5 - cs[k - 1]) / (cs[k] - cs[k - 1])
    return float(theta_s[k - 1] + (theta_s[k] - theta_s[k - 1]) * ratio)


def weighted_median_boot_se(Bx, By, Bxse, Byse, weights, iterations, seed) -> float:
    """Bootstrap SE for the weighted median (port of
    ``weighted.median.boot.se``).

    Uses an independent ``numpy`` RNG.  The R version uses R's Mersenne
    twister, so the SE will agree only up to Monte-Carlo error (typically
    well within ~5%).
    """
    rng = np.random.default_rng(seed)
    n = len(By)
    med = np.empty(iterations)
    for i in range(iterations):
        bx_b = rng.normal(Bx, Bxse)
        by_b = rng.normal(By, Byse)
        med[i] = weighted_median(by_b / bx_b, weights)
    return float(np.std(med, ddof=1))


def mr_median(
    obj: MRInput,
    weighting: str = "weighted",
    distribution: str = "normal",
    alpha: float = 0.05,
    iterations: int = 10000,
    seed: int = _RDEFAULT_SEED,
) -> MedianResult:
    """Median-based estimator (port of ``mr_median``).

    ``weighting`` is one of ``simple``, ``weighted`` or ``penalized``.
    The standard error is obtained by parametric bootstrap; because the
    RNG differs from R's, the SE agrees up to Monte-Carlo error.
    """
    if len(obj.betaX) < 3:
        raise ValueError("Method requires data on >2 variants.")
    Bx, By = obj.betaX, obj.betaY
    Bxse, Byse = obj.betaXse, obj.betaYse
    theta = By / Bx
    n = len(Bx)

    if weighting == "simple":
        w = np.repeat(1.0 / n, n)
    elif weighting == "weighted":
        w = (Bx / Byse) ** 2
    elif weighting == "penalized":
        weighted_w = (Bx / Byse) ** 2
        penalty = stats.chi2.sf(
            weighted_w * (theta - weighted_median(theta, weighted_w)) ** 2, df=1
        )
        w = weighted_w * np.minimum(1.0, penalty * 20)
    else:
        raise ValueError("weighting must be one of: simple, weighted, penalized.")

    est = weighted_median(theta, w)
    se = weighted_median_boot_se(Bx, By, Bxse, Byse, w, iterations, seed)

    if distribution == "normal":
        ci_l = ci_normal("l", est, se, alpha)
        ci_u = ci_normal("u", est, se, alpha)
        pval = 2.0 * stats.norm.cdf(-abs(est / se))
    else:
        df = n - 1
        ci_l = ci_t("l", est, se, df, alpha)
        ci_u = ci_t("u", est, se, df, alpha)
        pval = 2.0 * stats.t.cdf(-abs(est / se), df=df)

    name = {"simple": "Simple median", "weighted": "Weighted median",
            "penalized": "Penalized weighted median"}[weighting]
    return MedianResult(
        method=name,
        exposure=obj.exposure,
        outcome=obj.outcome,
        estimate=float(est),
        se=float(se),
        ci_lower=float(ci_l),
        ci_upper=float(ci_u),
        pvalue=float(pval),
        nsnps=n,
        alpha=alpha,
        weighting=weighting,
    )


# --------------------------------------------------------------------------
# Mode-based estimate
# --------------------------------------------------------------------------
def _r_density_max(x, weights, bw):
    """Replicate R's ``density()`` Gaussian KDE peak location.

    R evaluates the density on a regular grid of 512 points spanning
    ``[min - 3*bw, max + 3*bw]`` and returns the grid x at the maximum.
    """
    x = np.asarray(x, float)
    weights = np.asarray(weights, float)
    n_grid = 512
    lo = x.min() - 3.0 * bw
    hi = x.max() + 3.0 * bw
    grid = np.linspace(lo, hi, n_grid)
    # Gaussian kernel density, weighted
    diff = (grid[:, None] - x[None, :]) / bw
    kern = np.exp(-0.5 * diff ** 2) / np.sqrt(2.0 * np.pi)
    dens = (kern * weights[None, :]).sum(axis=1) / bw
    return float(grid[np.argmax(dens)])


def _mad(x):
    """Median absolute deviation, R default constant 1.4826."""
    x = np.asarray(x, float)
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def _mbe_est(beta_iv, se_beta_iv, phi):
    s = 0.9 * min(np.std(beta_iv, ddof=1), _mad(beta_iv)) / len(beta_iv) ** 0.2
    s_weights = se_beta_iv ** -2 / np.sum(se_beta_iv ** -2)
    h = s * phi
    return _r_density_max(beta_iv, s_weights, h)


def mr_mbe(
    obj: MRInput,
    weighting: str = "weighted",
    stderror: str = "delta",
    phi: float = 1.0,
    seed: int = _RDEFAULT_SEED,
    iterations: int = 10000,
    distribution: str = "normal",
    alpha: float = 0.05,
) -> MBEResult:
    """Mode-based estimate (port of ``mr_mbe``).

    Bootstrap-based; the SE agrees with R up to Monte-Carlo error.
    """
    Bx, By = obj.betaX, obj.betaY
    Bxse, Byse = obj.betaXse, obj.betaYse
    n = len(Bx)
    if weighting not in ("weighted", "unweighted"):
        raise ValueError("weighting must be one of: weighted, unweighted.")
    if stderror not in ("simple", "delta"):
        raise ValueError("stderror must be one of: simple, delta.")

    beta_iv = By / Bx
    if stderror == "simple":
        se_iv = Byse / np.abs(Bx)
    else:
        se_iv = np.sqrt(Byse ** 2 / np.abs(Bx) ** 2 + By ** 2 * Bxse ** 2 / Bx ** 4)

    if weighting == "weighted":
        beta_mbe = _mbe_est(beta_iv, se_iv, phi)
    else:
        beta_mbe = _mbe_est(beta_iv, np.ones(len(beta_iv)), phi)

    rng = np.random.default_rng(seed)
    boot = np.empty(iterations)
    for i in range(iterations):
        b = rng.normal(beta_iv, se_iv)
        if weighting == "weighted":
            boot[i] = _mbe_est(b, se_iv, phi)
        else:
            boot[i] = _mbe_est(b, np.ones(len(beta_iv)), phi)
    se = _mad(boot)

    if distribution == "normal":
        pval = 2.0 * stats.norm.cdf(-abs(beta_mbe / se))
        ci_l = ci_normal("l", beta_mbe, se, alpha)
        ci_u = ci_normal("u", beta_mbe, se, alpha)
    else:
        df = n - 1
        pval = 2.0 * stats.t.cdf(-abs(beta_mbe / se), df=df)
        ci_l = ci_t("l", beta_mbe, se, df, alpha)
        ci_u = ci_t("u", beta_mbe, se, df, alpha)

    return MBEResult(
        method="MBE",
        exposure=obj.exposure,
        outcome=obj.outcome,
        estimate=float(beta_mbe),
        se=float(se),
        ci_lower=float(ci_l),
        ci_upper=float(ci_u),
        pvalue=float(pval),
        nsnps=n,
        alpha=alpha,
        weighting=weighting,
        stderror=stderror,
        phi=phi,
    )


# --------------------------------------------------------------------------
# Maximum-likelihood
# --------------------------------------------------------------------------
def _loglik(param, x, sigmax, y, sigmay):
    n = len(x)
    return (0.5 * np.sum((x - param[:n]) ** 2 / sigmax ** 2)
            + 0.5 * np.sum((y - param[n] * param[:n]) ** 2 / sigmay ** 2))


def _loglik_correl(param, x, taux, y, tauy):
    n = len(x)
    dx = x - param[:n]
    dy = y - param[n] * param[:n]
    return float(0.5 * dx @ taux @ dx + 0.5 * dy @ tauy @ dy)


def mr_maxlik(
    obj: MRInput,
    model: str = "default",
    correl: bool = False,
    psi: float = 0.0,
    distribution: str = "normal",
    alpha: float = 0.05,
) -> MaxLikResult:
    """Maximum-likelihood estimator (port of ``mr_maxlik``).

    Minimises the negative log-likelihood with ``scipy.optimize.minimize``
    (Nelder-Mead, mirroring R's ``optim`` default) and obtains the SE from
    the numerical Hessian.
    """
    Bx, By = obj.betaX, obj.betaY
    Bxse, Byse = obj.betaXse, obj.betaYse
    rho = obj.correlation
    n = len(Bx)
    if model == "default":
        model = "fixed" if n < 4 else "random"
    if rho is not None:
        correl = True
    if model not in ("random", "fixed"):
        raise ValueError("model must be one of: default, random, fixed.")

    x0 = np.append(Bx, np.sum(Bx * By / Byse ** 2) / np.sum(Bx ** 2 / Byse ** 2))

    if psi == 0 and not (correl and rho is not None):
        fun = lambda p: _loglik(p, Bx, Bxse, By, Byse)
    elif psi == 0 and correl and rho is not None:
        taux = np.linalg.inv(np.outer(Bxse, Bxse) * rho)
        tauy = np.linalg.inv(np.outer(Byse, Byse) * rho)
        fun = lambda p: _loglik_correl(p, Bx, taux, By, tauy)
    else:
        rho_use = rho if (correl and rho is not None) else np.eye(n)
        sigmax = np.outer(Bxse, Bxse) * rho_use
        sigmay = np.outer(Byse, Byse) * rho_use
        sigmaxy = np.outer(Bxse, Byse) * rho_use * psi
        big = np.block([[sigmax, sigmaxy], [sigmaxy, sigmay]])
        tauxy = np.linalg.inv(big)

        def fun(p):
            dx = Bx - p[:n]
            dy = By - p[n] * p[:n]
            v = np.concatenate([dx, dy])
            return float(0.5 * v @ tauxy @ v)

    res = nmmin(fun, x0, maxit=25000)
    par = res["x"]
    fval = res["fval"]
    theta = float(par[n])
    hess = _numeric_hessian(fun, par)
    hess_inv = np.linalg.inv(hess)
    var_theta = hess_inv[n, n]
    if model == "fixed":
        se = np.sqrt(var_theta)
    else:
        se = np.sqrt(var_theta * max(2.0 * fval / (n - 1), 1.0))
    rse = np.sqrt(2.0 * fval / (n - 1))

    if distribution == "normal":
        ci_l = ci_normal("l", theta, se, alpha)
        ci_u = ci_normal("u", theta, se, alpha)
        pval = 2.0 * stats.norm.cdf(-abs(theta / se))
    else:
        df = n - 1
        ci_l = ci_t("l", theta, se, df, alpha)
        ci_u = ci_t("u", theta, se, df, alpha)
        pval = 2.0 * stats.t.cdf(-abs(theta / se), df=df)
    heter = 2.0 * fval
    heter_p = stats.chi2.sf(heter, df=n - 1)

    return MaxLikResult(
        method="MaxLik",
        exposure=obj.exposure,
        outcome=obj.outcome,
        estimate=float(theta),
        se=float(se),
        ci_lower=float(ci_l),
        ci_upper=float(ci_u),
        pvalue=float(pval),
        nsnps=n,
        alpha=alpha,
        model=model,
        psi=psi,
        rse=float(rse),
        heter_stat=float(heter),
        heter_pvalue=float(heter_p),
    )


def _numeric_hessian(fun, x, eps=1e-4):
    """Central-difference numerical Hessian (mirrors R ``optim`` hessian)."""
    n = len(x)
    h = eps * (np.abs(x) + eps)
    hess = np.zeros((n, n))
    f0 = fun(x)
    for i in range(n):
        for j in range(i, n):
            xi = x.copy()
            if i == j:
                xi[i] = x[i] + h[i]
                fp = fun(xi)
                xi[i] = x[i] - h[i]
                fm = fun(xi)
                hess[i, i] = (fp - 2 * f0 + fm) / (h[i] ** 2)
            else:
                xpp = x.copy(); xpp[i] += h[i]; xpp[j] += h[j]
                xpm = x.copy(); xpm[i] += h[i]; xpm[j] -= h[j]
                xmp = x.copy(); xmp[i] -= h[i]; xmp[j] += h[j]
                xmm = x.copy(); xmm[i] -= h[i]; xmm[j] -= h[j]
                val = (fun(xpp) - fun(xpm) - fun(xmp) + fun(xmm)) / (
                    4 * h[i] * h[j])
                hess[i, j] = val
                hess[j, i] = val
    return hess


# --------------------------------------------------------------------------
# Debiased IVW
# --------------------------------------------------------------------------
def mr_divw(
    obj: MRInput,
    over_dispersion: bool = True,
    alpha: float = 0.05,
) -> DIVWResult:
    """Debiased inverse-variance weighted estimator (port of ``mr_divw``)."""
    Bx, By = obj.betaX, obj.betaY
    Bxse, Byse = obj.betaXse, obj.betaYse
    se_ratio = Bxse / Byse
    beta = np.sum(By * Bx / Byse ** 2) / np.sum((Bx ** 2 - Bxse ** 2) / Byse ** 2)
    mu = Bx / Bxse
    condition = (np.mean(mu ** 2) - 1) * np.sqrt(len(Bx))
    if not over_dispersion:
        tau2 = 0.0
    else:
        tau2 = max(0.0, np.sum(
            ((By - beta * Bx) ** 2 - Byse ** 2 - beta ** 2 * Bxse ** 2) / Byse ** 2
        ) / np.sum(Byse ** -2))
    V1 = np.sum(
        se_ratio ** 2 * mu ** 2
        + beta ** 2 * se_ratio ** 4 * (mu ** 2 + 1)
        + tau2 * se_ratio ** 2 / Byse ** 2 * mu ** 2
    )
    V2 = np.sum(se_ratio ** 2 * (mu ** 2 - 1))
    se = np.sqrt(V1 / V2 ** 2)
    c_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    ci_l = beta - c_alpha * se
    ci_u = beta + c_alpha * se
    pval = 2.0 * stats.norm.cdf(-abs(beta / se))
    return DIVWResult(
        method="DIVW",
        exposure=obj.exposure,
        outcome=obj.outcome,
        estimate=float(beta),
        se=float(se),
        ci_lower=float(ci_l),
        ci_upper=float(ci_u),
        pvalue=float(pval),
        nsnps=len(By),
        alpha=alpha,
        over_dispersion=over_dispersion,
        condition=float(condition),
    )


# --------------------------------------------------------------------------
# Contamination mixture
# --------------------------------------------------------------------------
def mr_conmix(
    obj: MRInput,
    psi: float = 0.0,
    ci_min: float = None,
    ci_max: float = None,
    ci_step: float = 0.01,
    alpha: float = 0.05,
) -> ConMixResult:
    """Contamination-mixture estimator (port of ``mr_conmix``)."""
    Bx, By = obj.betaX, obj.betaY
    Byse = obj.betaYse
    n = len(Bx)
    ratio = By / Bx
    ratio_se = np.abs(Byse / Bx)
    if ci_min is None:
        ci_min = float(np.min((By - 2 * Byse) / Bx))
    if ci_max is None:
        ci_max = float(np.max((By + 2 * Byse) / Bx))
    if psi <= 0:
        psi = 1.5 * np.std(ratio, ddof=1)

    n_steps = int(round((ci_max - ci_min) / ci_step)) + 1
    theta = ci_min + ci_step * np.arange(n_steps)
    lik = np.empty(n_steps)
    valid_best = None
    for j in range(n_steps):
        lik_inc = (-(theta[j] - ratio) ** 2 / 2.0 / ratio_se ** 2
                   - np.log(np.sqrt(2.0 * np.pi * ratio_se ** 2)))
        lik_exc = (-ratio ** 2 / 2.0 / (psi ** 2 + ratio_se ** 2)
                   - np.log(np.sqrt(2.0 * np.pi * (psi ** 2 + ratio_se ** 2))))
        valid = (lik_inc > lik_exc).astype(int)
        lik[j] = np.sum(np.where(valid == 1, lik_inc, lik_exc))
        if np.argmax(lik[:j + 1]) == j:
            valid_best = valid

    if np.sum(valid_best) < 1.5:
        phi = 1.0
    else:
        sel = valid_best == 1
        wmean = np.average(ratio[sel], weights=ratio_se[sel] ** -2)
        phi = max(
            np.sqrt(np.sum((ratio[sel] - wmean) ** 2 * ratio_se[sel] ** -2)
                    / (np.sum(valid_best) - 1)),
            1.0,
        )

    loglik = lik
    lik_inc0 = (-ratio ** 2 / 2.0 / ratio_se ** 2
                - np.log(np.sqrt(2.0 * np.pi * ratio_se ** 2)))
    lik_exc0 = (-ratio ** 2 / 2.0 / (psi ** 2 + ratio_se ** 2)
                - np.log(np.sqrt(2.0 * np.pi * (psi ** 2 + ratio_se ** 2))))
    valid0 = (lik_inc0 > lik_exc0).astype(int)
    loglik0 = np.sum(np.where(valid0 == 1, lik_inc0, lik_exc0))

    thresh = 2.0 * np.max(loglik) - stats.chi2.ppf(1.0 - alpha, df=1) * phi ** 2
    whichin = np.where(2.0 * loglik > thresh)[0]
    est = ci_min + ci_step * int(np.argmax(loglik))
    ci_range = ci_min + ci_step * whichin
    diffs = np.diff(ci_range)
    breaks = np.where(diffs > 1.01 * ci_step)[0]
    ci_lowers = np.concatenate([[ci_range[0]], ci_range[breaks + 1]])
    ci_uppers = np.concatenate([ci_range[breaks], [ci_range[-1]]])
    pval = stats.chi2.sf(2.0 * (np.max(loglik) - loglik0) * phi ** 2, df=1)

    valid_idx = list(np.where(valid_best == 1)[0])
    return ConMixResult(
        method="ConMix",
        exposure=obj.exposure,
        outcome=obj.outcome,
        estimate=float(est),
        se=float("nan"),
        ci_lower=float(ci_lowers[0]),
        ci_upper=float(ci_uppers[-1]),
        pvalue=float(pval),
        nsnps=n,
        alpha=alpha,
        psi=float(psi),
        ci_ranges=list(zip(ci_lowers.tolist(), ci_uppers.tolist())),
        valid=valid_idx,
        valid_snps=[obj.snps[i] for i in valid_idx],
    )


# --------------------------------------------------------------------------
# MR-Lasso
# --------------------------------------------------------------------------
def _glmnet_lasso_path(X, y, n_lambda=100, lambda_min_ratio=None, tol=1e-7,
                       max_iter=100000):
    """Cyclic coordinate-descent lasso path matching glmnet defaults.

    glmnet standardises X internally (1/n variance) and y, builds a
    log-spaced lambda grid from ``lambda_max`` down to
    ``lambda_max * lambda_min_ratio``, and fits with cyclic coordinate
    descent.  Coefficients are returned on the original scale, with no
    intercept (``intercept=FALSE``).
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, p = X.shape
    if lambda_min_ratio is None:
        lambda_min_ratio = 1e-4 if n < p else 1e-4
    # glmnet standardisation: column means + 1/n-variance sd
    xm = X.mean(axis=0)
    xs = np.sqrt(((X - xm) ** 2).mean(axis=0))
    xs[xs == 0] = 1.0
    Xs = (X - xm) / xs
    ym = y.mean()
    ys = np.sqrt(((y - ym) ** 2).mean())
    if ys == 0:
        ys = 1.0
    ysd = (y - ym) / ys
    # lambda_max on standardised scale
    lambda_max = np.max(np.abs(Xs.T @ ysd)) / n
    lam_grid = np.exp(np.linspace(
        np.log(lambda_max), np.log(lambda_max * lambda_min_ratio), n_lambda))
    beta = np.zeros(p)
    betas_std = []
    xx = (Xs ** 2).sum(axis=0) / n  # == 1 after standardisation
    for lam in lam_grid:
        for _ in range(max_iter):
            max_delta = 0.0
            r = ysd - Xs @ beta
            for j in range(p):
                bj_old = beta[j]
                rho = (Xs[:, j] @ r) / n + xx[j] * bj_old
                # soft-threshold
                if rho > lam:
                    bj_new = (rho - lam) / xx[j]
                elif rho < -lam:
                    bj_new = (rho + lam) / xx[j]
                else:
                    bj_new = 0.0
                delta = bj_new - bj_old
                if delta != 0.0:
                    r -= Xs[:, j] * delta
                    beta[j] = bj_new
                    max_delta = max(max_delta, abs(delta))
            if max_delta < tol:
                break
        betas_std.append(beta.copy())
    # de-standardise: beta_orig_j = beta_std_j * ys / xs_j
    betas_orig = np.array([b * ys / xs for b in betas_std])
    return lam_grid, betas_orig


def mr_lasso(
    obj: MRInput,
    distribution: str = "normal",
    alpha: float = 0.05,
    lambda_: float = None,
) -> LassoResult:
    """MR-Lasso estimator with post-selection IVW (port of ``mr_lasso``).

    The intercept-lasso is fitted with a glmnet-style coordinate-descent
    path; the heterogeneity-stopping rule then selects lambda and the
    post-lasso IVW estimate is computed on the SNPs deemed valid.
    """
    Bx = np.abs(obj.betaX)
    By = obj.betaY * np.sign(obj.betaX)
    Byse = obj.betaYse
    n = len(Bx)
    if distribution not in ("normal", "t-dist"):
        raise ValueError("distribution must be one of: normal, t-dist.")

    S = np.diag(Byse ** -2)
    S_half = np.diag(Byse ** -1)
    b = S_half @ Bx
    Pb = np.outer(b, b) / (b @ b)
    xlas = (np.eye(n) - Pb) @ S_half
    ylas = (np.eye(n) - Pb) @ S_half @ By

    if lambda_ is not None:
        lam_grid, betas = _glmnet_lasso_path(xlas, ylas, n_lambda=1)
        fit = betas[0]
        lam_used = lambda_
    else:
        lam_grid, betas = _glmnet_lasso_path(xlas, ylas, n_lambda=100)
        # glmnet returns lambdas decreasing; lamseq = sorted ascending
        order = np.argsort(lam_grid)
        lamseq = lam_grid[order]
        betas_sorted = betas[order]  # ascending lambda
        lamlen = len(lamseq)
        rse_arr = np.zeros((2, lamlen))
        for j in range(1, lamlen + 1):
            # R: las_fit$beta[, (lamlen - j + 1)] -> ascending index j-1
            coefs = betas_sorted[j - 1]
            av = np.where(coefs == 0)[0]
            if len(av) >= 2:
                Bx_av = Bx[av]
                By_av = By[av]
                w_av = Byse[av] ** -2
                coef, _, sigma = wls_no_intercept(By_av, Bx_av, w_av)
                rse_arr[0, j - 1] = sigma
            else:
                rse_arr[0, j - 1] = float("nan")
            rse_arr[1, j - 1] = len(av)
        rse_inc = rse_arr[0, 1:] - rse_arr[0, :-1]
        crit = stats.chi2.ppf(0.95, 1) / rse_arr[1, 1:]
        het = np.where((rse_arr[0, 1:] > 1) & (rse_inc > crit))[0]
        if len(het) == 0:
            lam_pos = lamlen
        else:
            lam_pos = int(np.min(het)) + 1  # convert to 1-based R index
        num_valid = rse_arr[1, :]
        min_lam_pos = int(np.min(np.where(num_valid > 1)[0])) + 1
        if lam_pos < min_lam_pos:
            lam_pos = min_lam_pos
        fit = betas_sorted[lam_pos - 1]
        lam_used = lamseq[lam_pos - 1]

    a = fit
    e = By - a
    reg_est = float((Bx @ S @ e) / (Bx @ S @ Bx))
    v = np.where(a == 0)[0]
    if len(v) > 1:
        Bx_v = Bx[v]
        By_v = By[v]
        w_v = Byse[v] ** -2
        coef, coef_se, sigma = wls_no_intercept(By_v, Bx_v, w_v)
        post_est = coef
        post_se = coef_se / min(sigma, 1.0)
    else:
        post_est = float("nan")
        post_se = float("nan")

    if distribution == "normal":
        ci_l = ci_normal("l", post_est, post_se, alpha)
        ci_u = ci_normal("u", post_est, post_se, alpha)
        pval = 2.0 * stats.norm.cdf(-abs(post_est / post_se))
    else:
        df = len(v) - 1
        ci_l = ci_t("l", post_est, post_se, df, alpha)
        ci_u = ci_t("u", post_est, post_se, df, alpha)
        pval = 2.0 * stats.t.cdf(-abs(post_est / post_se), df=df)

    return LassoResult(
        method="MR-Lasso",
        exposure=obj.exposure,
        outcome=obj.outcome,
        estimate=float(post_est),
        se=float(post_se),
        ci_lower=float(ci_l),
        ci_upper=float(ci_u),
        pvalue=float(pval),
        nsnps=n,
        alpha=alpha,
        lambda_=float(lam_used),
        n_valid=int(len(v)),
        valid_snps=[obj.snps[i] for i in v],
        reg_estimate=reg_est,
        reg_intercept=a,
    )


# --------------------------------------------------------------------------
# Constrained maximum likelihood (cML)
# --------------------------------------------------------------------------
def _cml_estimate(b_exp, b_out, se_exp, se_out, K, initial_theta=0.0,
                  initial_mu=None, maxit=100):
    """Single cML run (port of ``cML_estimate``)."""
    p = len(b_exp)
    theta = initial_theta
    theta_old = theta - 1.0
    mu_vec = np.zeros(p) if initial_mu is None else np.array(initial_mu, float)
    ite = 0
    v_bg = np.zeros(p)
    while (abs(theta_old - theta) > 1e-7) and (ite < maxit):
        theta_old = theta
        ite += 1
        if K > 0:
            v_importance = (b_out - theta * mu_vec) ** 2 / se_out ** 2
            # R: sort(order(decreasing)[1:K])
            nonzero = np.sort(np.argsort(-v_importance, kind="stable")[:K])
            v_bg = np.zeros(p)
            v_bg[nonzero] = (b_out - theta * mu_vec)[nonzero]
        else:
            v_bg = np.zeros(p)
        mu_vec = ((b_exp / se_exp ** 2 + theta * (b_out - v_bg) / se_out ** 2)
                  / (1.0 / se_exp ** 2 + theta ** 2 / se_out ** 2))
        theta = (np.sum((b_out - v_bg) * mu_vec / se_out ** 2)
                 / np.sum(mu_vec ** 2 / se_out ** 2))
    if K > 0:
        nonzero_ind = np.where(v_bg != 0)[0]
        mu_vec[nonzero_ind] = b_exp[nonzero_ind]
        v_bg[nonzero_ind] = (b_out - theta * mu_vec)[nonzero_ind]
    return {"theta": theta, "b_vec": mu_vec, "r_vec": v_bg}


def _cml_sd_theta(b_exp, b_out, se_exp, se_out, theta, b_vec, r_vec):
    """Port of ``cML_SdTheta``."""
    zero_ind = np.where(r_vec == 0)[0]
    var_theta = 1.0 / (
        np.sum((b_vec ** 2 / se_out ** 2)[zero_ind])
        - np.sum(((2 * b_vec * theta - b_out) ** 2 / se_out ** 4
                  * 1.0 / (1.0 / se_exp ** 2 + theta ** 2 / se_out ** 2))[zero_ind])
    )
    if var_theta <= 0:
        return float("nan")
    return np.sqrt(var_theta)


def _cml_estimate_random(b_exp, b_out, se_exp, se_out, K, random_start=0,
                         maxit=100, rng=None):
    """Port of ``cML_estimate_random`` (multi-start cML)."""
    p = len(b_exp)
    min_range = np.min(b_out / b_exp)
    max_range = np.max(b_out / b_exp)
    theta_cand, sd_cand, l_cand, inv_cand = [], [], [], []
    for ri in range(1 + random_start):
        if ri == 0:
            it0, im0 = 0.0, np.zeros(p)
        else:
            it0 = rng.uniform(min_range, max_range)
            im0 = rng.normal(b_exp, se_exp)
        mle = _cml_estimate(b_exp, b_out, se_exp, se_out, K,
                            initial_theta=it0, initial_mu=im0, maxit=maxit)
        neg_l = (np.sum((b_exp - mle["b_vec"]) ** 2 / (2 * se_exp ** 2))
                 + np.sum((b_out - mle["theta"] * mle["b_vec"] - mle["r_vec"]) ** 2
                          / (2 * se_out ** 2)))
        sd_theta = _cml_sd_theta(b_exp, b_out, se_exp, se_out, mle["theta"],
                                 mle["b_vec"], mle["r_vec"])
        theta_cand.append(mle["theta"])
        sd_cand.append(sd_theta)
        l_cand.append(neg_l)
        inv_cand.append(np.asarray(mle["r_vec"], float))
    idx = int(np.argmin(l_cand))
    return {
        "theta": theta_cand[idx],
        "se": sd_cand[idx],
        "l": l_cand[idx],
        "r_est": inv_cand[idx],
    }


def mr_cml(
    obj: MRInput,
    n: int,
    ma: bool = True,
    dp: bool = True,
    k_vec=None,
    random_start: int = 0,
    num_pert: int = 200,
    random_start_pert: int = 0,
    maxit: int = 100,
    random_seed: int = 314,
    alpha: float = 0.05,
) -> CMLResult:
    """Constrained-maximum-likelihood estimator (port of ``mr_cML``).

    ``n`` is the sample size used in the BIC penalty.  ``ma`` toggles
    model averaging (cML-MA vs cML-BIC); ``dp`` toggles data-perturbation.
    With ``dp=True`` results depend on a Gaussian perturbation RNG, so
    they agree with R up to Monte-Carlo error.
    """
    b_exp = obj.betaX
    b_out = obj.betaY
    se_exp = obj.betaXse
    se_out = obj.betaYse
    p = len(b_exp)
    if k_vec is None:
        k_vec = np.arange(0, p - 1)
    else:
        k_vec = np.asarray(k_vec)
    rng = np.random.default_rng(random_seed)

    rand_theta, rand_sd, rand_l, invalid_mat = [], [], [], []
    for K in k_vec:
        rr = _cml_estimate_random(b_exp, b_out, se_exp, se_out, K,
                                  random_start=random_start, maxit=maxit, rng=rng)
        rand_theta.append(rr["theta"])
        rand_sd.append(rr["se"])
        rand_l.append(rr["l"])
        invalid_mat.append(rr["r_est"])
    theta_v = np.array(rand_theta)
    sd_v = np.array(rand_sd)
    l_v = np.array(rand_l)
    invalid_mat = np.array(invalid_mat)

    gof1 = float("nan")
    gof2 = float("nan")

    if not dp:
        bic = np.log(n) * k_vec + 2 * l_v
        bic = bic - np.min(bic)
        if not ma:
            min_ind = int(np.argmin(bic))
            est = theta_v[min_ind]
            se = sd_v[min_ind]
            bic_invalid = list(np.where(invalid_mat[min_ind] != 0)[0])
        else:
            weight = np.exp(-0.5 * bic)
            weight = weight / np.sum(weight)
            est = np.sum(theta_v * weight)
            se = np.nansum(weight * np.sqrt(sd_v ** 2 + (theta_v - est) ** 2))
            bic_invalid = []
        pval = 2.0 * stats.norm.cdf(-abs(est / se))
    else:
        rand_pert_theta, rand_pert_sd, rand_pert_l = [], [], []
        for _ in range(num_pert):
            b_exp_new = b_exp + rng.normal(size=p) * se_exp
            b_out_new = b_out + rng.normal(size=p) * se_out
            tcol, scol, lcol = [], [], []
            for K in k_vec:
                mle = _cml_estimate_random(b_exp_new, b_out_new, se_exp, se_out,
                                           K, random_start=random_start_pert,
                                           maxit=maxit, rng=rng)
                tcol.append(mle["theta"])
                scol.append(mle["se"])
                lcol.append(mle["l"])
            rand_pert_theta.append(tcol)
            rand_pert_sd.append(scol)
            rand_pert_l.append(lcol)
        theta_pt = np.array(rand_pert_theta).T  # (K, num_pert)
        sd_pt = np.array(rand_pert_sd).T
        l_pt = np.array(rand_pert_l).T
        var_mat = sd_pt ** 2
        numer_perturb = theta_pt.shape[1]
        l_pt_mean = l_pt.mean(axis=1)
        sd_pt_v = np.sqrt(np.var(theta_pt, axis=1, ddof=1))
        theta_pt_mat = theta_pt
        theta_pt_v = theta_pt.mean(axis=1)

        bic = np.log(n) * k_vec + 2 * l_v
        bic = bic - np.min(bic)
        min_ind = int(np.argmin(bic))
        pt_sd = sd_pt_v[min_ind]
        origin_sd = sd_v[min_ind]
        more_var = np.var(var_mat[min_ind], ddof=1)
        x = theta_pt_mat[min_ind]
        sd_x = np.sqrt(
            (np.mean((x - np.mean(x)) ** 4)
             - (numer_perturb - 3) / (numer_perturb - 1) * np.var(x, ddof=1) ** 2)
            / numer_perturb + more_var
        )
        tstat = (origin_sd ** 2 - pt_sd ** 2) / sd_x
        gof1 = 2.0 * stats.norm.cdf(-abs(tstat))
        tstat2 = (origin_sd ** 2 - pt_sd ** 2) / np.sqrt(
            2.0 / (numer_perturb - 1) * pt_sd ** 4 + more_var)
        gof2 = 2.0 * stats.norm.cdf(-abs(tstat2))

        bic2 = np.log(n) * k_vec + 2 * l_pt_mean
        bic2 = bic2 - np.min(bic2)
        if not ma:
            min_ind2 = int(np.argmin(bic2))
            est = theta_pt_v[min_ind2]
            se = sd_pt_v[min_ind2]
        else:
            weight = np.exp(-0.5 * bic2)
            weight = weight / np.sum(weight)
            est = np.sum(theta_pt_v * weight)
            se = np.nansum(weight * np.sqrt(sd_pt_v ** 2 + (theta_pt_v - est) ** 2))
        pval = 2.0 * stats.norm.cdf(-abs(est / se))
        bic_invalid = []

    z = stats.norm.ppf(1.0 - alpha / 2.0)
    return CMLResult(
        method="MR-cML",
        exposure=obj.exposure,
        outcome=obj.outcome,
        estimate=float(est),
        se=float(se),
        ci_lower=float(est - z * se),
        ci_upper=float(est + z * se),
        pvalue=float(pval),
        nsnps=p,
        alpha=alpha,
        ma=ma,
        dp=dp,
        bic_invalid=[int(i) for i in bic_invalid],
        gof1_pvalue=float(gof1),
        gof2_pvalue=float(gof2),
    )
