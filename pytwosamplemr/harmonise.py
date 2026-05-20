"""TwoSampleMR-style harmonisation and diagnostics.

Implements the workflow side of two-sample MR: allele harmonisation of
SNP-exposure and SNP-outcome summary statistics, heterogeneity and
directional-pleiotropy tests, Steiger directionality / filtering, and the
per-SNP single-SNP / leave-one-out analyses.

These routines follow the *TwoSampleMR* R package (Hemani et al.,
*eLife* 2018) conventions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .core import MRInput, mr_input
from .estimators import mr_egger, mr_ivw

__all__ = [
    "harmonise_data",
    "mr_heterogeneity",
    "mr_pleiotropy_test",
    "mr_steiger",
    "directionality_test",
    "mr_singlesnp",
    "mr_leaveoneout",
]

_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}


def _flip(allele: str) -> str:
    return "".join(_COMPLEMENT.get(b, b) for b in str(allele).upper())


def _is_palindromic(a1: str, a2: str) -> bool:
    a1, a2 = str(a1).upper(), str(a2).upper()
    return _COMPLEMENT.get(a1) == a2


def harmonise_data(
    exposure: pd.DataFrame,
    outcome: pd.DataFrame,
    tolerance: float = 0.08,
    action: int = 2,
) -> pd.DataFrame:
    """Align SNP-exposure and SNP-outcome effects to the same allele.

    Parameters
    ----------
    exposure, outcome : pandas.DataFrame
        Must contain columns ``SNP``, ``beta``, ``se``,
        ``effect_allele``, ``other_allele``; ``eaf`` is optional but
        required for palindromic-SNP resolution under ``action=2``.
    tolerance : float
        Max allowed difference in allele frequency when inferring the
        strand of a palindromic SNP (``action=2``).
    action : {1, 2, 3}
        1 = assume all on the forward strand (no flipping);
        2 = try to resolve palindromic SNPs by allele frequency, drop
        ambiguous ones (the TwoSampleMR default);
        3 = drop all palindromic SNPs.

    Returns
    -------
    pandas.DataFrame
        One row per harmonised SNP with aligned ``beta.exposure`` /
        ``beta.outcome`` and a boolean ``mr_keep`` flag.
    """
    e = exposure.set_index("SNP")
    o = outcome.set_index("SNP")
    common = [s for s in e.index if s in o.index]
    rows = []
    for snp in common:
        er = e.loc[snp]
        orow = o.loc[snp]
        ea_e = str(er["effect_allele"]).upper()
        oa_e = str(er["other_allele"]).upper()
        ea_o = str(orow["effect_allele"]).upper()
        oa_o = str(orow["other_allele"]).upper()
        beta_e = float(er["beta"])
        beta_o = float(orow["beta"])
        se_e = float(er["se"])
        se_o = float(orow["se"])
        eaf_e = float(er["eaf"]) if "eaf" in e.columns and not pd.isna(
            er.get("eaf", np.nan)) else np.nan
        eaf_o = float(orow["eaf"]) if "eaf" in o.columns and not pd.isna(
            orow.get("eaf", np.nan)) else np.nan

        keep = True
        palindromic = _is_palindromic(ea_e, oa_e)

        if (ea_e, oa_e) == (ea_o, oa_o):
            pass  # already aligned
        elif (ea_e, oa_e) == (oa_o, ea_o):
            beta_o = -beta_o
            eaf_o = 1.0 - eaf_o if not np.isnan(eaf_o) else eaf_o
        elif action == 1:
            pass
        elif (ea_e, oa_e) == (_flip(ea_o), _flip(oa_o)):
            ea_o, oa_o = _flip(ea_o), _flip(oa_o)
        elif (ea_e, oa_e) == (_flip(oa_o), _flip(ea_o)):
            beta_o = -beta_o
            eaf_o = 1.0 - eaf_o if not np.isnan(eaf_o) else eaf_o
        else:
            keep = False  # incompatible alleles

        if palindromic:
            if action == 3:
                keep = False
            elif action == 2:
                if np.isnan(eaf_e) or np.isnan(eaf_o):
                    keep = False
                else:
                    # ambiguous if MAF too close to 0.5
                    if min(eaf_e, 1 - eaf_e) > 0.5 - tolerance:
                        keep = False
                    else:
                        same_side = (eaf_e - 0.5) * (eaf_o - 0.5) > 0
                        if not same_side:
                            beta_o = -beta_o
                            eaf_o = 1.0 - eaf_o

        rows.append({
            "SNP": snp,
            "effect_allele": ea_e,
            "other_allele": oa_e,
            "beta.exposure": beta_e,
            "se.exposure": se_e,
            "eaf.exposure": eaf_e,
            "beta.outcome": beta_o,
            "se.outcome": se_o,
            "eaf.outcome": eaf_o,
            "palindromic": palindromic,
            "mr_keep": keep,
        })
    return pd.DataFrame(rows)


def _as_input(dat) -> MRInput:
    """Coerce an :class:`MRInput` or a harmonised DataFrame to MRInput."""
    if isinstance(dat, MRInput):
        return dat
    d = dat[dat["mr_keep"]] if "mr_keep" in dat.columns else dat
    return mr_input(
        bx=d["beta.exposure"].values,
        bxse=d["se.exposure"].values,
        by=d["beta.outcome"].values,
        byse=d["se.outcome"].values,
        snps=d["SNP"].tolist() if "SNP" in d.columns else None,
    )


def mr_heterogeneity(dat, methods=("ivw", "egger")) -> pd.DataFrame:
    """Cochran's Q heterogeneity statistic per method (``mr_heterogeneity``)."""
    obj = _as_input(dat)
    rows = []
    if "ivw" in methods:
        r = mr_ivw(obj)
        rows.append({"method": "Inverse variance weighted", "Q": r.heter_stat,
                     "Q_df": obj.nsnps - 1, "Q_pval": r.heter_pvalue})
    if "egger" in methods:
        r = mr_egger(obj)
        rows.append({"method": "MR Egger", "Q": r.heter_stat,
                     "Q_df": obj.nsnps - 2, "Q_pval": r.heter_pvalue})
    return pd.DataFrame(rows)


def mr_pleiotropy_test(dat) -> pd.DataFrame:
    """MR-Egger intercept directional-pleiotropy test (``mr_pleiotropy_test``)."""
    obj = _as_input(dat)
    r = mr_egger(obj)
    return pd.DataFrame([{
        "egger_intercept": r.intercept,
        "se": r.intercept_se,
        "pval": r.intercept_pvalue,
    }])


def mr_steiger(
    p_exp,
    p_out,
    n_exp,
    n_out,
    r_exp=None,
    r_out=None,
) -> dict:
    """Steiger test of causal directionality (``mr_steiger``).

    Compares the variance explained in the exposure vs the outcome by the
    instruments.  Provide either p-values + sample sizes (the variance
    explained is then derived from the implied correlations) or the
    correlations directly via ``r_exp`` / ``r_out``.

    Returns a dict with ``r2_exp``, ``r2_out``, ``correct_causal_direction``
    and the Steiger p-value (``steiger_test``).
    """
    p_exp = np.asarray(p_exp, float)
    p_out = np.asarray(p_out, float)
    n_exp = np.asarray(n_exp, float)
    n_out = np.asarray(n_out, float)

    if r_exp is None:
        r_exp = _p_to_r(p_exp, n_exp)
    else:
        r_exp = np.asarray(r_exp, float)
    if r_out is None:
        r_out = _p_to_r(p_out, n_out)
    else:
        r_out = np.asarray(r_out, float)

    r2_exp = float(np.sum(r_exp ** 2))
    r2_out = float(np.sum(r_out ** 2))
    correct = bool(r2_exp > r2_out)

    # Steiger test of difference between two dependent correlations
    p_steiger = _steiger_pvalue(r2_exp, r2_out, n_exp, n_out)
    return {
        "r2_exp": r2_exp,
        "r2_out": r2_out,
        "correct_causal_direction": correct,
        "steiger_test": float(p_steiger),
    }


def directionality_test(dat, n_exp=None, n_out=None) -> pd.DataFrame:
    """Steiger directionality test on a harmonised dataset.

    Convenience wrapper around :func:`mr_steiger` that derives the per-SNP
    correlations from the harmonised effect sizes and effect-allele
    frequencies (``2 * eaf * (1 - eaf) * beta^2`` variance-explained
    approximation).
    """
    d = dat[dat["mr_keep"]] if "mr_keep" in dat.columns else dat
    eaf_e = d["eaf.exposure"].values
    eaf_o = d["eaf.outcome"].values
    be = d["beta.exposure"].values
    bo = d["beta.outcome"].values
    r2_exp = float(np.sum(2 * eaf_e * (1 - eaf_e) * be ** 2))
    r2_out = float(np.sum(2 * eaf_o * (1 - eaf_o) * bo ** 2))
    correct = bool(r2_exp > r2_out)
    if n_exp is not None and n_out is not None:
        p = _steiger_pvalue(r2_exp, r2_out,
                            np.full(len(d), n_exp), np.full(len(d), n_out))
    else:
        p = np.nan
    return pd.DataFrame([{
        "snp_r2.exposure": r2_exp,
        "snp_r2.outcome": r2_out,
        "correct_causal_direction": correct,
        "steiger_pval": float(p),
    }])


def _p_to_r(pval, n):
    """Convert a two-sided p-value + sample size to an absolute correlation.

    Uses ``F.isf`` (= 1 - cdf) which is numerically stable for the tiny
    p-values typical of GWAS instruments; the implied F-statistic is
    capped to avoid overflow.
    """
    pval = np.clip(np.asarray(pval, float), 1e-300, 1.0)
    n = np.asarray(n, float)
    t2 = stats.f.isf(pval, 1, n - 2)
    t2 = np.where(np.isfinite(t2), t2, 1e12)
    return np.sqrt(t2 / (t2 + n - 2))


def _steiger_pvalue(r2_exp, r2_out, n_exp, n_out):
    """Two-sided Steiger p-value for r2_exp != r2_out (Fisher z)."""
    n = float(np.mean(np.concatenate([np.atleast_1d(n_exp),
                                      np.atleast_1d(n_out)])))
    r_exp = np.sqrt(min(r2_exp, 0.9999))
    r_out = np.sqrt(min(r2_out, 0.9999))
    z_exp = np.arctanh(r_exp)
    z_out = np.arctanh(r_out)
    se = np.sqrt(2.0 / (n - 3))
    z = (z_exp - z_out) / se
    return 2.0 * stats.norm.sf(abs(z))


def mr_singlesnp(dat, single_method="ivw") -> pd.DataFrame:
    """Per-SNP Wald ratios plus the all-SNP combined estimate.

    Faithful to ``TwoSampleMR::mr_singlesnp``: each row is a single-SNP
    Wald ratio (``beta.outcome / beta.exposure``, SE
    ``|se.outcome / beta.exposure|``); the final rows give the IVW and
    MR-Egger estimates over all SNPs.
    """
    obj = _as_input(dat)
    Bx, By = obj.betaX, obj.betaY
    Byse = obj.betaYse
    rows = []
    for i in range(obj.nsnps):
        b = By[i] / Bx[i]
        se = abs(Byse[i] / Bx[i])
        rows.append({
            "SNP": obj.snps[i], "b": float(b), "se": float(se),
            "p": float(2.0 * stats.norm.cdf(-abs(b / se))),
        })
    ivw = mr_ivw(obj)
    rows.append({"SNP": "All - Inverse variance weighted", "b": ivw.estimate,
                 "se": ivw.se, "p": ivw.pvalue})
    if obj.nsnps >= 3:
        eg = mr_egger(obj)
        rows.append({"SNP": "All - MR Egger", "b": eg.estimate,
                     "se": eg.se, "p": eg.causal_pvalue})
    return pd.DataFrame(rows)


def mr_leaveoneout(dat, method="ivw") -> pd.DataFrame:
    """Leave-one-out IVW analysis (``mr_leaveoneout``).

    Re-estimates the causal effect with each SNP omitted in turn; the
    final row gives the all-SNP estimate.
    """
    obj = _as_input(dat)
    Bx, By = obj.betaX, obj.betaY
    Bxse, Byse = obj.betaXse, obj.betaYse
    n = obj.nsnps
    rows = []
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        sub = mr_input(bx=Bx[idx], bxse=Bxse[idx], by=By[idx], byse=Byse[idx],
                       snps=[obj.snps[j] for j in idx])
        r = mr_ivw(sub)
        rows.append({"SNP": obj.snps[i], "b": r.estimate, "se": r.se,
                     "p": r.pvalue})
    full = mr_ivw(obj)
    rows.append({"SNP": "All", "b": full.estimate, "se": full.se,
                 "p": full.pvalue})
    return pd.DataFrame(rows)
