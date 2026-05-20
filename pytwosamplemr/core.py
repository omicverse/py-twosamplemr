"""Core data structures and statistical helpers for py-twosamplemr.

This module mirrors the ``MRInput`` S4 class of the R/CRAN package
*MendelianRandomization* (Burgess et al.) and provides the small set of
shared helper functions (confidence-interval constructors, Egger CI
bounds) used throughout the estimator suite.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy import stats

__all__ = ["MRInput", "mr_input", "ci_normal", "ci_t", "egger_bounds"]


@dataclass
class MRInput:
    """Harmonised SNP-exposure / SNP-outcome summary statistics.

    Faithful port of the *MendelianRandomization* ``MRInput`` S4 object.

    Attributes
    ----------
    betaX, betaXse : numpy.ndarray
        SNP-exposure associations and their standard errors.
    betaY, betaYse : numpy.ndarray
        SNP-outcome associations and their standard errors.
    snps : list of str
        Variant identifiers.
    exposure, outcome : str
        Phenotype labels.
    correlation : numpy.ndarray or None
        Optional ``n x n`` SNP-SNP correlation matrix (for the
        ``correl`` estimator variants).
    effect_allele, other_allele : list or None
        Optional allele annotations.
    eaf : numpy.ndarray or None
        Optional effect-allele frequencies.
    """

    betaX: np.ndarray
    betaXse: np.ndarray
    betaY: np.ndarray
    betaYse: np.ndarray
    snps: list = field(default_factory=list)
    exposure: str = "exposure"
    outcome: str = "outcome"
    correlation: Optional[np.ndarray] = None
    effect_allele: Optional[list] = None
    other_allele: Optional[list] = None
    eaf: Optional[np.ndarray] = None

    @property
    def nsnps(self) -> int:
        return len(self.betaX)

    def __len__(self) -> int:
        return self.nsnps

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"MRInput(nsnps={self.nsnps}, exposure='{self.exposure}', "
            f"outcome='{self.outcome}', "
            f"correlated={self.correlation is not None})"
        )


def mr_input(
    bx: Sequence[float],
    bxse: Sequence[float],
    by: Sequence[float],
    byse: Sequence[float],
    snps: Optional[Sequence[str]] = None,
    exposure: str = "exposure",
    outcome: str = "outcome",
    correlation: Optional[np.ndarray] = None,
    effect_allele: Optional[Sequence[str]] = None,
    other_allele: Optional[Sequence[str]] = None,
    eaf: Optional[Sequence[float]] = None,
) -> MRInput:
    """Construct an :class:`MRInput` object.

    Mirrors ``MendelianRandomization::mr_input``.  When ``snps`` is not
    supplied, variants are auto-named ``snp_1``, ``snp_2`` ... as in R.
    """
    bx = np.asarray(bx, dtype=float)
    bxse = np.asarray(bxse, dtype=float)
    by = np.asarray(by, dtype=float)
    byse = np.asarray(byse, dtype=float)
    n = len(bx)
    if not (len(bxse) == len(by) == len(byse) == n):
        raise ValueError("bx, bxse, by, byse must all have the same length.")
    if snps is None or list(snps) == ["snp"]:
        snps = [f"snp_{i + 1}" for i in range(n)]
    else:
        snps = [str(s) for s in snps]
    corr = None
    if correlation is not None:
        corr = np.asarray(correlation, dtype=float)
        if corr.shape != (n, n):
            raise ValueError("correlation must be an n x n matrix.")
    return MRInput(
        betaX=bx,
        betaXse=bxse,
        betaY=by,
        betaYse=byse,
        snps=snps,
        exposure=str(exposure),
        outcome=str(outcome),
        correlation=corr,
        effect_allele=list(effect_allele) if effect_allele is not None else None,
        other_allele=list(other_allele) if other_allele is not None else None,
        eaf=np.asarray(eaf, dtype=float) if eaf is not None else None,
    )


# --------------------------------------------------------------------------
# confidence-interval helpers (faithful ports of ci_normal / ci_t)
# --------------------------------------------------------------------------
def ci_normal(kind: str, mean: float, se: float, alpha: float) -> float:
    """Normal-distribution confidence-interval bound (``ci_normal``)."""
    x = 1.0 - alpha / 2.0
    z = stats.norm.ppf(x)
    if kind == "l":
        return mean - z * se
    if kind == "u":
        return mean + z * se
    raise ValueError("kind must be 'l' or 'u'.")


def ci_t(kind: str, mean: float, se: float, df: float, alpha: float) -> float:
    """t-distribution confidence-interval bound (``ci_t``)."""
    x = 1.0 - alpha / 2.0
    t = stats.t.ppf(x, df=df)
    if kind == "l":
        return mean - t * se
    if kind == "u":
        return mean + t * se
    raise ValueError("kind must be 'l' or 'u'.")


def egger_bounds(
    kind: str,
    dist: str,
    theta: float,
    thetase: float,
    df: float,
    rse: float,
    alpha: float,
) -> float:
    """MR-Egger CI bound (faithful port of ``egger.bounds``)."""
    x = 1.0 - alpha / 2.0
    z = stats.norm.ppf(x)
    if kind == "u":
        if dist == "normal":
            return theta + z * thetase
        if dist == "t-dist":
            t = stats.t.ppf(x, df=df)
            if rse < 1:
                return max(theta + z * thetase, theta + t * thetase * rse)
            return theta + t * thetase
    elif kind == "l":
        if dist == "normal":
            return theta - z * thetase
        if dist == "t-dist":
            t = stats.t.ppf(x, df=df)
            if rse < 1:
                return min(theta - z * thetase, theta - t * thetase * rse)
            return theta - t * thetase
    return float("nan")
