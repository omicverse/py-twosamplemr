"""Matplotlib plots for two-sample MR.

Faithful in spirit to ``MendelianRandomization``/``TwoSampleMR`` plots:

* :func:`mr_plot` / :func:`mr_scatter` — SNP-effect scatter with the
  fitted method lines (IVW, MR-Egger ...).
* :func:`mr_forest` — per-SNP Wald-ratio forest plot.
* :func:`mr_funnel` — funnel plot of single-SNP estimates vs precision.
* :func:`mr_loo` — leave-one-out forest plot.
"""
from __future__ import annotations

import numpy as np

from .core import MRInput
from .estimators import mr_egger, mr_ivw
from .harmonise import mr_leaveoneout, mr_singlesnp

__all__ = ["mr_plot", "mr_scatter", "mr_forest", "mr_funnel", "mr_loo"]


def _lazy_plt():
    """Import :mod:`matplotlib.pyplot`, honouring the active backend.

    The backend is left untouched so the plots render inline in Jupyter
    and headlessly under Agg in CI / scripts.
    """
    import matplotlib.pyplot as plt
    return plt


def mr_scatter(obj: MRInput, methods=("ivw", "egger"), ax=None):
    """SNP-effect scatter plot with fitted causal-effect lines.

    Plots SNP-outcome vs SNP-exposure associations (with error bars) and
    overlays the fitted line for each requested method.
    """
    plt = _lazy_plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    Bx, By = obj.betaX, obj.betaY
    Bxse, Byse = obj.betaXse, obj.betaYse
    # orient exposure positive (as in MR-Egger)
    sign = np.sign(Bx)
    bx = np.abs(Bx)
    by = By * sign
    ax.errorbar(bx, by, xerr=Bxse, yerr=Byse, fmt="o", ms=4,
                color="#444444", ecolor="#999999", elinewidth=0.8,
                capsize=0, zorder=2, label="SNPs")
    xlim = np.array([0, bx.max() * 1.1])
    colors = {"ivw": "#d62728", "egger": "#1f77b4"}
    if "ivw" in methods:
        r = mr_ivw(obj)
        ax.plot(xlim, r.estimate * xlim, "-", color=colors["ivw"],
                lw=2, label=f"IVW ({r.estimate:.3f})", zorder=3)
    if "egger" in methods and obj.nsnps >= 3:
        r = mr_egger(obj)
        ax.plot(xlim, r.intercept + r.estimate * xlim, "-",
                color=colors["egger"], lw=2,
                label=f"MR-Egger ({r.estimate:.3f})", zorder=3)
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlabel(f"SNP effect on {obj.exposure}")
    ax.set_ylabel(f"SNP effect on {obj.outcome}")
    ax.set_title("MR scatter plot")
    ax.legend(fontsize=8, frameon=False)
    return ax


# alias matching the R generic name
mr_plot = mr_scatter


def mr_forest(obj: MRInput, alpha: float = 0.05, ax=None):
    """Per-SNP Wald-ratio forest plot with the combined IVW estimate."""
    plt = _lazy_plt()
    from scipy import stats
    df = mr_singlesnp(obj)
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 0.32 * len(df) + 1.5))
    ys = np.arange(len(df))[::-1]
    for y, (_, row) in zip(ys, df.iterrows()):
        is_summary = row["SNP"].startswith("All")
        col = "#d62728" if is_summary else "#333333"
        lo = row["b"] - z * row["se"]
        hi = row["b"] + z * row["se"]
        ax.plot([lo, hi], [y, y], "-", color=col, lw=1.6)
        ax.plot(row["b"], y, "s" if is_summary else "o", color=col,
                ms=7 if is_summary else 5)
    ax.set_yticks(ys)
    ax.set_yticklabels(df["SNP"], fontsize=7)
    ax.axvline(0, color="grey", lw=0.6, ls="--")
    ax.set_xlabel("MR effect size (per-SNP Wald ratio)")
    ax.set_title("Forest plot")
    return ax


def mr_funnel(obj: MRInput, ax=None):
    """Funnel plot: single-SNP estimate vs instrument precision (1/SE)."""
    plt = _lazy_plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    df = mr_singlesnp(obj)
    snp_rows = df[~df["SNP"].str.startswith("All")]
    precision = 1.0 / snp_rows["se"].values
    ax.scatter(snp_rows["b"], precision, s=22, color="#333333", zorder=2)
    ivw = mr_ivw(obj)
    ax.axvline(ivw.estimate, color="#d62728", lw=2,
               label=f"IVW ({ivw.estimate:.3f})")
    if obj.nsnps >= 3:
        eg = mr_egger(obj)
        ax.axvline(eg.estimate, color="#1f77b4", lw=2,
                   label=f"MR-Egger ({eg.estimate:.3f})")
    ax.set_xlabel("MR effect size")
    ax.set_ylabel(r"Instrument precision (1 / SE)")
    ax.set_title("Funnel plot")
    ax.legend(fontsize=8, frameon=False)
    return ax


def mr_loo(obj: MRInput, alpha: float = 0.05, ax=None):
    """Leave-one-out forest plot."""
    plt = _lazy_plt()
    from scipy import stats
    df = mr_leaveoneout(obj)
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 0.32 * len(df) + 1.5))
    ys = np.arange(len(df))[::-1]
    for y, (_, row) in zip(ys, df.iterrows()):
        is_summary = row["SNP"] == "All"
        col = "#d62728" if is_summary else "#333333"
        lo = row["b"] - z * row["se"]
        hi = row["b"] + z * row["se"]
        ax.plot([lo, hi], [y, y], "-", color=col, lw=1.6)
        ax.plot(row["b"], y, "s" if is_summary else "o", color=col,
                ms=7 if is_summary else 5)
    ax.set_yticks(ys)
    ax.set_yticklabels(df["SNP"], fontsize=7)
    ax.axvline(0, color="grey", lw=0.6, ls="--")
    ax.set_xlabel("MR leave-one-out estimate")
    ax.set_title("Leave-one-out plot")
    return ax
