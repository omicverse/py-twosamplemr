"""Result containers for the py-twosamplemr estimators.

Each estimator returns a lightweight dataclass that mirrors the slots of
the corresponding *MendelianRandomization* S4 object (``IVW``, ``Egger``,
``WeightedMedian`` ...), exposing the causal estimate, its standard
error, confidence interval, p-value and method-specific extras.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

__all__ = [
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
]


@dataclass
class MRResult:
    """Base result: causal estimate, SE, CI and p-value."""

    method: str
    exposure: str
    outcome: str
    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    pvalue: float
    nsnps: int
    alpha: float = 0.05

    def _summary_rows(self):
        return [(
            self.method, self.estimate, self.se,
            self.ci_lower, self.ci_upper, self.pvalue,
        )]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.method}: estimate={self.estimate:.5g} "
            f"se={self.se:.5g} "
            f"95% CI [{self.ci_lower:.5g}, {self.ci_upper:.5g}] "
            f"p={self.pvalue:.4g}  ({self.nsnps} SNPs)"
        )


@dataclass
class IVWResult(MRResult):
    """Inverse-variance weighted result (``IVW``)."""

    model: str = "random"
    robust: bool = False
    penalized: bool = False
    correlation: bool = False
    rse: float = float("nan")
    heter_stat: float = float("nan")
    heter_pvalue: float = float("nan")
    fstat: float = float("nan")


@dataclass
class EggerResult(MRResult):
    """MR-Egger regression result (``Egger``)."""

    intercept: float = 0.0
    intercept_se: float = float("nan")
    intercept_ci_lower: float = float("nan")
    intercept_ci_upper: float = float("nan")
    intercept_pvalue: float = float("nan")
    causal_pvalue: float = float("nan")
    pleio_pvalue: float = float("nan")
    model: str = "random"
    robust: bool = False
    penalized: bool = False
    rse: float = float("nan")
    heter_stat: float = float("nan")
    heter_pvalue: float = float("nan")
    i_sq: float = float("nan")

    def _summary_rows(self):
        return [
            ("MR-Egger", self.estimate, self.se,
             self.ci_lower, self.ci_upper, self.causal_pvalue),
            ("(intercept)", self.intercept, self.intercept_se,
             self.intercept_ci_lower, self.intercept_ci_upper,
             self.intercept_pvalue),
        ]


@dataclass
class MedianResult(MRResult):
    """Median-based estimator result (``WeightedMedian``)."""

    weighting: str = "weighted"


@dataclass
class MBEResult(MRResult):
    """Mode-based estimate result (``MRMBE``)."""

    weighting: str = "weighted"
    stderror: str = "delta"
    phi: float = 1.0


@dataclass
class MaxLikResult(MRResult):
    """Maximum-likelihood estimator result (``MaxLik``)."""

    model: str = "random"
    psi: float = 0.0
    rse: float = float("nan")
    heter_stat: float = float("nan")
    heter_pvalue: float = float("nan")


@dataclass
class DIVWResult(MRResult):
    """Debiased IVW estimator result (``DIVW``)."""

    over_dispersion: bool = True
    condition: float = float("nan")


@dataclass
class ConMixResult(MRResult):
    """Contamination-mixture estimator result (``MRConMix``)."""

    psi: float = 0.0
    ci_ranges: List[tuple] = field(default_factory=list)
    valid: List[int] = field(default_factory=list)
    valid_snps: List[str] = field(default_factory=list)


@dataclass
class LassoResult(MRResult):
    """MR-Lasso post-selection estimator result (``MRLasso``)."""

    lambda_: float = float("nan")
    n_valid: int = 0
    valid_snps: List[str] = field(default_factory=list)
    reg_estimate: float = float("nan")
    reg_intercept: Optional[np.ndarray] = None


@dataclass
class CMLResult(MRResult):
    """Constrained-maximum-likelihood estimator result (``MRcML``)."""

    ma: bool = True
    dp: bool = True
    bic_invalid: List[int] = field(default_factory=list)
    gof1_pvalue: float = float("nan")
    gof2_pvalue: float = float("nan")
