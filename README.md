# py-twosamplemr

**Pure-Python two-sample Mendelian randomization** — a faithful port of
the statistical-methods core of the R/CRAN package
[`MendelianRandomization`](https://cran.r-project.org/package=MendelianRandomization)
(Yavorska & Burgess, *Int. J. Epidemiol.* 2017) together with the
harmonisation / diagnostics workflow of
[`TwoSampleMR`](https://mrcieu.github.io/TwoSampleMR/)
(Hemani et al., *eLife* 2018).

* PyPI dist name: `pytwosamplemr` &nbsp;|&nbsp; import name: `pytwosamplemr`
* Pure Python — `numpy` / `scipy` / `pandas` / `matplotlib`. No rpy2.
* **Numerical parity with R is the design goal.** Closed-form estimators
  reproduce `MendelianRandomization` 0.10.0 bit-for-bit (relative
  difference `< 1e-5`); bootstrap-based estimators agree to within
  Monte-Carlo error.

## Installation

```bash
pip install pytwosamplemr
# or, from a checkout:
pip install -e .
```

## Quick start

```python
import pytwosamplemr as mr
from pytwosamplemr.datasets import ldl_chd_input

# the MendelianRandomization bundled lipid GWAS example (28 SNPs)
obj = ldl_chd_input()

mr.mr_ivw(obj)        # inverse-variance weighted
mr.mr_egger(obj)      # MR-Egger regression (+ pleiotropy intercept)
mr.mr_maxlik(obj)     # maximum likelihood
mr.mr_divw(obj)       # debiased IVW
mr.mr_conmix(obj)     # contamination mixture
mr.mr_lasso(obj)      # MR-Lasso post-selection
mr.mr_cml(obj, n=17723)   # constrained maximum likelihood

# panel + diagnostics
mr.mr_allmethods(obj, method="main")
mr.mr_heterogeneity(obj)
mr.mr_pleiotropy_test(obj)
mr.mr_singlesnp(obj)
mr.mr_leaveoneout(obj)

# plots
mr.mr_scatter(obj)    # SNP effects + fitted method lines
mr.mr_forest(obj)
mr.mr_funnel(obj)
mr.mr_loo(obj)
```

## What is implemented

### Input

| Function | Description |
|----------|-------------|
| `mr_input` / `MRInput` | harmonised SNP-exposure / SNP-outcome effect container |

### Estimators (full coverage)

| Function | Method | R parity |
|----------|--------|----------|
| `mr_ivw` | inverse-variance weighted (fixed / random, penalized, correlated SNPs, Cochran's Q) | bit-exact |
| `mr_egger` | MR-Egger regression: causal estimate + pleiotropy intercept + I²_GX | bit-exact |
| `mr_median` | simple / weighted / penalized weighted median | estimate exact, bootstrap SE ≈ |
| `mr_mbe` | mode-based estimate | estimate exact, bootstrap SE ≈ |
| `mr_maxlik` | maximum likelihood (R Nelder-Mead optimiser ported) | bit-exact |
| `mr_divw` | debiased IVW | bit-exact |
| `mr_conmix` | contamination mixture | bit-exact |
| `mr_lasso` | MR-Lasso post-selection IVW | post-estimate exact |
| `mr_cml` | constrained maximum likelihood (cML-MA / cML-BIC) | cML-BIC bit-exact; DP ≈ |
| `mr_allmethods` | run the standard panel and tabulate | — |

### Diagnostics / workflow (TwoSampleMR side)

`harmonise_data` · `mr_heterogeneity` · `mr_pleiotropy_test` ·
`mr_steiger` / `directionality_test` · `mr_singlesnp` · `mr_leaveoneout`

### Plots

`mr_plot` / `mr_scatter` · `mr_forest` · `mr_funnel` · `mr_loo`

## R parity

`py-twosamplemr` was validated against `MendelianRandomization` 0.10.0
on the package's own bundled lipid GWAS example data. Closed-form
estimators agree with R to machine precision:

| Method | R estimate | Python estimate | rel. diff |
|--------|-----------:|----------------:|----------:|
| IVW (random)      | 2.834214 | 2.834214 | 6e-16 |
| IVW (fixed)       | 2.834214 | 2.834214 | 6e-16 |
| MR-Egger          | 3.252890 | 3.252890 | 3e-16 |
| Maximum likelihood| 3.224944 | 3.224944 | 6e-12 |
| Debiased IVW      | 2.939646 | 2.939646 | 1e-15 |
| Contamination mix | 2.730000 | 2.730000 | 2e-16 |
| MR-Lasso          | 2.670896 | 2.670896 | 2e-15 |
| cML-MA-BIC        | 2.894808 | 2.894808 | 1e-15 |

A standout detail: `mr_maxlik` calls R's `optim(method="Nelder-Mead")`,
which on this rough 29-dimensional likelihood *stops at a non-global
local point* — so reproducing R requires a line-for-line port of R's
`nmmin` C routine (`pytwosamplemr/_nmmin.py`). With that port the
maximum-likelihood estimate matches R to 12 significant figures.

Bootstrap-based estimators (`mr_median`, `mr_mbe`, `mr_cml` with data
perturbation) cannot be bit-exact across RNGs: their point estimates
still match R exactly (the median / mode location does not depend on the
bootstrap), and the bootstrap SE agrees to within ~10%.

The R-parity test suite (`tests/test_r_parity.py`) runs the R reference
driver live and asserts these tolerances; it skips gracefully when R is
unavailable.

## Testing

```bash
python -m pytest tests/ -q
```

## License

GPL-3, matching the upstream `MendelianRandomization` package
(licensed GPL-2 | GPL-3).

## References

* Yavorska OO, Burgess S. *MendelianRandomization: an R package for
  performing Mendelian randomization analyses using summarized data.*
  Int J Epidemiol. 2017;46(6):1734-1739.
* Hemani G, et al. *The MR-Base platform supports systematic causal
  inference across the human phenome.* eLife. 2018;7:e34408.
