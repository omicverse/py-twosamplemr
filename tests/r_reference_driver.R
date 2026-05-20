#!/usr/bin/env Rscript
# R-parity reference driver for py-twosamplemr.
#
# Runs every MendelianRandomization 0.10.0 estimator on the package's own
# bundled lipid GWAS example data (the `ldlc` / `chdlodds` vectors) and
# writes one TSV row per method:  estimate / SE / CI / p-value plus the
# MR-Egger intercept and the Cochran-Q statistics.  py-twosamplemr runs
# on the identical input so the two sides can be compared directly.
#
# Usage:  Rscript r_reference_driver.R <output.tsv>

args <- commandArgs(trailingOnly = TRUE)
out_path <- if (length(args) >= 1) args[1] else "r_reference.tsv"

suppressMessages(library(MendelianRandomization))

# bundled example data --------------------------------------------------
mri <- mr_input(bx = ldlc, bxse = ldlcse, by = chdlodds, byse = chdloddsse)

rows <- list()
add <- function(method, est, se, ci_l, ci_u, pval, extra = NA) {
  rows[[length(rows) + 1]] <<- data.frame(
    method = method, estimate = est, se = se,
    ci_lower = ci_l, ci_upper = ci_u, pvalue = pval, extra = extra,
    stringsAsFactors = FALSE)
}

# IVW (random, fixed) ---------------------------------------------------
r <- mr_ivw(mri)
add("ivw_random", r@Estimate, r@StdError, r@CILower, r@CIUpper, r@Pvalue,
    r@Heter.Stat[1])
r <- mr_ivw(mri, model = "fixed")
add("ivw_fixed", r@Estimate, r@StdError, r@CILower, r@CIUpper, r@Pvalue, NA)
r <- mr_ivw(mri, penalized = TRUE)
add("ivw_penalized", r@Estimate, r@StdError, r@CILower, r@CIUpper, r@Pvalue, NA)

# MR-Egger --------------------------------------------------------------
r <- mr_egger(mri)
add("egger", r@Estimate, r@StdError.Est, r@CILower.Est, r@CIUpper.Est,
    r@Causal.pval, r@I.sq)
add("egger_intercept", r@Intercept, r@StdError.Int, r@CILower.Int,
    r@CIUpper.Int, r@Pleio.pval, r@Heter.Stat[1])

# median estimators -----------------------------------------------------
r <- mr_median(mri, weighting = "simple")
add("median_simple", r@Estimate, r@StdError, r@CILower, r@CIUpper, r@Pvalue, NA)
r <- mr_median(mri, weighting = "weighted")
add("median_weighted", r@Estimate, r@StdError, r@CILower, r@CIUpper, r@Pvalue,
    NA)
r <- mr_median(mri, weighting = "penalized")
add("median_penalized", r@Estimate, r@StdError, r@CILower, r@CIUpper,
    r@Pvalue, NA)

# mode-based estimate ---------------------------------------------------
r <- mr_mbe(mri)
add("mbe", r@Estimate, r@StdError, r@CILower, r@CIUpper, r@Pvalue, NA)

# maximum likelihood ----------------------------------------------------
r <- mr_maxlik(mri)
add("maxlik", r@Estimate, r@StdError, r@CILower, r@CIUpper, r@Pvalue,
    r@Heter.Stat[1])

# debiased IVW ----------------------------------------------------------
r <- mr_divw(mri)
add("divw", r@Estimate, r@StdError, r@CILower, r@CIUpper, r@Pvalue,
    r@Condition)

# contamination mixture -------------------------------------------------
r <- mr_conmix(mri)
add("conmix", r@Estimate, NA, min(r@CILower), max(r@CIUpper), r@Pvalue, NA)

# MR-Lasso --------------------------------------------------------------
r <- mr_lasso(mri)
add("lasso", r@Estimate, r@StdError, r@CILower, r@CIUpper, r@Pvalue, r@Valid)

# constrained maximum likelihood ---------------------------------------
r <- mr_cML(mri, MA = TRUE, DP = FALSE, n = 17723)
add("cml_ma_bic", r@Estimate, r@StdError, r@CILower, r@CIUpper, r@Pvalue, NA)
r <- mr_cML(mri, MA = FALSE, DP = FALSE, n = 17723)
add("cml_bic", r@Estimate, r@StdError, r@CILower, r@CIUpper, r@Pvalue, NA)

res <- do.call(rbind, rows)
write.table(res, out_path, sep = "\t", row.names = FALSE, quote = FALSE)
cat("Wrote", nrow(res), "rows to", out_path, "\n")
