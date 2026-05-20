"""Benchmark py-twosamplemr against MendelianRandomization 0.10.0.

Runs the full estimator panel on the bundled lipid GWAS example data and,
when the CMAP R environment is available, prints a side-by-side
R-vs-Python comparison plus wall-clock timings.

Usage:
    /scratch/users/steorra/env/omicdev/bin/python examples/benchmark.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytwosamplemr as mr
from pytwosamplemr.datasets import ldl_chd_input

CONDA_SH = "/home/users/steorra/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV = "/scratch/users/steorra/env/CMAP"
R_DRIVER = Path(__file__).resolve().parent.parent / "tests" / "r_reference_driver.R"


def run_python(obj):
    out = {}
    t0 = time.perf_counter()
    out["IVW"] = mr.mr_ivw(obj).estimate
    out["MR-Egger"] = mr.mr_egger(obj).estimate
    out["Maximum likelihood"] = mr.mr_maxlik(obj).estimate
    out["Debiased IVW"] = mr.mr_divw(obj).estimate
    out["Contamination mixture"] = mr.mr_conmix(obj).estimate
    out["MR-Lasso"] = mr.mr_lasso(obj).estimate
    out["cML-MA-BIC"] = mr.mr_cml(obj, n=17723, ma=True, dp=False).estimate
    out["Weighted median"] = mr.mr_median(obj, weighting="weighted").estimate
    out["Mode-based estimate"] = mr.mr_mbe(obj).estimate
    elapsed = time.perf_counter() - t0
    return out, elapsed


def run_r():
    if not R_DRIVER.exists():
        return None
    import pandas as pd
    tmp = Path("/tmp/_bench_r_reference.tsv")
    t0 = time.perf_counter()
    res = subprocess.run(
        ["bash", "-lc",
         f"source {CONDA_SH} && conda activate {CONDA_ENV} "
         f"&& Rscript {R_DRIVER} {tmp}"],
        capture_output=True, text=True, timeout=600, check=False,
    )
    elapsed = time.perf_counter() - t0
    if res.returncode != 0:
        print("R driver failed:", res.stderr)
        return None
    df = pd.read_csv(tmp, sep="\t").set_index("method")
    mapping = {
        "IVW": "ivw_random",
        "MR-Egger": "egger",
        "Maximum likelihood": "maxlik",
        "Debiased IVW": "divw",
        "Contamination mixture": "conmix",
        "MR-Lasso": "lasso",
        "cML-MA-BIC": "cml_ma_bic",
        "Weighted median": "median_weighted",
        "Mode-based estimate": "mbe",
    }
    return {k: df.loc[v]["estimate"] for k, v in mapping.items()}, elapsed


def main():
    obj = ldl_chd_input()
    print(f"Example data: {obj.nsnps} SNPs, "
          f"{obj.exposure} -> {obj.outcome}\n")

    py_res, py_time = run_python(obj)
    print(f"Python estimator panel: {py_time * 1000:.1f} ms\n")

    r_out = run_r()
    if r_out is None:
        print("(R unavailable — Python-only run)")
        for m, est in py_res.items():
            print(f"  {m:24s} {est:10.6f}")
        return

    r_res, r_time = r_out
    print(f"R estimator panel:      {r_time * 1000:.1f} ms\n")
    print(f"{'Method':24s} {'R':>11s} {'Python':>12s} {'rel-diff':>11s}")
    print("-" * 60)
    for m in py_res:
        r = r_res[m]
        p = py_res[m]
        rel = abs(p - r) / max(abs(r), 1e-12)
        print(f"{m:24s} {r:11.6f} {p:12.6f} {rel:11.2e}")


if __name__ == "__main__":
    main()
