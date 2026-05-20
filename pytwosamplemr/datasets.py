"""Built-in example data — the *MendelianRandomization* lipid GWAS vectors.

These are the exact ``ldlc``, ``hdlc``, ``trig`` and ``chdlodds``
summary-statistics vectors bundled with the R package
*MendelianRandomization* (28 SNPs associated with blood lipids, with
their associations with coronary-heart-disease log-odds).  They are used
throughout the documentation and R-parity tests so both languages
analyse identical input.
"""
from __future__ import annotations

import numpy as np

from .core import MRInput, mr_input

__all__ = ["ldlc", "ldlcse", "hdlc", "hdlcse", "trig", "trigse",
           "chdlodds", "chdloddsse", "ldl_chd_input", "example_inputs"]

ldlc = np.array([0.026, -0.044, -0.038, -0.023, -0.017, -0.031, -0.018, 0.046, 0.059, 0.004, 0.011, -0.005, 0.004, 0.022, -0.005, -0.002, -0.002, 0.004, 0.011, 0.009, -0.011, -0.003, -0.012, 0.0003, -0.015, -0.008, 0.009, -0.036])
ldlcse = np.array([0.004, 0.004, 0.004, 0.003, 0.003, 0.006, 0.004, 0.007, 0.004, 0.003, 0.004, 0.005, 0.005, 0.005, 0.004, 0.004, 0.003, 0.004, 0.004, 0.003, 0.004, 0.003, 0.004, 0.003, 0.003, 0.004, 0.003, 0.007])
hdlc = np.array([0.002, 0.005, 0.003, 0.001, 0.011, 0.031, -0.003, -0.007, -0.021, 0.018, -0.017, -0.047, 0.022, -0.029, 0.016, 0.034, 0.035, 0.019, 0.028, 0.0001, 0.016, 0.005, -0.01, -0.023, 0.012, 0.018, -0.006, 0.004])
hdlcse = np.array([0.004, 0.004, 0.004, 0.003, 0.003, 0.006, 0.004, 0.006, 0.004, 0.003, 0.003, 0.005, 0.004, 0.004, 0.003, 0.004, 0.003, 0.004, 0.004, 0.003, 0.003, 0.003, 0.004, 0.003, 0.003, 0.003, 0.003, 0.006])
trig = np.array([0.016, -0.004, -0.009, 0.003, -0.039, -0.142, 0.007, 0.095, 0.042, -0.023, 0.036, 0.097, 0.013, 0.142, -0.005, 0.019, 0.003, -0.018, -0.02, 0.035, -0.036, -0.054, 0.054, 0.067, -0.04, -0.067, 0.028, -0.07])
trigse = np.array([0.008, 0.008, 0.008, 0.006, 0.006, 0.012, 0.007, 0.013, 0.008, 0.006, 0.007, 0.01, 0.009, 0.009, 0.007, 0.007, 0.006, 0.008, 0.008, 0.007, 0.007, 0.006, 0.008, 0.006, 0.006, 0.007, 0.006, 0.013])
chdlodds = np.array([0.0677, -0.1625, -0.1054, -0.0619, -0.0834, -0.1278, -0.0408, 0.077, 0.157, -0.0305, 0.01, 0.1823, -0.0408, 0.1989, 0.01, 0.0488, 0.01, -0.0408, -0.0305, -0.0408, -0.0202, -0.0619, 0.0296, 0.0677, -0.0726, -0.0726, 0.0, 0.0198])
chdloddsse = np.array([0.0286, 0.03, 0.031, 0.0243, 0.0222, 0.0667, 0.0373, 0.0543, 0.0306, 0.0236, 0.0277, 0.0403, 0.0344, 0.0335, 0.0378, 0.0292, 0.0253, 0.0319, 0.0316, 0.0241, 0.0285, 0.0217, 0.0298, 0.0239, 0.022, 0.0246, 0.0255, 0.0647])


def ldl_chd_input() -> MRInput:
    """Return the LDL-cholesterol -> CHD :class:`MRInput` example."""
    return mr_input(
        bx=ldlc, bxse=ldlcse, by=chdlodds, byse=chdloddsse,
        exposure="ldl-cholesterol", outcome="coronary heart disease",
    )


def example_inputs() -> dict:
    """Return a dict of all three lipid -> CHD example :class:`MRInput`s."""
    return {
        "ldl": mr_input(bx=ldlc, bxse=ldlcse, by=chdlodds, byse=chdloddsse,
                        exposure="ldl-c", outcome="CHD"),
        "hdl": mr_input(bx=hdlc, bxse=hdlcse, by=chdlodds, byse=chdloddsse,
                        exposure="hdl-c", outcome="CHD"),
        "trig": mr_input(bx=trig, bxse=trigse, by=chdlodds, byse=chdloddsse,
                         exposure="triglycerides", outcome="CHD"),
    }
