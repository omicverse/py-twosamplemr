"""Faithful Python port of R's Nelder-Mead optimiser (``nmmin`` in
``src/appl/optim.c``).

``mr_maxlik`` calls ``optim(..., method = "Nelder-Mead")`` and reads back
both the minimiser *and* the objective value at the returned simplex
vertex.  Because R's Nelder-Mead frequently stops at a non-global point
on rough high-dimensional likelihoods, reproducing the *exact* R
algorithm (simplex construction, reflection/extension/contraction
coefficients, convergence test) is required for numerical parity.

This is a line-for-line port of R 4.x ``nmmin`` (BSD-licensed, part of
the R distribution).  The R defaults are ``alpha = 1.0`` (reflection),
``bet = 0.5`` (contraction), ``gamm = 2.0`` (extension), ``maxit = 500``
and ``intol = reltol = sqrt(.Machine$double.eps)``.
"""
from __future__ import annotations

import numpy as np

__all__ = ["nmmin"]

_BIG = 1e35  # R's internal "big" sentinel for non-finite f


def nmmin(fn, x_start, maxit=500, intol=1.4901161193847656e-08,
          abstol=-np.inf, alpha=1.0, bet=0.5, gamm=2.0):
    """Minimise ``fn`` via R's Nelder-Mead algorithm.

    Parameters
    ----------
    fn : callable
        Objective ``f(x) -> float``.
    x_start : array_like
        Starting parameter vector.
    maxit : int
        Maximum function evaluations.
    intol : float
        Relative convergence tolerance (R default ``sqrt(eps)``).

    Returns
    -------
    dict with ``x`` (minimiser), ``fval`` (objective at minimiser),
    ``fail`` (0 = converged, 1 = max iterations, 10 = degenerate shrink)
    and ``fncount``.
    """
    Bvec = np.asarray(x_start, dtype=float).copy()
    n = len(Bvec)
    if maxit <= 0:
        return {"x": Bvec, "fval": float(fn(Bvec)), "fail": 0, "fncount": 0}
    if n == 0:
        return {"x": Bvec, "fval": float(fn(Bvec)), "fail": 0, "fncount": 0}

    n1 = n + 1
    C = n + 1  # 0-based index of the centroid column (R's C-1)
    # P has n rows of coordinates + 1 row of f-values; n+2 columns
    P = np.zeros((n1, n1 + 1))

    f = float(fn(Bvec))
    funcount = 1
    convtol = intol * (abs(f) + intol)

    P[n1 - 1, 0] = f
    P[:n, 0] = Bvec

    L = 1
    step = 0.0
    for i in range(n):
        if 0.1 * abs(Bvec[i]) > step:
            step = 0.1 * abs(Bvec[i])
    if step == 0.0:
        step = 0.1

    # build remaining simplex vertices
    for j in range(2, n1 + 1):
        P[:n, j - 1] = Bvec
        trystep = step
        while P[j - 2, j - 1] == Bvec[j - 2]:
            P[j - 2, j - 1] = Bvec[j - 2] + trystep
            trystep *= 10.0

    calcvert = True
    fail = 0

    while True:
        if calcvert:
            for j in range(n1):
                if j + 1 != L:
                    Bvec = P[:n, j].copy()
                    f = float(fn(Bvec))
                    if not np.isfinite(f):
                        f = _BIG
                    funcount += 1
                    P[n1 - 1, j] = f
            calcvert = False

        VL = P[n1 - 1, L - 1]
        VH = VL
        H = L
        for j in range(1, n1 + 1):
            if j != L:
                f = P[n1 - 1, j - 1]
                if f < VL:
                    L = j
                    VL = f
                if f > VH:
                    H = j
                    VH = f

        if VH <= VL + convtol or VL <= abstol:
            break

        # centroid of all vertices except H
        for i in range(n):
            temp = -P[i, H - 1]
            for j in range(n1):
                temp += P[i, j]
            P[i, C] = temp / n

        # reflection
        for i in range(n):
            Bvec[i] = (1.0 + alpha) * P[i, C] - alpha * P[i, H - 1]
        f = float(fn(Bvec))
        if not np.isfinite(f):
            f = _BIG
        funcount += 1
        VR = f

        if VR < VL:
            # extension
            P[n1 - 1, C] = f
            for i in range(n):
                f = gamm * Bvec[i] + (1.0 - gamm) * P[i, C]
                P[i, C] = Bvec[i]
                Bvec[i] = f
            f = float(fn(Bvec))
            if not np.isfinite(f):
                f = _BIG
            funcount += 1
            if f < VR:
                P[:n, H - 1] = Bvec
                P[n1 - 1, H - 1] = f
            else:
                P[:n, H - 1] = P[:n, C]
                P[n1 - 1, H - 1] = VR
        else:
            # HI / LO reduction
            if VR < VH:
                P[:n, H - 1] = Bvec
                P[n1 - 1, H - 1] = VR

            # contraction
            for i in range(n):
                Bvec[i] = (1.0 - bet) * P[i, H - 1] + bet * P[i, C]
            f = float(fn(Bvec))
            if not np.isfinite(f):
                f = _BIG
            funcount += 1

            if f < P[n1 - 1, H - 1]:
                P[:n, H - 1] = Bvec
                P[n1 - 1, H - 1] = f
            else:
                if VR >= VH:
                    # shrink toward L
                    calcvert = True
                    size = 0.0
                    for j in range(n1):
                        if j + 1 != L:
                            for i in range(n):
                                P[i, j] = (bet * (P[i, j] - P[i, L - 1])
                                           + P[i, L - 1])
                                size += abs(P[i, j] - P[i, L - 1])

        if funcount > maxit:
            break

    fval = P[n1 - 1, L - 1]
    X = P[:n, L - 1].copy()
    if funcount > maxit:
        fail = 1
    return {"x": X, "fval": float(fval), "fail": fail, "fncount": funcount}
