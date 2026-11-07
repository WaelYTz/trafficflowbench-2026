"""Task 4 path-flow estimators.

Every panel's path set is a one-directional corridor: a path is an (origin zone, destination zone)
pair and the counts only see mainline links, so the identifiable information is the flow across
each of the ~92 zone boundaries ("segments"). The prior fills in everything else.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from common import PANELS, REL, SPLITS


def problem(panel: str, split: str) -> dict:
    nd = REL / "corridors" / panel / "network"
    ps = pd.read_csv(nd / "path_set.csv", dtype=str)
    inc = pd.read_csv(nd / "path_link_incidence.csv", dtype=str)
    t4 = REL / "task4" / panel / split
    c = pd.read_csv(t4 / "synthetic_link_counts.csv", dtype={"link_id": str})
    b = pd.read_csv(t4 / "synthetic_weak_prior.csv", dtype={"path_id": str}).set_index("path_id").reindex(ps.path_id)
    pid = {p: i for i, p in enumerate(ps.path_id)}
    lid = {l: i for i, l in enumerate(c.link_id)}
    sub = inc[inc.link_id.isin(lid)]
    A = np.zeros((len(c), len(ps)))
    A[sub.link_id.map(lid).to_numpy(), sub.path_id.map(pid).to_numpy()] = 1.0
    # Unique segments: links whose path sets are identical carry the same count.
    _, first, inv = np.unique(A, axis=0, return_index=True, return_inverse=True)
    return dict(panel=panel, split=split, paths=ps, A=A, c=c["count"].to_numpy(float),
                As=A[first], cs=c["count"].to_numpy(float)[first], b=b.path_flow.to_numpy(float),
                departure=str(b.departure_time.iloc[0]),
                o=ps.origin_zone.str[1:].astype(int).to_numpy(), d=ps.destination_zone.str[1:].astype(int).to_numpy())


def solve_l2(P: dict, lam: float = 0.05, prior=None) -> np.ndarray:
    """min ||A f - c||^2 + lam ||f - b||^2, f >= 0, over every counted link (public baseline objective)."""
    A, c = P["A"], P["c"]
    b = P["b"] if prior is None else prior
    AtA, Atc = A.T @ A, A.T @ c

    def fg(f):
        r = A @ f - c
        g = 2 * (A.T @ r) + 2 * lam * (f - b)
        return float(r @ r + lam * ((f - b) @ (f - b))), g
    res = minimize(fg, np.maximum(b, 0), jac=True, method="L-BFGS-B", bounds=[(0, None)] * len(b),
                   options=dict(maxiter=20000, ftol=1e-15, gtol=1e-10, maxcor=50))
    return res.x


def solve_kl(P: dict, prior=None, iters: int = 200) -> np.ndarray:
    """Entropy ODME: f = b * exp(A_s^T mu) with A_s f = c_s exactly (Newton on the convex dual)."""
    As, cs = P["As"], P["cs"]
    b = P["b"] if prior is None else prior
    mu = np.zeros(len(cs))
    for _ in range(iters):
        f = b * np.exp(As.T @ mu)
        g = As @ f - cs
        if np.abs(g).max() < 1e-8 * cs.max():
            break
        H = (As * f) @ As.T + 1e-9 * np.eye(len(cs))
        step = np.linalg.solve(H, g)
        t = 1.0
        dual = lambda m: (b * np.exp(As.T @ m)).sum() - cs @ m
        d0 = dual(mu)
        while dual(mu - t * step) > d0 - 1e-4 * t * (g @ step) and t > 1e-8:
            t *= 0.5
        mu = mu - t * step
    return b * np.exp(As.T @ mu)


def pooled_prior(panel: str, mode: str = "geo") -> np.ndarray:
    logs = [np.log(problem(panel, s)["b"]) for s in SPLITS]
    return np.exp(np.mean(logs, axis=0))


def diagnostics(P: dict, f: np.ndarray) -> dict:
    s_link = max(0.0, 1 - np.abs(P["A"] @ f - P["c"]).sum() / P["c"].sum())
    return dict(S_link=s_link, dev=np.abs(f - P["b"]).sum() / P["b"].sum(), total=f.sum())
