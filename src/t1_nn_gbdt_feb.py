"""GBDT reference for the neural Task 1 imputer: re-create the exact Feb holdout cells of t1_train.py
(cap-train 1500 / cap-hold 8000, same rng stream) and predict them with the saved Jun-Jan models of
out/t1m_exp (trained with --train-month-prof, Feb holdout 0.9464). Optional synthetic Task 2 gaps
(all links hidden for 18 slots) to test the gap cells of validation/private.
Writes out_t1nn/feb_gbdt[_gap].npz (cells in travel order + predictions + truth)."""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import xgboost as xgb

from common import PANELS, ROOT, days, load
from t1_features import Grid, build, train_month_profiles
from t1_train import CHUNK, HOLDOUT_FROM, SEED, bases, score

OUTD = ROOT / "out_t1nn"


def holdout_cells(panel: str, cap_train: int = 1500, cap_hold: int = 8000):
    """(day index in train split, slot, travel-order link) of the Feb holdout, identical to t1_train.panel_rows."""
    rng = np.random.default_rng(SEED + PANELS.index(panel))
    g0 = Grid(panel, "train", d0=0, d1=1)
    tg = load(panel, "train", "target", mmap=True)
    dl = days(panel, "train")
    D, T, L = [], [], []
    for d0 in range(0, len(dl), CHUNK):
        d1 = min(d0 + CHUNK, len(dl))
        d, t, l = np.nonzero(np.asarray(tg[d0:d1])[:, :, g0.order])
        keep = np.zeros(len(d), bool)
        for dd in np.unique(d):
            idx = np.flatnonzero(d == dd)
            cap = cap_hold if dl[d0 + dd] >= HOLDOUT_FROM else cap_train
            keep[rng.choice(idx, min(cap, len(idx)), replace=False)] = True
        d, t, l = d[keep] + d0, t[keep], l[keep]
        h = np.array([dl[x] >= HOLDOUT_FROM for x in d], bool)
        D.append(d[h]); T.append(t[h]); L.append(l[h])
    return np.concatenate(D), np.concatenate(T), np.concatenate(L)


def gap_slots(panel: str, n_days: int, seed: int = 7):
    """Synthetic Task 2 gaps: per day 1-3 blocks of 18 slots (all links hidden), starts in [60, 250]."""
    rng = np.random.default_rng(seed + PANELS.index(panel))
    out = []
    for _ in range(n_days):
        k = rng.integers(1, 4)
        m = np.zeros(288, bool)
        for s in rng.integers(60, 251, size=k):
            m[s:s + 18] = True
        out.append(m)
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    OUTD.mkdir(exist_ok=True)
    t0 = time.time()
    mv = xgb.Booster(model_file=str(ROOT / "out" / "t1m_exp" / "t1h_v.json"))
    mq = xgb.Booster(model_file=str(ROOT / "out" / "t1m_exp" / "t1h_q.json"))
    for m in (mv, mq):
        m.set_param({"device": a.device})
    print(f"models loaded {time.time() - t0:.0f}s", flush=True)
    R = {k: [] for k in ("panel", "d", "t", "l", "pv", "pq", "yv", "yq", "lanes", "regime", "ingap")}
    for pi, panel in enumerate(PANELS):
        d, t, l = holdout_cells(panel)
        dl = days(panel, "train")
        f0 = min(d)
        prof, mid = train_month_profiles(panel)
        d0 = (f0 // CHUNK) * CHUNK
        g = Grid(panel, "train", d0=d0, d1=len(dl))
        ingap = np.zeros(len(d), bool)
        if a.gap:
            feb = np.arange(f0, len(dl))
            gs = gap_slots(panel, len(feb))
            gm = np.zeros((g.D, 288), bool)
            gm[feb - d0] = gs
            g.v = np.array(g.v, np.float32); g.q = np.array(g.q, np.float32)
            g.v[gm] = np.nan; g.q[gm] = np.nan
            g.obs = np.isfinite(g.v) & np.isfinite(g.q)
            ingap = gm[d - d0, t]
        X, names = build(g, d - d0, t, l, prof, loo=False, key=mid[d0:d0 + g.D])
        bv, bq = bases(X, names)
        dm = xgb.DMatrix(X)
        pv = np.clip((bv + mv.predict(dm)) * g.vf[l], 3, 140)
        pq = np.clip(bq + mq.predict(dm) * g.lanes[l], 60, None)
        sc = g.order[l]
        yv = np.asarray(load(panel, "train", "v", mmap=True)[d, t, sc], float)
        yq = np.asarray(load(panel, "train", "q", mmap=True)[d, t, sc], float)
        for k, x in (("panel", np.full(len(d), pi)), ("d", d), ("t", t), ("l", l), ("pv", pv), ("pq", pq), ("yv", yv),
                     ("yq", yq), ("lanes", g.lanes[l]), ("regime", g.regime[d - d0]), ("ingap", ingap)):
            R[k].append(x)
        print(f"{panel}: {len(d):,} cells ({ingap.sum():,} in gaps), {time.time() - t0:.0f}s", flush=True)
        del X, dm, g
    A = {k: np.concatenate(v) for k, v in R.items()}
    s, per = score(A["pv"], A["pq"], A["yv"], A["yq"], A["lanes"], A["panel"], A["regime"])
    rv = np.sqrt(np.mean((A["pv"] - A["yv"]) ** 2)); rq = np.sqrt(np.mean(((A["pq"] - A["yq"]) / A["lanes"]) ** 2))
    print(f"GBDT Feb S_state {s:.5f} rmse_v {rv:.3f} rmse_q_lane {rq:.2f} n={len(A['yv']):,}", flush=True)
    if a.gap:
        g_ = A["ingap"]
        rv = np.sqrt(np.mean((A["pv"][g_] - A["yv"][g_]) ** 2)); rq = np.sqrt(np.mean(((A["pq"][g_] - A["yq"][g_]) / A["lanes"][g_]) ** 2))
        print(f"  gap cells: rmse_v {rv:.3f} rmse_q_lane {rq:.2f} n={g_.sum():,}", flush=True)
    np.savez_compressed(OUTD / ("feb_gbdt_gap.npz" if a.gap else "feb_gbdt.npz"), **A)


if __name__ == "__main__":
    main()
