"""Pseudo-windows cut from the masked layer, to measure Task 2 models on the scored months.

The scored months ship only the masked layer, but it still holds their real queues. Windows are cut with
the training selector on the masked layer (Feb from train's masked layer, March from validation) with
the coverage bar lowered to 0.45, because the Task 1 mask removes 20-50% of the cells. Labels are the
observed queued cells in the horizon. Feb masked vs Feb raw calibrates the proxy. March masked is
evidence for the April windows (March precedes them) but not for March's own windows.
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd

from common import OUT, QUEUE_PANELS, T, days, load
from t2_data import H, Recent, Static, priors, train_arrays, window_features

SOURCES = {"feb": ("train", "2031-02"), "mar": ("validation", "2031-03"), "apr": ("private", "2031-04")}


def masked(st: Static, split: str):
    o = st.order
    if split == "train":
        v, q, e, vm, qm = train_arrays(st)
        return vm, qm, e & np.isfinite(vm)
    v = load(st.panel, split, "v")[:, :, o]
    q = load(st.panel, split, "q")[:, :, o]
    e = load(st.panel, split, "elig")[:, :, o].astype(bool)
    return v, q, e & np.isfinite(v)


def pseudo(panel: str, src: str, ongoing_step: int = 12, official: bool = False):
    split, month = SOURCES[src]
    st = Static(panel)
    v, q, e = masked(st, split)
    dl = days(panel, split)
    keep_days = np.array([d.startswith(month) for d in dl])
    vt, _, et, _, _ = train_arrays(st)
    mcode = pd.factorize(pd.Index([x[:7] for x in days(panel, "train")]), sort=True)[0]
    onset_c, queued_c, seen_c = priors(st, vt, et, mcode)
    del vt, et
    if split == "train":  # leave Feb out of the priors, as build_train does
        m = mcode.max()
        op = (onset_c.sum(0) - onset_c[m]) / (len(mcode) - (mcode == m).sum())
        qp = (queued_c.sum(0) - queued_c[m]) / np.maximum(seen_c.sum(0) - seen_c[m], 1)
    else:
        op = onset_c.sum(0) / len(mcode); qp = queued_c.sum(0) / np.maximum(seen_c.sum(0), 1)
    rec = Recent(st)
    vv = np.where(keep_days[:, None, None], v, np.nan)  # cut windows from the chosen month only
    wins = _official(vv, e, st.cut) if official else _origins(vv, e, st.cut[None, :], ongoing_step)
    dow = pd.to_datetime(pd.Series(dl)).dt.dayofweek.to_numpy()
    rows = {"onset": [], "ongoing": []}
    names = None
    for d, t, cond in wins:
        V, Q = v[d, t - H:t + 1], q[d, t - H:t + 1]
        X, names = window_features(st, V, Q, t, dow[d], op, qp, rec.features(split, d, t))
        lab = e[d, t + 1:t + 7] & (v[d, t + 1:t + 7] <= st.cut[None, :])
        key = "onset" if cond == "queue_onset" else "ongoing"
        rows[key].append(dict(X=X, y=lab, elig=e[d, t + 1:t + 7].copy(), day=dl[d], slot=t, panel=panel,
                              hist=(v[d, t - H:t].copy(), q[d, t - H:t].copy(), e[d, t - H:t].astype(float))))
    return rows, names


def _origins(v, e, cut, ongoing_step):
    """training_origins with the coverage bar at 0.45 (the masked layer keeps 50-80% of cells)."""
    out = []
    for d in range(v.shape[0]):
        if np.isfinite(v[d]).any():
            out += [(d, t, c) for t, c in _origins_day(v[d], e[d], cut[0], ongoing_step)]
    return out


def _official(v, e, cut, min_cov=0.45, pers_max=0.75):
    """The published selector (earliest origin first, 360-minute spacing shared by both conditions,
    scanning the whole month) on the masked layer, as t2_data.select_windows does on raw data."""
    qobs = np.isfinite(v) & (v <= cut)
    qtrue = qobs & e
    out, last = [], -10**9
    for d in range(v.shape[0]):
        if not np.isfinite(v[d]).any():
            continue
        cov = np.convolve(np.isfinite(v[d]).mean(axis=1), np.ones(H) / H, mode="full")[H - 1:T]
        for t in range(H, T - 6):
            if d * T + t - last < 72 or cov[t - H] < min_cov:
                continue
            hor = qtrue[d, t + 1:t + 7]
            if not hor.any():
                continue
            hq = qobs[d, t - H:t]
            if not hq.any():
                cond = "queue_onset"
            elif hq.sum() >= 2 and hq.sum(axis=0).max() >= 2:
                pers = np.broadcast_to(qobs[d, t - 1], hor.shape) & e[d, t + 1:t + 7]
                union = (pers | hor).sum()
                if union and (pers & hor).sum() / union > pers_max:
                    continue
                cond = "queue_ongoing"
            else:
                continue
            last = d * T + t
            out.append((d, t, cond))
    return out


def _origins_day(v, e, cut, step, min_cov=0.45):
    qobs = np.isfinite(v) & (v <= cut)
    qtrue = qobs & e
    cov = np.convolve(np.isfinite(v).mean(axis=1), np.ones(H) / H, mode="full")[H - 1:T]
    out, last = [], -10**9
    for t in range(H, T - 6):
        if cov[t - H] < min_cov:
            continue
        hor = qtrue[t + 1:t + 7]
        if not hor.any():
            continue
        hq = qobs[t - H:t]
        if not hq.any():
            if hor[5].any() and not hor[:5].any():
                out.append((t, "queue_onset"))
            continue
        if hq.sum() < 2 or hq.sum(axis=0).max() < 2 or t - last < step:
            continue
        pers = np.broadcast_to(qobs[t - 1], hor.shape) & e[t + 1:t + 7]
        union = (pers | hor).sum()
        if union and (pers & hor).sum() / union > 0.85:
            continue
        last = t
        out.append((t, "queue_ongoing"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", default=["feb", "mar"])
    ap.add_argument("--official", action="store_true", help="earliest-first selector windows (t2_pseudo_<src>_off.pkl)")
    a = ap.parse_args()
    for src in a.src:
        f = OUT / f"t2_pseudo_{src}{'_off' if a.official else ''}.pkl"
        if f.exists():
            continue
        allrows = {"onset": [], "ongoing": []}
        for p in QUEUE_PANELS:
            rows, names = pseudo(p, src, official=a.official)
            for k in rows:
                allrows[k] += rows[k]
            print(src, p, {k: len(v) for k, v in rows.items()}, flush=True)
        pickle.dump((allrows, names), open(f, "wb"))


if __name__ == "__main__":
    main()
