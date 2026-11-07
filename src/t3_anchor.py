"""Anchor data for the Task 3 'anchor-offset' density correction, in submission-key order -> kaggle_pkg/src_ds/t3_anchor.npz.

For every Task 1 target cell: the released (eligible, finite) cells of the same link and day just before (a) and after (b)
the hidden run that contains it, their densities ka = q/v and kb, the position f = (t - a) / (b - a), the key rows of the
run's first and last target cells (i0, i1) and the run length. The correction itself (subkey.write_submission,
state_anchor) moves the submitted density to  k + (ka - k[i0]) + ((kb - k[i1]) - (ka - k[i0])) * f,  i.e. our own shape
inside the run plus the time-linear interpolation of the offsets at the two released neighbours. Uses only the released
masked layer of the same split (Task 1 is offline imputation).

  python t3_anchor.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import subkey
from common import CACHE, PANELS, REL, ROOT, link_index, load


def main(dst=ROOT / "kaggle_pkg" / "src_ds" / "t3_anchor.npz"):
    if not (CACHE / "subkey.npz").exists():
        subkey.build()
    k = subkey.key()
    n = len(k["task"])
    ka = np.full(n, np.nan, np.float32); kb = np.full(n, np.nan, np.float32); f = np.zeros(n, np.float32)
    i0 = np.arange(n, dtype=np.int64); i1 = np.arange(n, dtype=np.int64); run = np.full(n, 10**6, np.int32)
    ff = np.zeros(n, bool)
    for pi, p in enumerate(PANELS):
        for si, s in ((1, "validation"), (2, "private")):
            v = load(p, s, "v").astype(np.float64); q = load(p, s, "q").astype(np.float64); e = load(p, s, "elig").astype(bool)
            anc = e & np.isfinite(v) & np.isfinite(q) & (v > 0)
            K = np.where(anc, q / np.maximum(v, 1.0), np.nan)
            D, T, L = v.shape
            tt = np.arange(T)[None, :, None]
            pa = np.maximum.accumulate(np.where(anc, tt, -1), axis=1)
            nb = np.minimum.accumulate(np.where(anc, tt, T)[:, ::-1], axis=1)[:, ::-1]
            m = np.flatnonzero((k["task"] == 0) & (k["panel"] == pi) & (k["split"] == si))
            d, t, l = k["day"][m], k["slot"][m], k["link"][m]
            idx = np.full((D, T, L), -1, np.int64); idx[d, t, l] = m
            a, b = pa[d, t, l], nb[d, t, l]
            ok = (a >= 0) & (b < T)
            a0, b0 = np.maximum(a, 0), np.minimum(b, T - 1)
            ka[m] = np.where(ok, K[d, a0, l], np.nan); kb[m] = np.where(ok, K[d, b0, l], np.nan)
            f[m] = np.where(ok, (t - a0) / np.maximum(b0 - a0, 1), 0.0)
            s0 = idx[d, np.minimum(a0 + 1, T - 1), l]; s1 = idx[d, np.maximum(b0 - 1, 0), l]
            i0[m] = np.where(ok & (s0 >= 0), s0, m); i1[m] = np.where(ok & (s1 >= 0), s1, m)
            run[m] = np.where(ok, b0 - a0 - 1, 10**6)
            li = link_index(p); fd = pd.read_csv(REL / "corridors" / p / "network" / "fd_parameters.csv").drop_duplicates("link_id")
            vf = np.full(len(li), np.nan); vf[[li[x] for x in fd.link_id]] = fd.free_speed_kmh.to_numpy(float)
            kA, kB = K[d, a0, l], K[d, b0, l]
            with np.errstate(invalid="ignore"):   # both anchors free-flow and nearly equal density: the regime gate
                ff[m] = ok & (v[d, a0, l] > 0.85 * vf[l]) & (v[d, b0, l] > 0.85 * vf[l]) & (np.abs(kB - kA) < 0.06 * 0.5 * (kA + kB))
            print(f"  free-flow gate {ff[m].mean():.3f}", flush=True)
            print(f"{p} {s}: {len(m):,} targets, anchored {ok.mean():.3f}, run 1 {np.mean(run[m] == 1):.3f}, "
                  f"run<=4 {np.mean(run[m] <= 4):.3f}, run>18 {np.mean((run[m] > 18) & ok):.3f}", flush=True)
    np.savez_compressed(dst, ka=ka, kb=kb, f=f, i0=i0, i1=i1, run=run, ff=ff)
    print("wrote", dst, flush=True)


def apply(sp, fl, st, anchor, w, alpha, max_run, w_off=None):
    """In place on the state rows st: moves density q/v by w * (log k_new - log k), the change split alpha (flow) /
    1 - alpha (speed). With w_off, w applies only to gated (free-flow, flat) runs and w_off to the rest."""
    kh = fl / np.maximum(sp, 1.0)
    ka, kb, f, i0, i1, run = (anchor[x] for x in ("ka", "kb", "f", "i0", "i1", "run"))
    oa = ka - kh[i0]; ob = kb - kh[i1]
    kn = kh + oa + (ob - oa) * f
    g = st & np.isfinite(kn) & (kn > 0) & (kh > 0) & (run <= max_run)
    dl = np.zeros_like(kh)
    wv = np.full(kh.shape, float(w)) if w_off is None else np.where(anchor["ff"], float(w), float(w_off))
    dl[g] = wv[g] * (np.log(kn[g]) - np.log(kh[g]))
    fl[g] = np.clip(fl[g] * np.exp(alpha * dl[g]), 0, 150000)
    sp[g] = np.clip(sp[g] * np.exp(-(1 - alpha) * dl[g]), 1, 180)
    return int(g.sum())


if __name__ == "__main__":
    main()
