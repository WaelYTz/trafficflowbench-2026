"""Anchor-noise offsets for Task 3 (improved anchor correction), submission-key order -> kaggle_pkg/src_ds/t3_noise.npz.

For every Task 1 target cell inside a hidden run with released anchors a (before) and b (after) on the same link and day:
  r_a = x(a) - s_loo(a),  r_b = x(b) - s_loo(b)   per variable (flow q and speed v separately),
where s_loo is a leave-one-out tricube-weighted local linear fit over the released cells within +-W slots of the anchor
(the anchor itself and hidden cells excluded), i.e. an estimate of the anchor's measurement noise. The offset added to the
model's own prediction is  rho * ((1 - lam) r_a + lam r_b),  lam = (t - a) / (b - a). The model's shape is kept; only the
interpolated anchor noise is added (applied in subkey.write_submission, state_noise). Uses only released cells of the same
link and day (Task 1 is offline imputation).

  python t3_noise.py
"""
from __future__ import annotations

import numpy as np

import pandas as pd

import subkey
from common import CACHE, PANELS, REL, ROOT, link_index, load

W = 3


def loo_resid(z: np.ndarray, rel: np.ndarray) -> np.ndarray:
    """z, rel: (D, T, L). Residual z - leave-one-out local linear fit at released cells (NaN elsewhere / too few points)."""
    D, T, L = z.shape
    zz = np.where(rel, z, 0.0)
    S0 = np.zeros(z.shape); S1 = np.zeros(z.shape); S2 = np.zeros(z.shape); T0 = np.zeros(z.shape); T1 = np.zeros(z.shape)
    n = np.zeros(z.shape)
    for o in range(-W, W + 1):
        if o == 0:
            continue
        w = (1 - (abs(o) / (W + 1)) ** 3) ** 3
        sh_z = np.zeros(z.shape); sh_r = np.zeros(z.shape, bool)
        if o > 0:
            sh_z[:, :-o] = zz[:, o:]; sh_r[:, :-o] = rel[:, o:]
        else:
            sh_z[:, -o:] = zz[:, :o]; sh_r[:, -o:] = rel[:, :o]
        wr = w * sh_r
        S0 += wr; S1 += wr * o; S2 += wr * o * o; T0 += wr * sh_z; T1 += wr * o * sh_z; n += sh_r
    det = S0 * S2 - S1 * S1
    with np.errstate(invalid="ignore", divide="ignore"):
        lin = (S2 * T0 - S1 * T1) / det
        mean = T0 / S0
    s = np.where((n >= 3) & (det > 1e-9), lin, np.where(n >= 1, mean, np.nan))
    return np.where(rel, z - s, np.nan)


def spatial_resid(q: np.ndarray, rel: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Per-detector flow noise estimate at released cells: q - weighted mean of released q on the travel-order neighbours
    l+-1 (weight 1) and l+-2 (weight 0.5) at the same time step (adjacent links carry nearly the same total flow)."""
    qo, ro = q[:, :, order], rel[:, :, order]
    qz = np.where(ro, qo, 0.0); num = np.zeros(qo.shape); den = np.zeros(qo.shape)
    for k_ in (-2, -1, 1, 2):
        sz = np.zeros(qo.shape); sr = np.zeros(qo.shape, bool)
        if k_ > 0:
            sz[:, :, :-k_] = qz[:, :, k_:]; sr[:, :, :-k_] = ro[:, :, k_:]
        else:
            sz[:, :, -k_:] = qz[:, :, :k_]; sr[:, :, -k_:] = ro[:, :, :k_]
        wk = 1.0 if abs(k_) == 1 else 0.5
        num += wk * sz * sr; den += wk * sr
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(ro & (den > 0), qo - num / den, np.nan)
    out = np.full(q.shape, np.nan); out[:, :, order] = r
    return out


def main(dst=ROOT / "kaggle_pkg" / "src_ds" / "t3_noise.npz"):
    if not (CACHE / "subkey.npz").exists():
        subkey.build()
    k = subkey.key()
    n = len(k["task"])
    oq = np.zeros(n, np.float32); ov = np.zeros(n, np.float32); run = np.full(n, 10**6, np.int32)
    for pi, p in enumerate(PANELS):
        for si, s in ((1, "validation"), (2, "private")):
            v = load(p, s, "v").astype(np.float64); q = load(p, s, "q").astype(np.float64); e = load(p, s, "elig").astype(bool)
            rel = e & np.isfinite(v) & np.isfinite(q) & (v > 0)
            from t1_features import Grid
            order = Grid(p, s, d0=0, d1=1).order
            li = link_index(p); fd = pd.read_csv(REL / "corridors" / p / "network" / "fd_parameters.csv").drop_duplicates("link_id")
            vf = np.full(len(li), np.nan); vf[[li[x] for x in fd.link_id]] = fd.free_speed_kmh.to_numpy(float)
            rq = spatial_resid(q, rel, order)                  # flow only; speed noise is small and its residual is dynamics
            rv = np.zeros(q.shape)
            D, T, L = v.shape
            tt = np.arange(T)[None, :, None]
            pa = np.maximum.accumulate(np.where(rel, tt, -1), axis=1)
            nb = np.minimum.accumulate(np.where(rel, tt, T)[:, ::-1], axis=1)[:, ::-1]
            m = np.flatnonzero((k["task"] == 0) & (k["panel"] == pi) & (k["split"] == si))
            d, t, l = k["day"][m], k["slot"][m], k["link"][m]
            a, b = pa[d, t, l], nb[d, t, l]
            ok = (a >= 0) & (b < T)
            a0, b0 = np.maximum(a, 0), np.minimum(b, T - 1)
            lam = (t - a0) / np.maximum(b0 - a0, 1)
            ff = (v[d, a0, l] > 0.85 * vf[l]) & (v[d, b0, l] > 0.85 * vf[l])      # both anchors in free flow
            ra, rb = rq[d, a0, l], rq[d, b0, l]
            off = (1 - lam) * ra + lam * rb
            qa = 0.5 * (q[d, a0, l] + q[d, b0, l])
            oq[m] = np.where(ok & ff & np.isfinite(off) & np.isfinite(qa), np.clip(off, -0.08 * qa, 0.08 * qa), 0.0)
            ov[m] = 0.0
            run[m] = np.where(ok, b0 - a0 - 1, 10**6)
            print(f"{p} {s}: {len(m):,} targets, offset rms q {np.sqrt(np.mean(oq[m] ** 2)):.1f} vph, v {np.sqrt(np.mean(ov[m] ** 2)):.2f} km/h",
                  flush=True)
    np.savez_compressed(dst, oq=oq, ov=ov, run=run)
    print("wrote", dst, flush=True)


def apply(sp, fl, st, nz, rho, max_run):
    g = st & (nz["run"] <= max_run)
    fl[g] = np.clip(fl[g] + rho * nz["oq"][g], 0, 150000)
    sp[g] = np.clip(sp[g] + rho * nz["ov"][g], 1, 180)
    return int(g.sum())


if __name__ == "__main__":
    main()
