"""Task 1 features, built on whole (day, slot, link) grids and gathered at the requested cells.

Links are handled in travel order (x = milepost * direction, increasing downstream), so "up" always
means upstream. Flow features are in total vph: vehicles are conserved along the mainline (ramps
carry ~30 vph), so a neighbour's total flow is a direct estimate of this link's total flow.
"""
from __future__ import annotations

import numpy as np

from common import PANELS, T, days, load, network

FREE_KM_PER_STEP = 117.0 * 5 / 60  # distance a free-flowing car covers in one 5-minute step
WAVE_KM_PER_STEP = 0.8             # congestion fronts move upstream at roughly 10 km/h


class Grid:
    """Observed layer of one panel/split in travel order, plus static link attributes."""

    def __init__(self, panel: str, split: str, use_masked: bool = True, d0: int = 0, d1: int | None = None):
        net = network(panel)
        self.d0 = d0
        x = net.milepost.to_numpy() * net.attrs["direction"]
        self.order = np.argsort(x, kind="stable")          # travel position -> scorer index
        self.rank = np.empty_like(self.order)
        self.rank[self.order] = np.arange(len(self.order))  # scorer index -> travel position
        o = self.order
        self.panel, self.split = panel, split
        self.pno = PANELS.index(panel)
        self.x = (x - x.min())[o]
        self.lanes = net.lanes.to_numpy(float)[o]
        self.vf = net.free_speed_kmh.to_numpy(float)[o]
        self.cap = net.capacity_vph.to_numpy(float)[o]
        self.length = net.length_km.to_numpy(float)[o]
        sl = slice(d0, d1)
        mm = lambda name: load(panel, split, name, mmap=True)[sl][:, :, o]
        if split == "train" and use_masked:
            self.v, self.q = mm("vm"), mm("qm")
        else:
            self.v, self.q = mm("v"), mm("q")
        self.elig = mm("elig")
        self.pct = mm("pct")
        self.target = mm("target")
        self.regime = load(panel, split, "regime")[sl]
        self.D, _, self.L = self.v.shape
        self.obs = np.isfinite(self.v) & np.isfinite(self.q)
        from common import days
        import pandas as pd
        self.days = days(panel, split)[sl]
        self.dow = pd.to_datetime(pd.Series(self.days)).dt.dayofweek.to_numpy()


def _fill_index(valid: np.ndarray, forward: bool) -> np.ndarray:
    """Index along axis 1 of the last (forward) or next (backward) valid slot; -1 / T if none."""
    ar = np.arange(valid.shape[1])[None, :, None]
    if forward:
        idx = np.where(valid, ar, -1)
        return np.maximum.accumulate(idx, axis=1)
    idx = np.where(valid, ar, valid.shape[1])
    return np.minimum.accumulate(idx[:, ::-1], axis=1)[:, ::-1]


def _window_mean(vals: np.ndarray, valid: np.ndarray, x: np.ndarray, lo_km: float, hi_km: float,
                 exclude_self: bool) -> tuple[np.ndarray, np.ndarray]:
    """Mean over links whose position lies in [x_i + lo, x_i + hi], per (day, slot, link)."""
    v = np.where(valid, vals, 0.0).astype(np.float64)
    c = valid.astype(np.float64)
    cs = np.concatenate([np.zeros(v.shape[:2] + (1,)), np.cumsum(v, axis=2)], axis=2)
    cc = np.concatenate([np.zeros(c.shape[:2] + (1,)), np.cumsum(c, axis=2)], axis=2)
    a = np.searchsorted(x, x + lo_km, side="left")
    b = np.searchsorted(x, x + hi_km, side="right")
    s = cs[:, :, b] - cs[:, :, a]
    n = cc[:, :, b] - cc[:, :, a]
    if exclude_self:
        inside = (a <= np.arange(len(x))) & (np.arange(len(x)) < b)
        s = s - np.where(inside[None, None, :], v, 0)
        n = n - np.where(inside[None, None, :], c, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(n > 0, s / n, np.nan)
    return m.astype(np.float32), n.astype(np.float32)


def _shift_t(a: np.ndarray, k: int) -> np.ndarray:
    """a[:, t + k] with NaN outside the day."""
    out = np.full_like(a, np.nan)
    if k > 0:
        out[:, :-k] = a[:, k:]
    elif k < 0:
        out[:, -k:] = a[:, :k]
    else:
        out[:] = a
    return out


def _shift_l(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(a, np.nan)
    if k > 0:
        out[:, :, :-k] = a[:, :, k:]
    elif k < 0:
        out[:, :, -k:] = a[:, :, :k]
    return out


def profiles(panel: str, order: np.ndarray):
    """Train-split sums and counts per (dow, slot, link) for speed and flow, eligible cells only."""
    v = load(panel, "train", "v")[:, :, order]
    q = load(panel, "train", "q")[:, :, order]
    e = load(panel, "train", "elig")[:, :, order] & np.isfinite(v) & np.isfinite(q)
    import pandas as pd
    from common import days
    dow = pd.to_datetime(pd.Series(days(panel, "train"))).dt.dayofweek.to_numpy()
    L = v.shape[2]
    sv = np.zeros((7, T, L)); sq = np.zeros((7, T, L)); n = np.zeros((7, T, L))
    for d in range(v.shape[0]):
        m = e[d]
        sv[dow[d]] += np.where(m, v[d], 0); sq[dow[d]] += np.where(m, q[d], 0); n[dow[d]] += m
    return dict(sv=sv, sq=sq, n=n, v=v, q=q, e=e, dow=dow)


def month_sums(g: Grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Speed / flow sums and counts per (slot, link) over the grid's observed eligible cells, all days pooled."""
    e = g.obs & g.elig
    return np.where(e, g.v, 0).sum(0), np.where(e, g.q, 0).sum(0), e.sum(0).astype(float)


def month_profiles(g: Grid) -> dict:
    """Profiles from the grid's own observed eligible cells, all days pooled (every dow gets the same
    month-wide mean per slot and link). For a scored split this is the month's own regime: where bottlenecks
    moved, the train-split profiles describe the old places. Use with loo=False (targets are unobserved)."""
    sv, sq, n = month_sums(g)
    rep = lambda a: np.repeat(a[None], 7, axis=0)
    return dict(sv=rep(sv), sq=rep(sq), n=rep(n))


def train_month_profiles(panel: str) -> tuple[dict, np.ndarray]:
    """Month-own profiles for every train month from the train masked layer (targets are masked there, so no
    leave-one-out is needed), as a (month, slot, link) table, plus each train day's month index."""
    dl = days(panel, "train")
    months = sorted({d[:7] for d in dl})
    mid = np.array([months.index(d[:7]) for d in dl])
    parts = []
    for m in range(len(months)):
        idx = np.flatnonzero(mid == m)
        parts.append(month_sums(Grid(panel, "train", d0=idx[0], d1=idx[-1] + 1)))
    sv, sq, n = (np.stack(a) for a in zip(*parts))
    return dict(sv=sv, sq=sq, n=n), mid


def build(g: Grid, d: np.ndarray, t: np.ndarray, l: np.ndarray, prof: dict, loo: bool,
          key: np.ndarray | None = None) -> tuple[np.ndarray, list[str]]:
    """Feature matrix for cells (d, t, l); l is a travel-order position. key: the profile row of each grid
    day (default: its weekday)."""
    F, names = [], []

    def add(name, arr):
        names.append(name); F.append(np.asarray(arr, np.float32))

    def at(grid):
        return grid[d, t, l]

    v, q, obs = g.v, g.q, g.obs
    vn = np.where(obs, v / g.vf[None, None, :], np.nan).astype(np.float32)
    qo = np.where(obs, q, np.nan).astype(np.float32)
    n = len(d)
    add("panel", np.full(n, g.pno)); add("slot", t); add("dow", g.dow[d]); add("regime", g.regime[d])
    add("lanes", g.lanes[l]); add("vf", g.vf[l]); add("cap_lane", g.cap[l] / g.lanes[l]); add("len", g.length[l])
    add("pos", g.x[l] / g.x.max()); add("pct", g.pct[d, t, l])

    # Own link along time.
    fi, bi = _fill_index(obs, True), _fill_index(obs, False)
    p, nx = fi[d, t, l], bi[d, t, l]
    hp, hn = p >= 0, nx < T
    pv = np.where(hp, vn[d, np.maximum(p, 0), l], np.nan); nv = np.where(hn, vn[d, np.minimum(nx, T - 1), l], np.nan)
    pq = np.where(hp, qo[d, np.maximum(p, 0), l], np.nan); nq = np.where(hn, qo[d, np.minimum(nx, T - 1), l], np.nan)
    gp = np.where(hp, t - p, 99).astype(float); gn = np.where(hn, nx - t, 99).astype(float)
    w = np.where(hp & hn, gn / (gp + gn), np.where(hp, 1.0, 0.0))
    iv = np.where(hp & hn, w * pv + (1 - w) * nv, np.where(hp, pv, nv))
    iq = np.where(hp & hn, w * pq + (1 - w) * nq, np.where(hp, pq, nq))
    for nm, a in (("pv", pv), ("nv", nv), ("pq", pq), ("nq", nq), ("gp", gp), ("gn", gn), ("iv", iv), ("iq", iq)):
        add(nm, a)
    for k in (-2, -1, 1, 2):
        add(f"v_t{k:+d}", at(_shift_t(vn, k))); add(f"q_t{k:+d}", at(_shift_t(qo, k)))

    # Same slot, neighbouring links (travel order: negative = upstream).
    for k in (-3, -2, -1, 1, 2, 3):
        add(f"v_l{k:+d}", at(_shift_l(vn, k))); add(f"q_l{k:+d}", at(_shift_l(qo, k)))
    fl, bl = (_fill_index(np.swapaxes(obs, 1, 2), True), _fill_index(np.swapaxes(obs, 1, 2), False))
    up = np.swapaxes(fl, 1, 2)[d, t, np.maximum(l - 1, 0)]
    up = np.where(l > 0, up, -1)
    dn = np.swapaxes(bl, 1, 2)[d, t, np.minimum(l + 1, g.L - 1)]
    dn = np.where(l < g.L - 1, dn, g.L)
    hu, hd = up >= 0, dn < g.L
    add("up_v", np.where(hu, vn[d, t, np.maximum(up, 0)], np.nan)); add("up_q", np.where(hu, qo[d, t, np.maximum(up, 0)], np.nan))
    add("up_km", np.where(hu, g.x[l] - g.x[np.maximum(up, 0)], 99))
    add("dn_v", np.where(hd, vn[d, t, np.minimum(dn, g.L - 1)], np.nan)); add("dn_q", np.where(hd, qo[d, t, np.minimum(dn, g.L - 1)], np.nan))
    add("dn_km", np.where(hd, g.x[np.minimum(dn, g.L - 1)] - g.x[l], 99))

    # Windowed means over distance, same slot and neighbouring slots.
    for lo, hi, tag in ((-0.6, 0.6, "w06"), (-1.5, 1.5, "w15"), (-4, 4, "w40"), (-1.5, 0, "u15"), (0, 1.5, "d15")):
        mq, cnt = _window_mean(qo, obs, g.x, lo, hi, True)
        mv, _ = _window_mean(vn, obs, g.x, lo, hi, True)
        add(f"q_{tag}", at(mq)); add(f"v_{tag}", at(mv)); add(f"n_{tag}", at(cnt))
        if tag == "w15":
            for k in (-1, 1):
                add(f"q_{tag}_t{k:+d}", at(_shift_t(mq, k))); add(f"v_{tag}_t{k:+d}", at(_shift_t(mv, k)))
    mins = np.fmin.reduce([_shift_l(vn, k) for k in (-3, -2, -1, 1, 2, 3)])
    add("vmin_l3", at(mins))

    # Characteristics: free-flow vehicles (downstream drift) and congestion fronts (upstream drift).
    for k in (1, 2):
        dx = FREE_KM_PER_STEP * k
        m, _ = _window_mean(qo, obs, g.x, -dx - 1.5, -dx + 1.5, False)
        add(f"q_char_up{k}", at(_shift_t(m, -k)))
        m, _ = _window_mean(qo, obs, g.x, dx - 1.5, dx + 1.5, False)
        add(f"q_char_dn{k}", at(_shift_t(m, k)))
    m, _ = _window_mean(vn, obs, g.x, WAVE_KM_PER_STEP - 0.5, WAVE_KM_PER_STEP + 0.5, False)
    add("v_wave_dn_tm1", at(_shift_t(m, -1)))
    m, _ = _window_mean(vn, obs, g.x, -WAVE_KM_PER_STEP - 0.5, -WAVE_KM_PER_STEP + 0.5, False)
    add("v_wave_up_tp1", at(_shift_t(m, 1)))

    # Weekday x slot profiles, leave-one-day-out on train.
    kk = g.dow if key is None else key
    dw = kk[d]
    sv, sq, cn = prof["sv"][dw, t, l], prof["sq"][dw, t, l], prof["n"][dw, t, l]
    if loo:
        gd = g.d0 + d
        own = prof["e"][gd, t, l]
        sv = sv - np.where(own, prof["v"][gd, t, l], 0); sq = sq - np.where(own, prof["q"][gd, t, l], 0); cn = cn - own
    with np.errstate(invalid="ignore", divide="ignore"):
        pvf = np.where(cn > 0, sv / cn, np.nan) / g.vf[l]
        pqf = np.where(cn > 0, sq / cn, np.nan)
    add("prof_v", pvf); add("prof_q", pqf)
    # Corridor-wide demand level of this day around this slot, relative to the profile.
    with np.errstate(invalid="ignore", divide="ignore"):
        pq_grid = np.where(prof["n"] > 0, prof["sq"] / prof["n"], np.nan)[kk]  # (D, T, L)
        ratio = np.where(obs, qo / pq_grid, np.nan)
        rs = np.nanmean(ratio, axis=2)  # (D, T)
    k = np.ones(7) / 7
    rsm = np.stack([np.convolve(np.nan_to_num(r, nan=1.0), k, mode="same") for r in rs])
    add("day_ratio", rsm[d, t]); add("prof_q_scaled", pqf * rsm[d, t])
    if SHOCK:
        if getattr(g, "_shock", None) is None:
            g._shock = shock_grid(g)
        lev, c, ps = g._shock
        add("ff_level", lev[d, l]); add("shock", c[d, t]); add("ps_n", ps[d, t, l])
    return np.column_stack(F), names


# Free-flow speed is generated as level_l * (1 + c_t + e): c_t is one shock per (day, slot) shared by every link
# of the corridor (std 0.4-1.0%, white in time) and the link levels are redrawn for each split. Both are read from
# the grid's own visible cells (offline imputation), so the model can follow them in a new month.
SHOCK = __import__("os").environ.get("TFB_T1_SHOCK") == "1"
VCLIP = 120.75


def shock_grid(g: Grid):
    """(level / vf per (day, link), shock per (day, slot), censored free-flow prediction / vf per cell).
    Levels are per calendar month of the grid."""
    from scipy.stats import norm
    V = np.where(g.obs & g.elig.astype(bool), np.asarray(g.v, float), np.nan)
    fr = V > 0.9 * g.vf[None, None, :]
    months = np.array([x[:7] for x in g.days])
    mu = np.empty((g.D, g.L))
    for m in np.unique(months):
        k = months == m
        lv = np.nanmedian(np.where(fr[k], V[k], np.nan).reshape(-1, g.L), axis=0)
        mu[k] = np.where(np.isfinite(lv), lv, g.vf)[None, :]
    use = fr & (mu[:, None, :] < 118.0) & (V < VCLIP - 0.01)
    rel = np.where(use, V / mu[:, None, :] - 1, np.nan)
    c = np.nanmedian(rel, axis=2)
    c = np.where(np.isfinite(c), c, 0.0)
    sig = np.nanstd(rel - c[:, :, None])
    sig = sig if np.isfinite(sig) and sig > 0 else 0.01
    m_ = mu[:, None, :] * (1 + c[:, :, None]); s_ = sig * mu[:, None, :]
    a = (VCLIP - m_) / s_
    ps = m_ * norm.cdf(a) - s_ * norm.pdf(a) + VCLIP * (1 - norm.cdf(a))
    return (mu / g.vf[None, :]).astype(np.float32), c.astype(np.float32), (ps / g.vf[None, None, :]).astype(np.float32)
