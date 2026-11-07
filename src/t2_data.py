"""Task 2 windows and features.

Training windows are cut from the train split with the published selection rule (60-minute history,
70% coverage, a queue in the horizon, 360-minute spacing per condition, earliest origin first;
ongoing also needs persistence IoU <= 0.9). History comes from the full train layer, exactly what
window_history.parquet holds; the row at T comes from the masked layer, as it does on scored splits.

Every feature uses only slots T-12 .. T. Arrays are in travel order (upstream -> downstream).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from common import QUEUE_PANELS, REL, T, days, load, network

H = 12          # history slots before T
TODAY2 = os.environ.get("TFB_TODAY2") == "1"   # richer same-day context features (experiment)
SPACING = 72    # 360 minutes
K = 5           # neighbouring links on each side in the patch


class Static:
    def __init__(self, panel: str):
        net = network(panel)
        x = net.milepost.to_numpy() * net.attrs["direction"]
        self.order = np.argsort(x, kind="stable")
        o = self.order
        self.panel, self.pno = panel, QUEUE_PANELS.index(panel)
        self.L = len(o)
        self.x = (x - x.min())[o]
        self.vf = net.free_speed_kmh.to_numpy(float)[o]
        self.cut = 0.6 * self.vf
        self.cap = net.capacity_vph.to_numpy(float)[o]
        self.lanes = net.lanes.to_numpy(float)[o]
        self.kc = net.critical_density.to_numpy(float)[o]
        self.length = net.length_km.to_numpy(float)[o]
        nxt = np.r_[np.arange(1, self.L), self.L - 1]
        self.lane_drop = self.lanes[nxt] - self.lanes
        self.cap_ratio = self.cap[nxt] / self.cap


def train_arrays(st: Static):
    o = st.order
    v = load(st.panel, "train", "v")[:, :, o]
    q = load(st.panel, "train", "q")[:, :, o]
    e = load(st.panel, "train", "elig")[:, :, o]
    vm = load(st.panel, "train", "vm")[:, :, o]
    qm = load(st.panel, "train", "qm")[:, :, o]
    return v, q, e, vm, qm


def select_windows(v: np.ndarray, e: np.ndarray, cut: np.ndarray, pers_max: float = 0.75) -> list[tuple[int, int, str]]:
    """(day, origin slot, condition) following the published selector, scanning forward in time.

    Spacing is shared by both conditions (that reproduces the official train windows best), and the
    noise-free persistence bound of 0.9 is approximated by pers_max on noisy observed labels.
    """
    qobs = np.isfinite(v) & (v <= cut)          # visible queue in history
    qtrue = qobs & e                            # scored queue cells (eligible)
    out = []
    last = -10**9
    for d in range(v.shape[0]):
        for t in range(H, T - 6):
            g = d * T + t
            hist = slice(t - H, t)
            hor = qtrue[d, t + 1:t + 7]
            if not hor.any():
                continue
            if np.isfinite(v[d, hist]).mean() < 0.7:
                continue
            hq = qobs[d, hist]
            if not hq.any():
                cond = "queue_onset"
            elif hq.sum() >= 2 and hq.sum(axis=0).max() >= 2:
                cond = "queue_ongoing"
                pers = np.broadcast_to(qobs[d, t - 1], hor.shape) & e[d, t + 1:t + 7]
                union = (pers | hor).sum()
                if union and (pers & hor).sum() / union > pers_max:
                    continue
            else:
                continue
            if g - last < SPACING:
                continue
            last = g
            out.append((d, t, cond))
    return out


def training_origins(v: np.ndarray, e: np.ndarray, cut: np.ndarray, ongoing_step: int = 6,
                     pers_max: float = 0.85) -> list[tuple[int, int, str]]:
    """Denser training windows of the same kinds the selector produces.

    onset:   no visible queue in the history, the first scored queue exactly at T+30 (one origin
             per onset event, which is what the earliest-origin selector yields).
    ongoing: visible queue (>= 2 observations, >= 2 on one link), a queue in the horizon, and a
             queue that changes (noisy persistence IoU <= pers_max), one origin every ongoing_step slots.
    """
    qobs = np.isfinite(v) & (v <= cut)
    qtrue = qobs & e
    out = []
    for d in range(v.shape[0]):
        last = -10**9
        cov = np.convolve(np.isfinite(v[d]).mean(axis=1), np.ones(H) / H, mode="full")[H - 1:T]  # mean over t..t+H-1
        for t in range(H, T - 6):
            if cov[t - H] < 0.7:
                continue
            hor = qtrue[d, t + 1:t + 7]
            if not hor.any():
                continue
            hq = qobs[d, t - H:t]
            if not hq.any():
                if hor[5].any() and not hor[:5].any():
                    out.append((d, t, "queue_onset"))
                continue
            young = YOUNG_STEP and not hq[:H // 2].any()      # queue first visible in the last 30 minutes
            if hq.sum() < 2 or hq.sum(axis=0).max() < 2 or t - last < (YOUNG_STEP if young else ongoing_step):
                continue
            pers = np.broadcast_to(qobs[d, t - 1], hor.shape) & e[d, t + 1:t + 7]
            union = (pers | hor).sum()
            if union and (pers & hor).sum() / union > pers_max:
                continue
            last = t
            out.append((d, t, "queue_ongoing"))
    return out


def priors(st: Static, v: np.ndarray, e: np.ndarray, months: np.ndarray):
    """Per-month counts: onset starts and queued cells per (link, hour) on train."""
    nm = months.max() + 1
    onset = np.zeros((nm, st.L, 24)); queued = np.zeros((nm, st.L, 24)); seen = np.zeros((nm, st.L, 24))
    qobs = np.isfinite(v) & (v <= st.cut) & e
    active = qobs.any(axis=2)
    for d in range(v.shape[0]):
        m = months[d]
        hours = np.arange(T) // 12
        np.add.at(queued[m], (slice(None), hours), qobs[d].T)
        np.add.at(seen[m], (slice(None), hours), e[d].T)
        for t in range(12, T):
            if active[d, t] and not active[d, t - 12:t].any():
                onset[m, :, t // 12] += qobs[d, t]
    return onset, queued, seen


CAL_SPLITS = ("train", "validation", "private")
RECENT_DAYS = 14
# The scored splits' masked layers hide T+1..T+18 of every official window on all links, i.e. exactly the
# queues that later windows' memory should count; the train layer has no such holes. With TFB_FILL_BLANKS=1
# Recent bridges each whole-corridor gap per link from the nearest values before and after it (both stamped
# before any later window's origin, since origins are >= 360 minutes apart).
FILL_BLANKS = os.environ.get("TFB_FILL_BLANKS") == "1"
# A quarter of the official ongoing windows hold a queue first visible in the last 30 minutes of the history,
# against ~7% of the dense training origins. TFB_YOUNG_STEP=k takes every k-th origin for such young queues
# (instead of one per ongoing_step) so training sees them as often as scoring does. 0 = off.
YOUNG_STEP = int(os.environ.get("TFB_YOUNG_STEP", 0))
# Ramp flows (ramp_states, released for every split) as per-link inputs, slots T-12..T only. On the March bench
# they lift a memory-free ongoing LightGBM from 0.881 to 0.889 (scripts/ramp_ongoing_bench.py).
RAMPS = os.environ.get("TFB_RAMPS") == "1"
RAMP_NAMES = ("ramp_or_now", "ramp_or_hist", "ramp_or_trend", "ramp_fr_now", "ramp_fr_trend",
              "ramp_or_up2", "ramp_or_dn2", "ramp_or_up2_tr", "ramp_fr_dn2")
# Bottleneck tracking (experiment, TFB_BNECK=1). Onset heads are a small fixed set per panel and per-link capacity
# is fixed (queue discharge = fd capacity_vph in every split); what moves between splits is DEMAND, and a head
# stops firing once its recent peak free-flow demand falls below capacity (D12_I5_S head 56: q95/cap 1.01 -> 0.96).
# Per link, demand = flow in free-flow cells (speed > 0.9 vf) of the masked layer over the fixed fd capacity:
# q95 / q99 over the previous BNECK_DAYS calendar days (train -> March -> April) and q95 earlier the same day
# (slots < T-12), each also as the max over the link and the links within BN_KM downstream (where a queue would
# form); flow / capacity at T-10..T, its downstream max and its ratio to the recent q95; recent free-flow speed
# level; on-ramp inflow at T-10..T over capacity. Prefix "bn_": no model filter (MEMORY / LONG / PLACE) drops them.
BNECK = os.environ.get("TFB_BNECK") == "1"
BNECK_DAYS = 7
BN_KM = 2.0


def fill_slabs(a: np.ndarray, min_len: int = 6, reach: int = 6) -> np.ndarray:
    """a: (days, slots, links). Linear bridge, per link, across runs of >= min_len slots where every link is
    missing, from the last value within `reach` slots before the run to the first within `reach` after it
    (one-sided anchors are carried; no anchor leaves NaN)."""
    a = np.array(a, dtype=float, copy=True)
    allnan = ~np.isfinite(a).any(axis=2)
    for d in range(a.shape[0]):
        t = 0
        while t < a.shape[1]:
            if not allnan[d, t]:
                t += 1
                continue
            s = t
            while t < a.shape[1] and allnan[d, t]:
                t += 1
            if t - s < min_len:
                continue
            bef, aft = a[d, max(s - reach, 0):s], a[d, t:t + reach]
            if not len(bef):
                bef = np.full((1, a.shape[2]), np.nan)
            if not len(aft):                                 # gap runs to the end of the day
                aft = np.full((1, a.shape[2]), np.nan)
            has_b, has_a = np.isfinite(bef).any(axis=0), np.isfinite(aft).any(axis=0)
            ib = np.where(has_b, bef.shape[0] - 1 - np.argmax(np.isfinite(bef[::-1]), axis=0), 0)
            ia = np.where(has_a, np.argmax(np.isfinite(aft), axis=0), 0)
            vb = np.where(has_b, bef[ib, np.arange(a.shape[2])], np.nan)
            va = np.where(has_a, aft[ia, np.arange(a.shape[2])], np.nan)
            vb, va = np.where(has_b, vb, va), np.where(has_a, va, vb)
            frac = (np.arange(1, t - s + 1) / (t - s + 1))[:, None]
            a[d, s:t] = vb[None] + (va - vb)[None] * frac
    return a


def ramp_attach(st: Static):
    """Ramp -> mainline link incidence (on-ramps, off-ramps; (ramps, L)) and the links within 2 km up/downstream
    of each link (L, L): up2[i, j] = 1 if j lies 0-2 km upstream of i."""
    net = network(st.panel)
    rank = np.empty(st.L, int); rank[st.order] = np.arange(st.L)
    li = {l: i for i, l in enumerate(net.link_id)}
    m = pd.read_csv(REL / "corridors" / st.panel / "network" / "ramp_attachment_map.csv", dtype=str)
    pos = np.array([rank[li[x]] if x in li else -1 for x in m.nearest_mainline_link_id])
    a_or = np.zeros((len(m), st.L)); a_fr = np.zeros((len(m), st.L))
    for k, (kind, ps) in enumerate(zip(m.ramp_type, pos)):
        if ps >= 0:
            (a_or if kind == "OR" else a_fr)[k, ps] = 1
    dx = st.x[None, :] - st.x[:, None]
    up2 = ((dx < 0) & (dx >= -2)).astype(float); dn2 = ((dx > 0) & (dx <= 2)).astype(float)
    return a_or, a_fr, up2, dn2


def _near(st: Static):
    """Boolean (L, L): links within BN_KM upstream / downstream of each link (excluding itself), and the
    downstream reach including the link itself."""
    if getattr(st, "_bn_near", None) is None:
        dx = st.x[None, :] - st.x[:, None]
        dn = (dx > 0) & (dx <= BN_KM)
        st._bn_near = ((dx < 0) & (dx >= -BN_KM), dn, dn | np.eye(st.L, dtype=bool))
    return st._bn_near


def _nanred(f, a, axis):
    import warnings
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore")
        return f(a, axis=axis)


def _nanq(x: np.ndarray, qs) -> np.ndarray:
    """np.nanpercentile(x, qs, axis=0) (linear method) for 2-D x, vectorised: (len(qs), x.shape[1])."""
    s = np.sort(np.asarray(x, np.float64), axis=0)                               # NaN sort last
    n = np.isfinite(s).sum(0)
    cols = np.arange(s.shape[1])
    out = []
    for q in qs:
        pos = np.maximum(n - 1, 0) * q / 100.0
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, np.maximum(n - 1, 0))
        frac = pos - lo
        out.append(np.where(n > 0, s[lo, cols] + (s[hi, cols] - s[lo, cols]) * frac, np.nan))
    return np.array(out)


def _dnmax(st: Static, a: np.ndarray) -> np.ndarray:
    """max of a over the link and the links within BN_KM downstream (NaN-skipping)."""
    return _nanred(np.nanmax, np.where(_near(st)[2], a[None, :], np.nan), 1)


BN_DAY = ("bn_d95", "bn_d99", "bn_d95_dn", "bn_d99_dn", "bn_dtd", "bn_dtd_dn", "bn_vff_rel", "bn_or_cap")
BN_WIN = ("bn_qc_max3", "bn_qc_dn", "bn_qc_dn3", "bn_qc_rel", "bn_qc_dn_rel", "bn_vr_last")


class Bneck:
    """Recent per-link free-flow demand vs the fixed fd capacity on the calendar train -> March -> April (TFB_BNECK=1).

    Row c of d95 / d99 / vff summarises calendar days c-BNECK_DAYS .. c-1 of the masked layers (train: vm/qm;
    scored splits: v/q, raw: holes from Task 1 targets and the official-window blanks are skipped by the
    quantiles). A window on calendar day c at slot T reads row c, its own day's masked layer before T-12, and ramp
    flows at T-10..T, so nothing stamped after T enters."""

    def __init__(self, st: Static):
        self.st = st
        o = st.order
        vs, self.ff, self.offset = [], {}, {}
        c = 0
        for split in CAL_SPLITS:
            q = np.asarray(load(st.panel, split, "qm" if split == "train" else "q", mmap=True), np.float32)[:, :, o]
            v = np.asarray(load(st.panel, split, "vm" if split == "train" else "v", mmap=True), np.float32)[:, :, o]
            free = v > 0.9 * st.vf[None, None, :]
            self.ff[split] = np.where(free, q, np.nan).astype(np.float32)          # free-flow flow (demand)
            vs.append(np.where(free, v, np.nan).astype(np.float32))
            self.offset[split] = c
            c += q.shape[0]
            del q, v, free
        F, V = np.concatenate([self.ff[s] for s in CAL_SPLITS]), np.concatenate(vs)
        del vs
        self.d95, self.d99, self.vff = (np.full((c + 1, st.L), np.nan) for _ in range(3))
        for k in range(1, c + 1):
            a = max(k - BNECK_DAYS, 0)
            self.d95[k], self.d99[k] = _nanq(F[a:k].reshape(-1, st.L), (95, 99)) / st.cap[None, :]
            self.vff[k] = _nanq(V[a:k].reshape(-1, st.L), (50,))[0]
        del F, V
        self.a_or, _, self.up2, _ = ramp_attach(st)
        self.rq = {split: load(st.panel, split, "rq", mmap=True) for split in CAL_SPLITS}

    def features(self, split: str, day: int, slot: int) -> dict:
        st = self.st
        c = self.offset[split] + day
        d95, d99 = self.d95[c], self.d99[c]
        today = self.ff[split][day, :max(slot - H, 0)]                             # same day, slots < T-12
        dtd = _nanq(today, (95,))[0] / st.cap if len(today) else np.full(st.L, np.nan)
        r = np.asarray(self.rq[split][day, max(slot - 2, 0):slot + 1], float)       # ramp flows T-10 min .. T
        o1 = np.nan_to_num(_nanred(np.nanmean, r, 0)) @ self.a_or                  # on-ramp inflow per link, vph
        vals = (d95, d99, _dnmax(st, d95), _dnmax(st, d99), dtd, _dnmax(st, dtd), self.vff[c] / st.vf,
                (o1 + self.up2 @ o1) / st.cap)
        return dict(zip(BN_DAY, vals))


def bneck_window(st: Static, extra: dict, last_v, last_q, q3) -> dict:
    """Window-level demand from window_features' own inputs (normalised by the fixed vf / cap): max flow / cap over
    T-10..T, the downstream max (link + 2 km) of the last observed and of that max flow / cap, current over recent
    q95 demand (here and downstream), and the last speed over the recent free-flow level."""
    qc3 = _nanred(np.nanmax, q3, 0)
    dn3 = _dnmax(st, qc3)
    with np.errstate(invalid="ignore", divide="ignore"):
        vals = (qc3, _dnmax(st, last_q), dn3, qc3 / extra["bn_d95"], dn3 / extra["bn_d95_dn"], last_v / extra["bn_vff_rel"])
    return dict(zip(BN_WIN, vals))


class Recent:
    """Causal same-season queue statistics from the masked layers, on one calendar train -> March -> April.

    For a window on calendar day c at slot T it reports, per link, what the masked observations showed
    on days c-14 .. c-1 (queued share and onset starts at that hour) and earlier the same day (slots
    before T-12). Only released observations stamped before T are read.
    """

    def __init__(self, st: Static):
        o = st.order
        qs, os_, ons, self.vm = [], [], [], {}
        self.offset = {}
        c = 0
        for split in CAL_SPLITS:
            v = load(st.panel, split, "vm" if split == "train" else "v")[:, :, o]
            if FILL_BLANKS:
                v = fill_slabs(v)
            self.vm[split] = v
            self.offset[split] = c
            c += v.shape[0]
            obs = np.isfinite(v)
            qd = obs & (v <= st.cut[None, None, :])
            hours = np.arange(T) // 12
            qh = np.zeros((v.shape[0], st.L, 24)); oh = np.zeros_like(qh); on = np.zeros_like(qh)
            for h in range(24):
                qh[:, :, h] = qd[:, hours == h].sum(axis=1)
                oh[:, :, h] = obs[:, hours == h].sum(axis=1)
            active = qd.any(axis=2)
            for d in range(v.shape[0]):
                for t in range(12, T):
                    if active[d, t] and not active[d, t - 12:t].any():
                        on[d, :, t // 12] += qd[d, t]
            qs.append(qh); os_.append(oh); ons.append(on)
        z = lambda a: np.concatenate([np.zeros((1,) + a[0].shape[1:]), np.cumsum(np.concatenate(a), axis=0)])
        self.cq, self.co, self.con = z(qs), z(os_), z(ons)
        self.cut = st.cut
        if RAMPS:
            self.a_or, self.a_fr, self.up2, self.dn2 = ramp_attach(st)
            self.rq = {split: load(st.panel, split, "rq", mmap=True) for split in CAL_SPLITS}
        if BNECK:
            self.bn = Bneck(st)

    def ramps(self, split: str, day: int, slot: int) -> dict:
        r = np.asarray(self.rq[split][day, max(slot - H, 0):slot + 1], float)     # stamped <= T
        with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore")
            now, hist = np.nanmean(r[-3:], 0), np.nanmean(r[:-3], 0)
        f = lambda a, A: np.nan_to_num(a) @ A
        o1, o0, f1, f0 = f(now, self.a_or), f(hist, self.a_or), f(now, self.a_fr), f(hist, self.a_fr)
        vals = (o1, o0, o1 - o0, f1, f1 - f0, self.up2 @ o1, self.dn2 @ o1, self.up2 @ (o1 - o0), self.dn2 @ f1)
        return dict(zip(RAMP_NAMES, vals))

    def features(self, split: str, day: int, slot: int) -> dict:
        c = self.offset[split] + day
        a = max(c - RECENT_DAYS, 0)
        h = (slot + 6) // 12 % 24
        hrs = [(h - 1) % 24, h, (h + 1) % 24]
        wts = np.array([0.5, 1.0, 0.5])
        q = (self.cq[c] - self.cq[a])[:, hrs] @ wts
        n = (self.co[c] - self.co[a])[:, hrs] @ wts
        on = (self.con[c] - self.con[a])[:, hrs] @ wts
        v = self.vm[split][day, :max(slot - H, 0)]
        qd = np.isfinite(v) & (v <= self.cut[None, :])
        today = qd.sum(axis=0)
        out = {"recent_q": q / np.maximum(n, 1), "recent_n": n, "recent_onset": on, "today_q_before": today}
        if RAMPS:
            out.update(self.ramps(split, day, slot))
        if BNECK:
            out.update(self.bn.features(split, day, slot))
        if TODAY2:
            # Same-day context just before the 60-minute history (slots T-24 .. T-13), so a queue that formed
            # shortly before the window can be told from one that ended hours earlier. Prefixed
            # today_q_before so the memory-free models keep excluding them.
            ph = qd[max(slot - 2 * H, 0):]
            last = np.where(qd, np.arange(qd.shape[0])[:, None], -1).max(axis=0) if qd.shape[0] else np.full(len(today), -1)
            out["today_q_before_ph"] = ph.sum(axis=0)
            out["today_q_before_age"] = np.where(last >= 0, qd.shape[0] - last, T)
            out["today_q_before_corr"] = np.full(len(today), float(ph.sum()))
        return out


def window_features(st: Static, V: np.ndarray, Q: np.ndarray, slot: int, dow: int,
                    onset_prior: np.ndarray, queued_prior: np.ndarray, extra: dict | None = None) -> tuple[np.ndarray, list[str]]:
    """Per-link features for one window. V, Q: (H+1, L) speed/flow for slots T-12..T (row H is T)."""
    L = st.L
    vn = V / st.vf[None, :]
    qn = Q / st.cap[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        kn = (Q / np.maximum(V, 1)) / st.kc[None, :]
    fin = np.isfinite(vn)
    idx = np.where(fin, np.arange(H + 1)[:, None], -1).max(axis=0)
    has = idx >= 0
    last_v = np.where(has, vn[np.maximum(idx, 0), np.arange(L)], np.nan)
    last_q = np.where(has, qn[np.maximum(idx, 0), np.arange(L)], np.nan)
    last_k = np.where(has, kn[np.maximum(idx, 0), np.arange(L)], np.nan)
    age = np.where(has, H - idx, 99)
    qlast = np.where(has, last_v <= 0.6, False)
    F, names = [], []

    def add(n, a):
        names.append(n); F.append(np.broadcast_to(np.asarray(a, np.float32), (L,)))

    add("panel", st.pno); add("slot", slot); add("dow", dow)
    add("pos", st.x / st.x.max()); add("vf", st.vf); add("lanes", st.lanes); add("cap_lane", st.cap / st.lanes)
    add("len", st.length); add("lane_drop", st.lane_drop); add("cap_ratio", st.cap_ratio)
    add("last_v", last_v); add("last_q", last_q); add("last_k", last_k); add("age", age); add("q_last", qlast)
    for s in range(H + 1):
        add(f"v_s{s}", vn[s])
    for s in (H - 2, H - 1, H):
        add(f"q_s{s}", qn[s]); add(f"k_s{s}", kn[s])
    with np.errstate(invalid="ignore"):
        add("v_trend", np.nanmean(vn[H - 2:], axis=0) - np.nanmean(vn[H - 6:H - 2], axis=0))
        add("q_trend", np.nanmean(qn[H - 2:], axis=0) - np.nanmean(qn[H - 6:H - 2], axis=0))
        add("v_min_hist", np.nanmin(np.where(fin, vn, np.nan), axis=0))
        add("n_queued_hist", (fin & (vn <= 0.6)).sum(axis=0))
    for k in range(-K, K + 1):
        if k == 0:
            continue
        add(f"lv_{k:+d}", _shift(last_v, k)); add(f"lk_{k:+d}", _shift(last_k, k))
        add(f"ql_{k:+d}", _shift(qlast.astype(float), k))
    # Distance to the nearest queued link downstream / upstream (by last observation).
    qi = np.flatnonzero(qlast)
    pos = np.arange(L)
    if len(qi):
        nd = np.searchsorted(qi, pos, side="left")
        dn = np.where(nd < len(qi), qi[np.minimum(nd, len(qi) - 1)], -1)
        up_i = np.searchsorted(qi, pos, side="right") - 1
        up = np.where(up_i >= 0, qi[np.maximum(up_i, 0)], -1)
        add("km_to_q_dn", np.where(dn >= 0, st.x[np.maximum(dn, 0)] - st.x, 99))
        add("km_to_q_up", np.where(up >= 0, st.x - st.x[np.maximum(up, 0)], 99))
    else:
        add("km_to_q_dn", 99); add("km_to_q_up", 99)
    add("n_queued_now", qlast.sum())
    # Queue dynamics: where the queue edges were 15 and 30 minutes ago, and the shockwave speed of
    # the tail just downstream, w = (q_up - q_queue) / (k_queue - k_up) (km/h, > 0 grows upstream).
    for s in (H - 3, H - 6):
        dn_s, up_s, _ = _queue_dist(st, fin[s] & (vn[s] <= 0.6))
        add(f"km_to_q_dn_m{H - s}", dn_s); add(f"km_to_q_up_m{H - s}", up_s)
        add(f"n_queued_m{H - s}", (fin[s] & (vn[s] <= 0.6)).sum())
    dn_now, _, j = _queue_dist(st, qlast)
    vl, ql = last_v * st.vf, last_q * st.cap
    kl = ql / np.maximum(vl, 1)
    jm = np.maximum(j - 1, 0)
    ok = (j > 0) & ~qlast[jm]
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(ok, (ql[jm] - ql[np.maximum(j, 0)]) / (kl[np.maximum(j, 0)] - kl[jm]), np.nan)
    add("tail_w", w)
    add("tail_reach_slots", np.where(np.isfinite(w) & (w > 0), dn_now / np.maximum(w, 1e-3) * 12, 99))
    h = (slot + 6) // 12 % 24
    hours = [(h - 1) % 24, h, (h + 1) % 24]
    add("onset_prior", onset_prior[:, hours] @ np.array([0.5, 1.0, 0.5]))
    add("queued_prior", queued_prior[:, hours] @ np.array([0.5, 1.0, 0.5]))
    for k, a in (extra or {}).items():
        add(k, a)
        if k in ("recent_q", "recent_onset", "today_q_before"):
            for s in (-2, -1, 1, 2):
                add(f"{k}_{s:+d}", _shift(a, s))
    if extra is not None and "bn_d95" in extra:            # TFB_BNECK=1
        for k, a in bneck_window(st, extra, last_v, last_q, qn[H - 2:]).items():
            add(k, a)
    return np.column_stack(F), names


def _queue_dist(st: Static, qm: np.ndarray):
    """km to the nearest queued link downstream / upstream, and the downstream one's index (-1 if none)."""
    qi = np.flatnonzero(qm)
    if not len(qi):
        return np.full(st.L, 99.0), np.full(st.L, 99.0), np.full(st.L, -1)
    pos = np.arange(st.L)
    nd = np.searchsorted(qi, pos, side="left")
    dn = np.where(nd < len(qi), qi[np.minimum(nd, len(qi) - 1)], -1)
    ui = np.searchsorted(qi, pos, side="right") - 1
    up = np.where(ui >= 0, qi[np.maximum(ui, 0)], -1)
    return (np.where(dn >= 0, st.x[np.maximum(dn, 0)] - st.x, 99.0),
            np.where(up >= 0, st.x - st.x[np.maximum(up, 0)], 99.0), dn)


def _shift(a: np.ndarray, k: int) -> np.ndarray:
    """a[i + k] (NaN outside)."""
    out = np.full(a.shape, np.nan, np.float64)
    if k > 0:
        out[:-k] = a[k:]
    elif k < 0:
        out[-k:] = a[:k]
    else:
        out[:] = a
    return out


def build_train(panel: str, mode: str = "train"):
    """Feature blocks for train-split windows of one panel.

    mode "train": dense training_origins; mode "eval": official-like selector windows."""
    st = Static(panel)
    v, q, e, vm, qm = train_arrays(st)
    dl = days(panel, "train")
    months = pd.Index([x[:7] for x in dl])
    mcode = pd.factorize(months, sort=True)[0]
    dow = pd.to_datetime(pd.Series(dl)).dt.dayofweek.to_numpy()
    onset_c, queued_c, seen_c = priors(st, v, e, mcode)
    wins = training_origins(v, e, st.cut[None, :]) if mode == "train" else select_windows(v, e, st.cut[None, :])
    rec = Recent(st)
    rows = {"onset": [], "ongoing": []}
    names = None
    for d, t, cond in wins:
        m = mcode[d]
        # Onsets per day, so the leave-one-month-out prior matches the all-months one used at scoring time.
        op = (onset_c.sum(0) - onset_c[m]) / (len(mcode) - (mcode == m).sum())
        qp = (queued_c.sum(0) - queued_c[m]) / np.maximum(seen_c.sum(0) - seen_c[m], 1)
        V = np.vstack([v[d, t - H:t], vm[d, t][None]])
        Q = np.vstack([q[d, t - H:t], qm[d, t][None]])
        X, names = window_features(st, V, Q, t, dow[d], op, qp, rec.features("train", d, t))
        lab = np.isfinite(v[d, t + 1:t + 7]) & (v[d, t + 1:t + 7] <= st.cut[None, :])
        el = e[d, t + 1:t + 7] & np.isfinite(v[d, t + 1:t + 7])
        key = "onset" if cond == "queue_onset" else "ongoing"
        rows[key].append(dict(X=X, y=lab, elig=el, day=dl[d], slot=t, panel=panel))
    return rows, names


def scored_windows(panel: str, split: str):
    """Feature blocks for the published windows of a scored split (history <= T only)."""
    st = Static(panel)
    o = st.order
    wi = pd.read_csv(REL / "task2" / panel / split / "window_index.csv")
    hist = pd.read_parquet(REL / "task2" / panel / split / "window_history.parquet",
                           columns=["window_id", "timestamp", "link_id", "speed_kmh", "flow_vph"])
    li = {l: i for i, l in enumerate(network(panel).link_id)}
    rank = np.empty(st.L, int); rank[o] = np.arange(st.L)
    dl = {d: i for i, d in enumerate(days(panel, split))}
    vs = load(panel, split, "v", mmap=True); qs = load(panel, split, "q", mmap=True)
    tr = days(panel, "train")
    v, _, e, _, _ = train_arrays(st)
    mcode = pd.factorize(pd.Index([x[:7] for x in tr]), sort=True)[0]
    onset_c, queued_c, seen_c = priors(st, v, e, mcode)
    op = onset_c.sum(0) / len(mcode); qp = queued_c.sum(0) / np.maximum(seen_c.sum(0), 1)
    rec = Recent(st)
    out = []
    for w in wi.itertuples(index=False):
        origin = str(w.forecast_origin)
        slot = int(origin[11:13]) * 12 + int(origin[14:16]) // 5
        h = hist[hist.window_id == w.window_id]
        ts = h.timestamp.str
        s = (ts[11:13].astype(int) * 12 + ts[14:16].astype(int) // 5).to_numpy() - (slot - H)
        V = np.full((H + 1, st.L), np.nan); Q = np.full((H + 1, st.L), np.nan)
        lr = rank[h.link_id.map(li).to_numpy()]
        V[s, lr] = h.speed_kmh.to_numpy(); Q[s, lr] = h.flow_vph.to_numpy()
        d = dl[origin[:10]]
        V[H] = np.asarray(vs[d, slot], float)[o]; Q[H] = np.asarray(qs[d, slot], float)[o]
        dow = pd.Timestamp(origin[:10]).dayofweek
        X, names = window_features(st, V, Q, slot, dow, op, qp, rec.features(split, d, slot))
        out.append(dict(X=X, window_id=w.window_id, condition=w.condition, order=o, panel=panel))
    return out, names
