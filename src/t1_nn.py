"""Neural Task 1 imputer: a dilated 2D residual conv net over (time 288 x links) per (panel, day).

Inputs are the scored setting itself: the released masked layer (train split: vm/qm, scored splits: v/q), plus
month-own (slot, link) profiles and the free-flow shock / level read from the same month's masked layer
(offline imputation, as t1_features). Labels are the full layer at the template's target cells (train split
only). Training days get synthetic Task 2 gaps (all links hidden for 18 slots) half of the time, because the
validation/private masked layers have them and the train split does not.

  python t1_nn.py train --tag h      # Jun-Jan fit, Feb holdout report (cells of t1_nn_gbdt_feb.py)
  python t1_nn.py train --tag f --all-days --epochs N    # Jun-Feb fit for the scored splits
  python t1_nn.py predict --tags h f # validation/private npz (t1_<panel>_<split>.npz, V/Q in cache layout)
"""
from __future__ import annotations

import argparse
import json
import queue
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import ndtr

from common import FAMILY, PANELS, ROOT, T, days, load
import os

from t1_features import FREE_KM_PER_STEP, VCLIP, WAVE_KM_PER_STEP, Grid, _fill_index, _shift_t, _window_mean

BASE = os.environ.get("T1NN_BASE", "1") == "1"   # GBDT-style local interpolation channels + output base
CHAR = os.environ.get("T1NN_CHAR", "1") == "1"   # km-based windows / characteristic lines as channels (minus base)
V2 = os.environ.get("T1NN_V2", "0") == "1"       # + ramp-flow demand ratio (ramps keep reporting in Task 2 gaps), congestion frequency
MD = os.environ.get("T1NN_MD", "0") == "1"       # + neighbouring days (d-1, d+1, d-7, d+7) of the same split's masked layer
MD_K = (-1, 1, -7, 7)

OUTD = ROOT / "out_t1nn"
FEB = "2031-02-01"
GAP = 18


class PanelData:
    """One panel / split: memmapped day slices in travel order plus per-month profiles and free-flow levels."""

    def __init__(self, panel: str, split: str):
        g = Grid(panel, split, d0=0, d1=1)
        self.panel, self.split, self.pno = panel, split, PANELS.index(panel)
        self.order, self.L = g.order, g.L
        self.vf, self.lanes, self.cap = g.vf.astype(np.float32), g.lanes.astype(np.float32), g.cap.astype(np.float32)
        x = g.x
        self.x = g.x
        self.rank = g.rank
        dx_up = np.r_[2.0, np.diff(x)]; dx_dn = np.r_[np.diff(x), 2.0]
        self.static = np.stack([self.lanes / 5, self.vf / 100, np.minimum(g.length, 2), self.cap / self.lanes / 2000,
                                x / max(x.max(), 1e-6), np.minimum(dx_up, 2), np.minimum(dx_dn, 2)]).astype(np.float32)
        tr = split == "train"
        self.V = load(panel, split, "vm" if tr else "v", mmap=True)
        self.Q = load(panel, split, "qm" if tr else "q", mmap=True)
        self.E = load(panel, split, "elig", mmap=True)
        self.TG = load(panel, split, "target", mmap=True)
        self.P = load(panel, split, "pct", mmap=True)
        if tr:
            self.VT = load(panel, split, "v", mmap=True); self.QT = load(panel, split, "q", mmap=True)
        self.regime = load(panel, split, "regime")
        self.days = days(panel, split)
        import pandas as pd
        self.dow = pd.to_datetime(pd.Series(self.days)).dt.dayofweek.to_numpy()
        months = sorted({d[:7] for d in self.days})
        self.mid = np.array([months.index(d[:7]) for d in self.days])
        self.prof_v, self.prof_q, self.has_prof, self.lev, self.sig, self.c_day, self.rs_day = [], [], [], [], [], [], []
        self.cf, self.rprof = [], []
        if V2:
            from common import REL, link_index
            rm = pd.read_csv(REL / 'corridors' / panel / 'network' / 'ramp_attachment_map.csv', dtype=str)
            li = link_index(panel)
            pos = np.array([self.rank[li[x]] if x in li else -1 for x in rm.nearest_mainline_link_id])
            self.RQ = load(panel, split, 'rq', mmap=True)
            assert self.RQ.shape[2] == len(pos)
            self.rvalid = pos >= 0
            xr = np.where(self.rvalid, self.x[np.maximum(pos, 0)], 1e9)
            self.RW = (np.abs(self.x[:, None] - xr[None, :]) <= 3.0).astype(np.float32)   # (L, R) ramps within 3 km
        o = self.order
        for m in range(len(months)):
            idx = np.flatnonzero(self.mid == m)
            V = np.asarray(self.V[idx[0]:idx[-1] + 1])[:, :, o]; Q = np.asarray(self.Q[idx[0]:idx[-1] + 1])[:, :, o]
            e = np.isfinite(V) & np.isfinite(Q) & np.asarray(self.E[idx[0]:idx[-1] + 1])[:, :, o].astype(bool)
            n = e.sum(0)
            with np.errstate(invalid="ignore", divide="ignore"):
                pv = np.where(e, V, 0).sum(0) / n / self.vf; pq = np.where(e, Q, 0).sum(0) / n / self.cap
            # fill never-observed (slot, link) cells: slot-smoothed then link mean
            for a, fb in ((pv, 1.0), (pq, 0.3)):
                lm = np.nanmean(np.where(np.isfinite(a), a, np.nan), axis=0)
                lm = np.where(np.isfinite(lm), lm, fb)
                bad = ~np.isfinite(a)
                a[bad] = np.broadcast_to(lm, a.shape)[bad]
            self.prof_v.append(pv.astype(np.float32)); self.prof_q.append(pq.astype(np.float32))
            self.has_prof.append(np.minimum(n, 20).astype(np.float32) / 20)
            Vf = np.where(e, V, np.nan)
            fr = Vf > 0.9 * self.vf
            with np.errstate(invalid="ignore"):
                lv = np.nanmedian(np.where(fr, Vf, np.nan).reshape(-1, self.L), axis=0)
            lv = np.where(np.isfinite(lv), lv, self.vf).astype(np.float32)
            use = fr & (lv < 118.0) & (Vf < VCLIP - 0.01)
            rel = np.where(use, Vf / lv - 1, np.nan)
            with np.errstate(invalid="ignore"):
                c = np.nanmedian(rel, axis=2)
            c = np.where(np.isfinite(c), c, 0.0)
            s = np.nanstd(rel - c[:, :, None])
            self.lev.append(lv); self.sig.append(float(s) if np.isfinite(s) and s > 0 else 0.01)
            if V2:
                self.cf.append(((e & (V < 0.6 * self.vf)).sum(0) / np.maximum(n, 1)).astype(np.float32))
                R_ = np.asarray(self.RQ[idx[0]:idx[-1] + 1], np.float32)
                with np.errstate(invalid='ignore'), __import__('warnings').catch_warnings():
                    __import__('warnings').simplefilter('ignore')
                    rp = np.nanmean(R_, axis=0)
                    rmn = np.nanmean(rp, axis=0)
                rp = np.where(np.isfinite(rp), rp, np.where(np.isfinite(rmn), rmn, 30.0)[None, :])
                self.rprof.append(np.maximum(rp, 3.0).astype(np.float32))
                del R_
            obs = np.isfinite(V) & np.isfinite(Q)
            with np.errstate(invalid="ignore", divide="ignore"), __import__("warnings").catch_warnings():
                __import__("warnings").simplefilter("ignore")
                rs = np.nanmean(np.where(obs, Q / self.cap / np.maximum(pq, 1e-3), np.nan), axis=2)
            self.c_day.append(c.astype(np.float32)); self.rs_day.append(rs.astype(np.float32))
            del V, Q, e, Vf, rel, obs
        self.c_day = np.concatenate(self.c_day); self.rs_day = np.concatenate(self.rs_day)
        self.D = len(self.days)
        ordn = pd.to_datetime(pd.Series(self.days)).map(pd.Timestamp.toordinal).to_numpy()
        self.by_ord = {int(x): i for i, x in enumerate(ordn)}; self.ordn = ordn

    def sample(self, d: int, gap: np.ndarray | None = None):
        """Input (C, T, L) float32; labels (2, T, L) as v/vf and q/cap; loss mask (T, L)."""
        o, vf, cap = self.order, self.vf, self.cap
        v = np.asarray(self.V[d])[:, o].astype(np.float32); q = np.asarray(self.Q[d])[:, o].astype(np.float32)
        e = np.asarray(self.E[d])[:, o].astype(bool)
        pct = np.asarray(self.P[d])[:, o].astype(np.float32) / 100
        obs = np.isfinite(v) & np.isfinite(q)
        if gap is not None:
            obs[gap] = False
            if gap.ndim == 2:   # cell-level hide (t1_nn_ttt): slots left with no visible link act like gap slots
                gap = ~obs.any(1)
        m = self.mid[d]
        pv, pq, hp, lev = self.prof_v[m], self.prof_q[m], self.has_prof[m], self.lev[m]
        vn = np.where(obs, v / vf, 0); qn = np.where(obs, q / cap, 0)
        rv = np.where(obs, vn - pv, 0); rq = np.where(obs, qn - pq, 0)
        # corridor-wide free-flow shock and demand ratio per slot (precomputed; a gap hides every link of its slots)
        c = self.c_day[d].copy(); rs = self.rs_day[d].copy()
        if gap is not None:
            c[gap] = 0.0; rs[gap] = np.nan
        rs = np.convolve(np.where(np.isfinite(rs), rs, 1.0), np.ones(7) / 7, mode="same").astype(np.float32)
        mu = lev[None, :] * (1 + c[:, None]); sd = self.sig[m] * lev[None, :]
        a = (VCLIP - mu) / sd
        Pa = ndtr(a)
        ps = (mu * Pa - sd * np.exp(-0.5 * a * a) / np.sqrt(2 * np.pi) + VCLIP * (1 - Pa)) / vf
        tt = np.arange(T, dtype=np.float32) / T
        L = self.L
        full = lambda x: np.broadcast_to(np.asarray(x, np.float32), (T, L))
        col = lambda x: np.broadcast_to(np.asarray(x, np.float32)[:, None], (T, L))
        row = lambda x: np.broadcast_to(np.asarray(x, np.float32)[None, :], (T, L))
        reg = self.regime[d]
        ch = [vn, qn, obs, e, pct, pv, pq, hp, rv, rq,
              full(reg == 0), full(reg == 1), full(reg == 2),
              col(np.sin(2 * np.pi * tt)), col(np.cos(2 * np.pi * tt)), col(np.sin(4 * np.pi * tt)), col(np.cos(4 * np.pi * tt)),
              full(self.dow[d] >= 5), full(self.dow[d] / 6),
              *[row(s) for s in self.static],
              row(lev / vf), col(c * 50), ps, col(rs), col(obs.mean(1))]
        if BASE:
            ch += self.base_channels(v, q, obs, pv, pq)
        if V2:
            from scipy.ndimage import uniform_filter1d
            r = np.asarray(self.RQ[d], np.float32) / self.rprof[m]
            fin = np.isfinite(r) & self.rvalid[None, :]
            r0 = np.where(fin, np.clip(r, 0, 4), 0).astype(np.float32)
            nf = fin.sum(1)
            corr = np.where(nf > 0, r0.sum(1) / np.maximum(nf, 1), 1.0).astype(np.float32)
            num = r0 @ self.RW.T; den = fin.astype(np.float32) @ self.RW.T
            loc = np.where(den > 0, num / np.maximum(den, 1e-6), corr[:, None]).astype(np.float32)
            corr = uniform_filter1d(corr, 5, mode="nearest"); loc = uniform_filter1d(loc, 5, axis=0, mode="nearest")
            ch += [col(corr), loc, self.cf[m]]
        if MD:   # offline imputation: other days of the same released masked layer (as the month profiles)
            for k in MD_K:
                j = self.by_ord.get(int(self.ordn[d]) + k)
                if j is None:
                    ch += [full(0), full(0), full(0)]
                    continue
                v2 = np.asarray(self.V[j])[:, o].astype(np.float32); q2 = np.asarray(self.Q[j])[:, o].astype(np.float32)
                ob2 = np.isfinite(v2) & np.isfinite(q2)
                ch += [np.where(ob2, v2 / vf - pv, 0), np.where(ob2, q2 / cap - pq, 0), ob2]
        x = np.stack([np.asarray(z, np.float32) for z in ch])
        if self.split != "train":
            return x, None, None
        tv = np.asarray(self.VT[d])[:, o] / vf; tq = np.asarray(self.QT[d])[:, o] / cap
        msk = np.asarray(self.TG[d])[:, o].astype(bool) & np.isfinite(tv) & np.isfinite(tq)
        y = np.stack([np.nan_to_num(tv), np.nan_to_num(tq)]).astype(np.float32)
        return x, y, msk


    def base_channels(self, v, q, obs, pv, pq):
        """t1_train.bases on the day grid: own-link time interpolation for speed, same-slot +-0.6 km mean for flow."""
        vf, cap, L = self.vf, self.cap, self.L
        vnn = np.where(obs, v / vf, np.nan)[None]; qo = np.where(obs, q, np.nan)[None]
        fi, bi = _fill_index(obs[None], True)[0], _fill_index(obs[None], False)[0]
        li, ti = np.arange(L)[None, :], np.arange(T)[:, None]
        hp_, hn_ = fi >= 0, bi < T
        pv_ = np.where(hp_, vnn[0][np.maximum(fi, 0), li], np.nan); nv_ = np.where(hn_, vnn[0][np.minimum(bi, T - 1), li], np.nan)
        pq_ = np.where(hp_, qo[0][np.maximum(fi, 0), li], np.nan); nq_ = np.where(hn_, qo[0][np.minimum(bi, T - 1), li], np.nan)
        gp = np.where(hp_, ti - fi, 99).astype(np.float32); gn = np.where(hn_, bi - ti, 99).astype(np.float32)
        w = np.where(hp_ & hn_, gn / np.maximum(gp + gn, 1), np.where(hp_, 1.0, 0.0))
        iv = np.where(hp_ & hn_, w * pv_ + (1 - w) * nv_, np.where(hp_, pv_, nv_))
        iq = np.where(hp_ & hn_, w * pq_ + (1 - w) * nq_, np.where(hp_, pq_, nq_))
        q06, n06 = _window_mean(qo, obs[None], self.x, -0.6, 0.6, True)
        v06, _ = _window_mean(vnn, obs[None], self.x, -0.6, 0.6, True)
        q15, _ = _window_mean(qo, obs[None], self.x, -1.5, 1.5, True)
        v15, _ = _window_mean(vnn, obs[None], self.x, -1.5, 1.5, True)
        q06, n06, v06, q15, v15 = q06[0], n06[0], v06[0], q15[0], v15[0]
        bv = iv.copy()
        for alt in (v15,):
            bv = np.where(np.isfinite(bv), bv, alt)
        bq = q06.copy()
        for alt in (q15, iq):
            bq = np.where(np.isfinite(bq), bq, alt)
        bv = np.where(np.isfinite(bv), bv, pv); bq = np.where(np.isfinite(bq), bq / cap, pq)
        extra = []
        if CHAR:
            ob = obs[None]
            fq = []
            for k in (1, 2):
                dx = FREE_KM_PER_STEP * k
                fq.append(_shift_t(_window_mean(qo, ob, self.x, -dx - 1.5, -dx + 1.5, False)[0], -k))
                fq.append(_shift_t(_window_mean(qo, ob, self.x, dx - 1.5, dx + 1.5, False)[0], k))
            fq += [q15[None], _window_mean(qo, ob, self.x, -4, 4, True)[0]]
            fv = [_shift_t(_window_mean(vnn, ob, self.x, WAVE_KM_PER_STEP - 0.5, WAVE_KM_PER_STEP + 0.5, False)[0], -1),
                  _shift_t(_window_mean(vnn, ob, self.x, -WAVE_KM_PER_STEP - 0.5, -WAVE_KM_PER_STEP + 0.5, False)[0], 1),
                  v15[None], _window_mean(vnn, ob, self.x, -4, 4, True)[0]]
            extra = [5 * np.nan_to_num(f[0] / cap - bq) for f in fq] + [5 * np.nan_to_num(f[0] - bv) for f in fv]
        return extra + [bv, bq, np.isfinite(iv), np.nan_to_num(iv), np.isfinite(v06), np.nan_to_num(v06), np.nan_to_num(q06) / cap,
                np.minimum(gp, 48) / 48, np.minimum(gn, 48) / 48]


N_IN = ((40 + 10 * CHAR) if BASE else 31) + 3 * V2 + 3 * len(MD_K) * MD
B0 = 41 if CHAR else 31   # channel of bv (bq follows)


class Block(nn.Module):
    def __init__(self, w: int, dt: int, dl: int):
        super().__init__()
        self.n1 = nn.GroupNorm(8, w); self.c1 = nn.Conv2d(w, w, 3, padding=(dt, dl), dilation=(dt, dl))
        self.n2 = nn.GroupNorm(8, w); self.c2 = nn.Conv2d(w, w, 3, padding=1)
        self.g = nn.Linear(w, w)

    def forward(self, x):
        h = self.c1(F.gelu(self.n1(x)))
        h = h + self.g(h.mean(3).transpose(1, 2)).transpose(1, 2)[..., None]   # corridor-wide context per slot
        return x + self.c2(F.gelu(self.n2(h)))


DIL = [(1, 1), (2, 2), (4, 4), (8, 8), (16, 4), (1, 1), (2, 2), (4, 4), (8, 8), (32, 4)]


class Net(nn.Module):
    def __init__(self, w: int = 64, n_in: int = N_IN):
        super().__init__()
        self.stem = nn.Conv2d(n_in, w, 3, padding=1)
        self.blocks = nn.ModuleList([Block(w, *d) for d in DIL])
        self.head = nn.Sequential(nn.GroupNorm(8, w), nn.GELU(), nn.Conv2d(w, w, 1), nn.GELU(), nn.Conv2d(w, 2, 1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        h = self.stem(x)
        for b in self.blocks:
            h = b(h)
        out = self.head(h).float()
        if BASE:   # local interpolation base (month profile where nothing is visible) + correction
            bv, bq = x[:, B0], x[:, B0 + 1]
            return torch.stack([bv.float() + out[:, 0], bq.float() + out[:, 1]], 1)
        return torch.stack([x[:, 5] + out[:, 0], x[:, 6] + out[:, 1]], 1)   # month profile + correction


def batch_size(L: int) -> int:
    return max(1, 384 // L)


def to_dev(xs, dev):
    return torch.from_numpy(np.stack(xs)).to(dev, non_blocking=True)


def loss_fn(pred, y, msk, scale_v, scale_q):
    """S_state units: speed error / 25 km/h and per-lane flow error / 600 vph, per-sample mean over target cells."""
    ev = (pred[:, 0] - y[:, 0]) * scale_v[:, None, :]
    eq = (pred[:, 1] - y[:, 1]) * scale_q[:, None, :]
    m = msk.float()
    per = ((ev ** 2 + 0.95 * eq ** 2) * m).sum((1, 2)) / m.sum((1, 2)).clamp(min=1)
    return per.sum() / 2


GAP_P, GAP_W, GAP_LO, GAP_HI = 0.5, 1.0, 30, T - GAP


def make_gap(rng) -> np.ndarray | None:
    if rng.random() >= GAP_P:
        return None
    g = np.zeros(T, bool)
    for s in rng.integers(GAP_LO, GAP_HI, size=rng.integers(1, 4)):
        g[s:s + GAP] = True
    return g


def batches(data: list[PanelData], day_sets: list[np.ndarray], rng):
    out = []
    for pi, pd_ in enumerate(data):
        ds = rng.permutation(day_sets[pi])
        b = batch_size(pd_.L)
        out += [(pi, ds[i:i + b]) for i in range(0, len(ds), b)]
    return [out[i] for i in rng.permutation(len(out))]


def producer(data, plan, q_: queue.Queue, seed: int, gaps: bool):
    rng = np.random.default_rng(seed)
    for pi, ds in plan:
        pd_ = data[pi]
        xs, ys, ms = [], [], []
        for d in ds:
            gp_ = make_gap(rng) if gaps else None
            x, y, m = pd_.sample(int(d), gp_)
            m = m.astype(np.float32)
            if gp_ is not None and GAP_W != 1.0:
                m = m * np.where(gp_, GAP_W, 1.0).astype(np.float32)[:, None]
            xs.append(x); ys.append(y); ms.append(m)
        q_.put((pi, np.stack(xs), np.stack(ys), np.stack(ms)))
    q_.put(None)


@torch.no_grad()
def predict_days(net, pd_: PanelData, ds, dev, gaps: dict | None = None):
    """Returns predicted v (km/h) and q (vph) of shape (len(ds), T, L) in travel order."""
    net.eval()
    V = np.zeros((len(ds), T, pd_.L), np.float32); Q = np.zeros_like(V)
    b = batch_size(pd_.L)
    for i in range(0, len(ds), b):
        xs = [pd_.sample(int(d), None if gaps is None else gaps.get(int(d)))[0] for d in ds[i:i + b]]
        with torch.autocast("cuda", dtype=torch.float16):
            p = net(to_dev(xs, dev)).float().cpu().numpy()
        V[i:i + len(xs)] = p[:, 0] * pd_.vf; Q[i:i + len(xs)] = p[:, 1] * pd_.cap
    net.train()
    return np.clip(V, 3, 140), np.clip(Q, 60, None)


def s_state(pv, pq, yv, yq, lanes, panel, regime):
    per = {}
    for p in np.unique(panel):
        s = []
        for r in range(3):
            m = (panel == p) & (regime == r)
            if m.any():
                rv = np.sqrt(np.mean((pv[m] - yv[m]) ** 2)); rq = np.sqrt(np.mean(((pq[m] - yq[m]) / lanes[m]) ** 2))
                s.append(0.54 * max(0, 1 - rv / 25) + 0.46 * max(0, 1 - rq / 600))
        per[PANELS[p]] = float(np.mean(s))
    fam = {}
    for p, s in per.items():
        fam.setdefault(FAMILY[p], []).append(s)
    return float(np.mean([np.mean(v) for v in fam.values()]))


def feb_eval(net, data, dev, ref: dict, gaps_ref: dict | None = None):
    """NN predictions at the reference (GBDT) cells. ref: arrays of t1_nn_gbdt_feb.py."""
    pv = np.zeros(len(ref["d"]), np.float32); pq = np.zeros_like(pv)
    for pi, pd_ in enumerate(data):
        m = ref["panel"] == pi
        if not m.any():
            continue
        d = ref["d"][m]
        ds = np.unique(d)
        gp = None
        if gaps_ref is not None:
            gp = {int(x): gaps_ref[pi][j] for j, x in enumerate(ds)}
        V, Q = predict_days(net, pd_, ds, dev, gp)
        pos = np.searchsorted(ds, d)
        pv[m] = V[pos, ref["t"][m], ref["l"][m]]; pq[m] = Q[pos, ref["t"][m], ref["l"][m]]
    return pv, pq


def report(name, pv, pq, ref, sub=None):
    m = np.ones(len(pv), bool) if sub is None else sub
    a = [x[m] for x in (pv, pq, ref["yv"], ref["yq"], ref["lanes"], ref["panel"], ref["regime"])]
    s = s_state(*a)
    rv = np.sqrt(np.mean((a[0] - a[2]) ** 2)); rq = np.sqrt(np.mean(((a[1] - a[3]) / a[4]) ** 2))
    regs = []
    for r in range(3):
        k = a[6] == r
        regs.append((s_state(*[x[k] for x in a]), np.sqrt(np.mean((a[0][k] - a[2][k]) ** 2)),
                     np.sqrt(np.mean(((a[1][k] - a[3][k]) / a[4][k]) ** 2))))
    txt = f"{name:22s} S_state {s:.5f} rmse_v {rv:.3f} rmse_q_lane {rq:.2f} | " + " ".join(
        f"R{r + 1} {x[0]:.4f} ({x[1]:.2f}/{x[2]:.1f})" for r, x in enumerate(regs))
    print(txt, flush=True)
    return dict(S=s, rmse_v=float(rv), rmse_q=float(rq), regimes=regs)


def train(a):
    global GAP_P, GAP_W, GAP_LO, GAP_HI
    GAP_P, GAP_W = a.gap_p, a.gap_w
    if a.gap_real:
        GAP_LO, GAP_HI = 55, 251
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    t0 = time.time()
    data = [PanelData(p, "train") for p in a.panels]
    print(f"data ready {time.time() - t0:.0f}s", flush=True)
    day_sets = [np.array([i for i, d in enumerate(pd_.days) if a.all_days or d < FEB]) for pd_ in data]
    ref = dict(np.load(OUTD / "feb_gbdt.npz")) if not a.all_days else None
    net = Net(a.width).to(dev).to(memory_format=torch.channels_last)
    print(f"params {sum(p.numel() for p in net.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    steps_ep = sum(-(-len(s) // batch_size(pd_.L)) for s, pd_ in zip(day_sets, data))
    total = steps_ep * a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.08, final_div_factor=50)
    scaler = torch.amp.GradScaler()
    sv = [torch.from_numpy(pd_.vf / 25).to(dev) for pd_ in data]
    sq = [torch.from_numpy(pd_.cap / pd_.lanes / 600).to(dev) for pd_ in data]
    rng = np.random.default_rng(a.seed)
    log = []
    for ep in range(a.epochs):
        plan = batches(data, day_sets, rng)
        q_ = queue.Queue(maxsize=6)
        th = threading.Thread(target=producer, args=(data, plan, q_, a.seed * 1000 + ep, not a.no_gaps), daemon=True)
        th.start()
        tl, n, te = 0.0, 0, time.time()
        while (item := q_.get()) is not None:
            pi, x, y, m = item
            x = torch.from_numpy(x).to(dev, non_blocking=True).to(memory_format=torch.channels_last)
            y = torch.from_numpy(y).to(dev); m = torch.from_numpy(m).to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                p = net(x)
            loss = loss_fn(p.float(), y, m, sv[pi].expand(len(x), -1), sq[pi].expand(len(x), -1))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tl += float(loss) * 2 / len(x); n += 1
        msg = f"ep {ep + 1}/{a.epochs} loss {tl / n:.5f} {time.time() - te:.0f}s (total {time.time() - t0:.0f}s, " \
              f"gpu max {torch.cuda.max_memory_allocated() / 2**30:.2f} GB)"
        print(msg, flush=True)
        ent = dict(ep=ep + 1, loss=tl / n)
        if ref is not None and ((ep + 1) % a.eval_every == 0 or ep + 1 == a.epochs):
            pv, pq = feb_eval(net, data, dev, ref)
            ent["feb"] = report(f"  NN ep{ep + 1}", pv, pq, ref)
        log.append(ent)
        torch.save(net.state_dict(), OUTD / f"net_{a.tag}.pt")
    if ref is not None:
        np.savez_compressed(OUTD / f"feb_nn_{a.tag}.npz", pv=pv, pq=pq)
        rg = OUTD / "feb_gbdt_gap.npz"
        if rg.exists():
            gref = dict(np.load(rg))
            from t1_nn_gbdt_feb import gap_slots
            gaps = [gap_slots(p, int((np.array(pd_.days) >= FEB).sum())) for p, pd_ in zip(a.panels, data)]
            pv, pq = feb_eval(net, data, dev, gref, gaps)
            np.savez_compressed(OUTD / f"feb_nn_gap_{a.tag}.npz", pv=pv, pq=pq)
    (OUTD / f"log_{a.tag}.json").write_text(json.dumps(dict(args=vars(a), log=log, secs=time.time() - t0), indent=1, default=str))
    print(f"done {time.time() - t0:.0f}s", flush=True)


def predict(a):
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")
    nets = []
    for tag in a.tags:
        net = Net(a.width).to(dev).to(memory_format=torch.channels_last)
        net.load_state_dict(torch.load(OUTD / f"net_{tag}.pt", map_location=dev))
        nets.append(net)
    od = OUTD / a.out_sub
    od.mkdir(exist_ok=True)
    for p in a.panels:
        for s in ("validation", "private"):
            pd_ = PanelData(p, s)
            ds = np.arange(pd_.D)
            V = np.zeros((pd_.D, T, pd_.L), np.float32); Q = np.zeros_like(V)
            for net in nets:
                v, q = predict_days(net, pd_, ds, dev)
                V += v / len(nets); Q += q / len(nets)
            tg = np.asarray(pd_.TG).astype(bool)
            Vs = np.full(tg.shape, np.nan, np.float32); Qs = np.full(tg.shape, np.nan, np.float32)
            Vs[:, :, pd_.order] = V; Qs[:, :, pd_.order] = Q          # travel -> scorer order
            Vs[~tg] = np.nan; Qs[~tg] = np.nan
            np.savez_compressed(od / f"t1_{p}_{s}.npz", V=Vs, Q=Qs)
            print(f"{p} {s}: {int(tg.sum()):,} target cells", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "predict"])
    ap.add_argument("--panels", nargs="*", default=PANELS)
    ap.add_argument("--tag", default="h")
    ap.add_argument("--tags", nargs="*", default=["h"])
    ap.add_argument("--out-sub", default="pred")
    ap.add_argument("--all-days", action="store_true")
    ap.add_argument("--no-gaps", action="store_true")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gap-p", type=float, default=0.5, help="share of training days with synthetic all-link gaps")
    ap.add_argument("--gap-w", type=float, default=1.0, help="loss weight of target cells inside synthetic gaps")
    ap.add_argument("--gap-real", action="store_true", help="gap starts in slots 55-250 like the validation/private gaps")
    a = ap.parse_args()
    OUTD.mkdir(exist_ok=True)
    train(a) if a.mode == "train" else predict(a)


if __name__ == "__main__":
    main()
