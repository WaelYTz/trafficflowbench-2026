"""Ramp-demand features for Task 2 and an onset test of them.

Bottleneck shifts in the scored months follow ramp demand (e.g. D12_I5_S on-ramp flow near links 40-56:
1102 vph in Feb, 831 in March, 1154 in April). Ramp flows are released observations, so readings stamped
<= T may be used. Per mainline link (travel order), normalised by link capacity:
  r_on / r_off        attached on-/off-ramp flow, mean of T-10..T
  r_on_tr / r_off_tr  that minus the mean of T-60..T-15
  r_net_up3 / dn3     on - off summed over the link and the 3 links upstream / downstream
A link without ramps gets 0; a link whose ramps are all unobserved gets NaN.
"""
from __future__ import annotations

import argparse
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import OUT, REL, days, load, network
from t2_data import CAL_SPLITS, H, Static
from t2_model import HOLD, MEMORY, PARAMS, aggregate, decode, iou, predict_window, stack
from t2_onset_abl import LONG, onset_rows
from t2_onset_recent import refresh

NAMES = ("r_on", "r_off", "r_on_tr", "r_off_tr", "r_net_up3", "r_net_dn3")


class Ramps:
    def __init__(self, st: Static):
        net = network(st.panel)
        rm = pd.read_csv(REL / "corridors" / st.panel / "network" / "ramp_attachment_map.csv", dtype=str)
        li = {l: i for i, l in enumerate(net.link_id)}
        rank = np.empty(st.L, int); rank[st.order] = np.arange(st.L)
        self.pos = rank[rm.nearest_mainline_link_id.map(li).to_numpy().astype(int)]
        self.on = (rm.ramp_type == "OR").to_numpy()
        self.L, self.cap = st.L, st.cap
        self.rq = {s: load(st.panel, s, "rq", mmap=True) for s in CAL_SPLITS}
        self.has = {k: np.bincount(self.pos[m], minlength=self.L) > 0 for k, m in (("on", self.on), ("off", ~self.on))}

    def _per_link(self, x: np.ndarray, mask: np.ndarray, kind: str) -> np.ndarray:
        s = np.zeros(self.L); n = np.zeros(self.L)
        ok = mask & np.isfinite(x)
        np.add.at(s, self.pos[ok], x[ok]); np.add.at(n, self.pos[mask], np.isfinite(x[mask]))
        return np.where(self.has[kind] & (n == 0), np.nan, s)

    def features(self, split: str, day: int, slot: int) -> dict:
        x = np.asarray(self.rq[split][day, slot - H:slot + 1], np.float64)      # T-60 .. T
        with np.errstate(invalid="ignore"):
            now, past = np.nanmean(x[-3:], axis=0), np.nanmean(x[:-3], axis=0)
        f = {}
        for kind, m in (("on", self.on), ("off", ~self.on)):
            a, b = self._per_link(now, m, kind), self._per_link(past, m, kind)
            f[f"r_{kind}"] = a / self.cap
            f[f"r_{kind}_tr"] = (a - b) / self.cap
        net = np.nan_to_num(f["r_on"]) - np.nan_to_num(f["r_off"])
        c = np.concatenate([[0], np.cumsum(net)])
        i = np.arange(self.L)
        f["r_net_up3"] = c[i + 1] - c[np.maximum(i - 3, 0)]
        f["r_net_dn3"] = c[np.minimum(i + 4, self.L)] - c[i]
        return {k: f[k] for k in NAMES}


def augment(rows_by_split):
    """Append the ramp columns (NAMES order) to each row's X, in place."""
    for panel in sorted({r["panel"] for rs in rows_by_split.values() for r in rs}):
        rp = Ramps(Static(panel))
        for split, rs in rows_by_split.items():
            di = {d: i for i, d in enumerate(days(panel, split))}
            for r in rs:
                if r["panel"] == panel:
                    f = rp.features(split, di[r["day"]], r["slot"])
                    r["X"] = np.column_stack([r["X"]] + [np.asarray(f[k], np.float32) for k in NAMES])


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    COND = "queue_onset"
    tr, ev, names = onset_rows()
    feb = [r for r in ev if r["day"] >= HOLD]
    ps = {s: pickle.load(open(OUT / f"t2_pseudo_{s}.pkl", "rb"))[0]["onset"] for s in ("feb", "mar", "feb_off", "mar_off")}
    groups = {"train": tr + feb + ps["feb"] + ps["feb_off"], "validation": ps["mar"] + ps["mar_off"]}
    refresh(groups, names, 7)
    augment(groups)
    alln = names + list(NAMES)
    ramp = list(range(len(names), len(alln)))
    sets = {"feb": feb, **{("feb_ps" if k == "feb" else k): v for k, v in ps.items()}}
    kb = [i for i, n in enumerate(names) if not n.startswith(MEMORY)]
    kr = [i for i, n in enumerate(names) if not n.startswith(LONG)]
    variants = {"base": kb, "base_ramp": kb + ramp, "rec7": kr, "rec7_ramp": kr + ramp}
    P = {}
    for var, keep in variants.items():
        nm = [alln[i] for i in keep]
        sub = lambda rs: [dict(r, X=r["X"][:, keep]) for r in rs]
        X, y, w = stack(sub([r for r in tr if r["day"] < HOLD]), COND, (6,), nm, seed=1)
        Xv, yv, wv = stack(sub([r for r in tr if r["day"] >= HOLD]), COND, (6,), nm, seed=2)
        dt = lgb.Dataset(X, y, weight=w, feature_name=nm + ["h"])
        m = lgb.train(PARAMS, dt, 4000, valid_sets=[lgb.Dataset(Xv, yv, weight=wv, reference=dt)],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        m.save_model(str(OUT / f"t2ramp_{var}_onset.txt"))
        P[var] = {s: [predict_window(m, r["X"][:, keep], COND) for r in rs] for s, rs in sets.items()}
        g = m.feature_importance("gain"); g = g / g.sum()
        print(f"{var:10s} it {m.best_iteration:4d} ramp gain share {sum(g[nm.index(k)] for k in NAMES if k in nm):.3f}", flush=True)
    pickle.dump(P, open(OUT / "t2ramp_onset_probs.pkl", "wb"))
    combos = {"base": ("base", None), "base_ramp": ("base_ramp", None), "rec7": ("rec7", None), "rec7_ramp": ("rec7_ramp", None),
              "rec7+base": ("rec7", "base"), "rec7_ramp+base": ("rec7_ramp", "base"), "rec7_ramp+base_ramp": ("rec7_ramp", "base_ramp")}
    for name, (a, b) in combos.items():
        line = []
        for s, rs in sets.items():
            PP = P[a][s] if b is None else [0.7 * x + 0.3 * y for x, y in zip(P[a][s], P[b][s])]
            res = {thr: aggregate([(r["panel"], COND, iou(decode(p, COND, thr, 3), r["y"], r["elig"])) for p, r in zip(PP, rs)])["overall"]
                   for thr in (0.3, 0.45)}
            line.append(f"{s} {res[0.3]:.3f}/{res[0.45]:.3f}")
        print(f"{name:20s} thr .3/.45 | " + " | ".join(line), flush=True)


if __name__ == "__main__":
    main()
