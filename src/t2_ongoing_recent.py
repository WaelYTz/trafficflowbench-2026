"""Ongoing model with recent memory (last 7 days, masked layer) vs the memory-free base, scored with the
official aggregation on Feb official-like windows and on Feb / March pseudo-windows."""
from __future__ import annotations

import argparse
import pickle

import lightgbm as lgb
import numpy as np

from common import OUT
from t2_model import HOLD, MEMORY, PARAMS, aggregate, decode, iou, predict_window, stack
from t2_onset_abl import LONG
from t2_onset_recent import refresh

COND = "queue_ongoing"
HZ = (1, 2, 3, 4, 5, 6)
SUB = dict(window_stride=3, n_h=3, far_keep=0.15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    a = ap.parse_args()
    tr, ev, names = pickle.load(open(OUT / "t2_rowse.pkl", "rb"))
    tr, ev = tr["ongoing"], [r for r in ev["ongoing"] if r["day"] >= HOLD]
    ps = {s: pickle.load(open(OUT / f"t2_pseudo_{s}.pkl", "rb"))[0]["ongoing"] for s in ("feb", "mar")}
    sets = {"feb": ev, "feb_ps": ps["feb"], "mar_ps": ps["mar"]}
    variants = {"base": [i for i, n in enumerate(names) if not n.startswith(MEMORY)],
                f"rec{a.days}": [i for i, n in enumerate(names) if not n.startswith(LONG)]}
    for var, keep in variants.items():
        if var != "base":
            refresh({"train": tr + ev + ps["feb"], "validation": ps["mar"]}, names, a.days)
        nm = [names[i] for i in keep]
        sub = lambda rs: [dict(r, X=r["X"][:, keep]) for r in rs]
        X, y, w = stack(sub([r for r in tr if r["day"] < HOLD]), COND, HZ, nm, seed=1, **SUB)
        Xv, yv, wv = stack(sub([r for r in tr if r["day"] >= HOLD]), COND, HZ, nm, seed=2, **SUB)
        dt = lgb.Dataset(X, y, weight=w, feature_name=nm + ["h"])
        m = lgb.train(PARAMS, dt, 6000, valid_sets=[lgb.Dataset(Xv, yv, weight=wv, reference=dt)],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        del X, y, w, Xv, yv, wv
        m.save_model(str(OUT / f"t2og_{var}_ongoing.txt"))
        line = []
        for s, rs in sets.items():
            P = [predict_window(m, r["X"][:, keep], COND) for r in rs]
            res = {thr: aggregate([(r["panel"], COND, iou(decode(p, COND, thr), r["y"], r["elig"])) for p, r in zip(P, rs)])["overall"]
                   for thr in (0.35, 0.45, 0.55)}
            line.append(f"{s} " + " ".join(f"{res[t]:.4f}" for t in (0.35, 0.45, 0.55)))
        print(f"{var:6s} it {m.best_iteration:5d} | thr .35/.45/.55 | " + " | ".join(line), flush=True)


if __name__ == "__main__":
    main()
