"""CatBoost (GPU) on the recent-memory (7-day) features, for onset or ongoing, blended with the LightGBM
recent-memory model and the memory-free base. Scored with the official aggregation on Feb official-like
windows and on Feb / March pseudo-windows (dense and official-style)."""
from __future__ import annotations

import argparse
import pickle

import lightgbm as lgb
import numpy as np
from catboost import CatBoostClassifier

from common import OUT
from t2_model import HOLD, MEMORY, aggregate, decode, iou, predict_window, stack
from t2_onset_abl import LONG, onset_rows
from t2_onset_recent import refresh

SUB = dict(window_stride=3, n_h=3, far_keep=0.15)


def cat_window(m, X, cond):
    L = len(X)
    P = np.zeros((6, L))
    for h in ((6,) if cond == "queue_onset" else range(1, 7)):
        P[h - 1] = m.predict_proba(np.column_stack([X, np.full(L, h, np.float32)]))[:, 1]
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="onset", choices=("onset", "ongoing"))
    ap.add_argument("--iters", type=int, default=6000)
    a = ap.parse_args()
    cond, hz, kw, thr, fb = (("queue_onset", (6,), {}, 0.3, 3) if a.key == "onset"
                             else ("queue_ongoing", (1, 2, 3, 4, 5, 6), SUB, 0.45, 1))
    if a.key == "onset":
        tr, ev, names = onset_rows()
    else:
        tr_all, ev_all, names = pickle.load(open(OUT / "t2_rowse.pkl", "rb"))
        tr, ev = tr_all["ongoing"], ev_all["ongoing"]
        del tr_all, ev_all
    feb = [r for r in ev if r["day"] >= HOLD]
    ps = {s: pickle.load(open(OUT / f"t2_pseudo_{s}.pkl", "rb"))[0][a.key] for s in ("feb", "mar", "feb_off", "mar_off")}
    refresh({"train": tr + feb + ps["feb"] + ps["feb_off"], "validation": ps["mar"] + ps["mar_off"]}, names, 7)
    sets = {"feb": feb, **{("feb_ps" if k == "feb" else k): v for k, v in ps.items()}}
    kb = [i for i, n in enumerate(names) if not n.startswith(MEMORY)]
    keep = [i for i, n in enumerate(names) if not n.startswith(LONG)]
    nm = [names[i] for i in keep]
    base = lgb.Booster(model_file=str(OUT / ("t2abl_base_onset.txt" if a.key == "onset" else "t2og_base_ongoing.txt")))
    recl = lgb.Booster(model_file=str(OUT / ("t2rec7_onset.txt" if a.key == "onset" else "t2og_rec7_ongoing.txt")))
    sub = lambda rs: [dict(r, X=r["X"][:, keep]) for r in rs]
    X, y, w = stack(sub([r for r in tr if r["day"] < HOLD]), cond, hz, nm, seed=1, **kw)
    Xv, yv, wv = stack(sub([r for r in tr if r["day"] >= HOLD]), cond, hz, nm, seed=2, **kw)
    m = CatBoostClassifier(iterations=a.iters, depth=8, learning_rate=0.06, l2_leaf_reg=10, task_type="GPU",
                           random_seed=20261023, loss_function="Logloss", early_stopping_rounds=200, verbose=1000)
    m.fit(X, y, sample_weight=w, eval_set=(Xv, yv), use_best_model=True)
    del X, y, w, Xv, yv, wv
    m.save_model(str(OUT / f"t2catrec7_{a.key}.cbm"))
    P = {}
    for s, rs in sets.items():
        P[s] = dict(base=[predict_window(base, r["X"][:, kb], cond) for r in rs],
                    lgb=[predict_window(recl, r["X"][:, keep], cond) for r in rs],
                    cat=[cat_window(m, r["X"][:, keep], cond) for r in rs])
    pickle.dump(P, open(OUT / f"t2catrec7_{a.key}_probs.pkl", "wb"))
    combos = {"lgb+base (current)": dict(lgb=0.7, base=0.3), "cat+base": dict(cat=0.7, base=0.3),
              "lgb+cat+base": dict(lgb=0.35, cat=0.35, base=0.3), "lgb+cat": dict(lgb=0.5, cat=0.5)}
    for name, wts in combos.items():
        res = {}
        for s, rs in sets.items():
            PP = [sum(wv_ * P[s][k][i] for k, wv_ in wts.items()) for i in range(len(rs))]
            res[s] = aggregate([(r["panel"], cond, iou(decode(p, cond, thr, fb), r["y"], r["elig"])) for p, r in zip(PP, rs)])["overall"]
        print(f"{a.key} {name:20s} | " + " | ".join(f"{s} {v:.4f}" for s, v in res.items()), flush=True)


if __name__ == "__main__":
    main()
