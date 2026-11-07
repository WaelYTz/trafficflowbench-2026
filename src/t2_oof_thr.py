"""Per-panel decode thresholds from out-of-fold predictions on the train months.

The recent-memory and memory-free models are refit with each month group held out (fixed rounds), the
0.7 / 0.3 blend is decoded on the held-out windows, and each panel gets the threshold that maximises its
out-of-fold IoU. Thresholds come from train months only, so they are admissible for every scored window.
They are then checked on Feb official-like windows and Feb / March pseudo-windows with the holdout models.
"""
from __future__ import annotations

import argparse
import json
import pickle

import lightgbm as lgb
import numpy as np

from common import OUT, QUEUE_PANELS
from t2_model import HOLD, MEMORY, PARAMS, aggregate, decode, iou, predict_window, stack
from t2_onset_abl import LONG, onset_rows
from t2_onset_recent import refresh

THRS = np.round(np.arange(0.15, 0.61, 0.05), 2)
SUB = dict(window_stride=3, n_h=3, far_keep=0.15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="onset", choices=("onset", "ongoing"))
    ap.add_argument("--folds", type=int, default=4)
    a = ap.parse_args()
    cond, hz, kw, fb, glob_thr, rounds = (("queue_onset", (6,), {}, 3, 0.3, (424, 441)) if a.key == "onset"
                                          else ("queue_ongoing", (1, 2, 3, 4, 5, 6), SUB, 1, 0.45, (4243, 4359)))
    if a.key == "onset":
        tr, ev, names = onset_rows()
    else:
        tr_all, ev_all, names = pickle.load(open(OUT / "t2_rowse.pkl", "rb"))
        tr, ev = tr_all["ongoing"], ev_all["ongoing"]
        del tr_all, ev_all
    feb = [r for r in ev if r["day"] >= HOLD]
    ps = {s: pickle.load(open(OUT / f"t2_pseudo_{s}.pkl", "rb"))[0][a.key] for s in ("feb", "mar", "feb_off", "mar_off")}
    refresh({"train": tr + feb + ps["feb"] + ps["feb_off"], "validation": ps["mar"] + ps["mar_off"]}, names, 7)
    kb = [i for i, n in enumerate(names) if not n.startswith(MEMORY)]
    kr = [i for i, n in enumerate(names) if not n.startswith(LONG)]
    months = sorted({r["day"][:7] for r in tr})
    fold = {m: i % a.folds for i, m in enumerate(months)}
    oof = [None] * len(tr)
    for f in range(a.folds):
        fit_rows = [r for r in tr if fold[r["day"][:7]] != f]
        idx = [i for i, r in enumerate(tr) if fold[r["day"][:7]] == f]
        for keep, wt, n in ((kr, 0.7, rounds[1]), (kb, 0.3, rounds[0])):
            nm = [names[i] for i in keep]
            X, y, w = stack([dict(r, X=r["X"][:, keep]) for r in fit_rows], cond, hz, nm, seed=3, **kw)
            m = lgb.train(PARAMS, lgb.Dataset(X, y, weight=w, feature_name=nm + ["h"]), int(n / 1.1))
            del X, y, w
            for i in idx:
                p = wt * predict_window(m, tr[i]["X"][:, keep], cond)
                oof[i] = p if oof[i] is None else oof[i] + p
        print(f"fold {f}: {len(idx)} windows held out ({[mm for mm in months if fold[mm] == f]})", flush=True)
    best = {}
    for p in QUEUE_PANELS:
        ii = [i for i, r in enumerate(tr) if r["panel"] == p]
        sc = {t: np.mean([iou(decode(oof[i], cond, t, fb), tr[i]["y"], tr[i]["elig"]) for i in ii]) for t in THRS}
        best[p] = float(max(sc, key=sc.get))
        print(f"{p:10s} n={len(ii):5d} OOF IoU at global {glob_thr}: {sc[glob_thr]:.4f} | best thr {best[p]} -> {sc[best[p]]:.4f}", flush=True)
    (OUT / f"t2_oof_thr_{a.key}.json").write_text(json.dumps(best, indent=1))
    # check on the holdout sets with the holdout models
    hold = {"onset": ("t2rec7_onset.txt", "t2abl_base_onset.txt"), "ongoing": ("t2og_rec7_ongoing.txt", "t2og_base_ongoing.txt")}[a.key]
    mr, mb = (lgb.Booster(model_file=str(OUT / f)) for f in hold)
    for s, rs in {"feb": feb, **{("feb_ps" if k == "feb" else k): v for k, v in ps.items()}}.items():
        P = [0.7 * predict_window(mr, r["X"][:, kr], cond) + 0.3 * predict_window(mb, r["X"][:, kb], cond) for r in rs]
        g = aggregate([(r["panel"], cond, iou(decode(p, cond, glob_thr, fb), r["y"], r["elig"])) for p, r in zip(P, rs)])["overall"]
        pp = aggregate([(r["panel"], cond, iou(decode(p, cond, best[r["panel"]], fb), r["y"], r["elig"])) for p, r in zip(P, rs)])["overall"]
        print(f"{s:8s} global thr {g:.4f} | per-panel thr {pp:.4f} | diff {pp - g:+.4f}", flush=True)


if __name__ == "__main__":
    main()
