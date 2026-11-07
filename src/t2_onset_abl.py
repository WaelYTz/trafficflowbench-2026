"""Onset feature ablations scored on Feb (official-like windows), Feb pseudo and March pseudo windows.

The onset model leans on place features (pos, panel, slot ~40% of gain). March pseudo-windows show every
model losing ~0.11 onset IoU, so this checks whether dropping place information transfers better.
"""
from __future__ import annotations

import argparse
import pickle

import lightgbm as lgb
import numpy as np

from common import OUT
from t2_model import HOLD, MEMORY, PARAMS, decode, iou, predict_window, stack

COND = "queue_onset"
PLACE = ("panel", "pos", "slot", "dow")
STATIC = ("vf", "lanes", "cap_lane", "len", "lane_drop", "cap_ratio")
LONG = ("onset_prior", "queued_prior")   # long-term train-month place memory
VARIANTS = {
    "base": (),
    "no_place": PLACE,
    "no_pos_panel": ("panel", "pos"),
    "no_place_static": PLACE + STATIC,
    # recent (last 14 days, masked layer) memory instead of static place identity
    "recent": ("+recent",),
    "recent_no_pos_panel": ("+recent", "panel", "pos"),
    "recent_no_place_static": ("+recent",) + PLACE + STATIC,
    "recent_no_pos_panel_static": ("+recent", "panel", "pos") + STATIC,
}


def onset_rows():
    f = OUT / "t2_rowse_onset.pkl"
    if not f.exists():
        tr, ev, names = pickle.load(open(OUT / "t2_rowse.pkl", "rb"))
        pickle.dump((tr["onset"], ev["onset"], names), open(f, "wb"))
        del tr, ev
    return pickle.load(open(f, "rb"))


def main():
    tr, ev, names = onset_rows()
    sets = {"feb": [r for r in ev if r["day"] >= HOLD]}
    for src in ("feb", "mar"):
        rows, nm = pickle.load(open(OUT / f"t2_pseudo_{src}.pkl", "rb"))
        assert nm == names
        sets[f"{src}_ps"] = rows["onset"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    a = ap.parse_args()
    for var in a.variants:
        drop = VARIANTS[var]
        mem_ok = lambda n: not n.startswith(MEMORY) or ("+recent" in drop and not n.startswith(LONG))
        keep = [i for i, n in enumerate(names) if mem_ok(n) and n not in drop]
        nm = [names[i] for i in keep]
        sub = lambda rs: [dict(r, X=r["X"][:, keep]) for r in rs]
        X, y, w = stack(sub([r for r in tr if r["day"] < HOLD]), COND, (6,), nm, seed=1)
        Xv, yv, wv = stack(sub([r for r in tr if r["day"] >= HOLD]), COND, (6,), nm, seed=2)
        dt = lgb.Dataset(X, y, weight=w, feature_name=nm + ["h"])
        m = lgb.train(PARAMS, dt, 4000, valid_sets=[lgb.Dataset(Xv, yv, weight=wv, reference=dt)],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        m.save_model(str(OUT / f"t2abl_{var}_onset.txt"))
        line = []
        for s, rs in sets.items():
            P = [predict_window(m, r["X"][:, keep], COND) for r in rs]
            best = max(((np.mean([iou(decode(p, COND, thr, 3), r["y"], r["elig"]) for p, r in zip(P, rs)]), thr)
                        for thr in (0.2, 0.3, 0.45)))
            at45 = np.mean([iou(decode(p, COND, 0.45, 3), r["y"], r["elig"]) for p, r in zip(P, rs)])
            line.append(f"{s} {best[0]:.4f}@{best[1]} (0.45: {at45:.4f})")
        print(f"{var:16s} it {m.best_iteration:5d} | " + " | ".join(line), flush=True)


if __name__ == "__main__":
    main()
