"""Onset models whose place knowledge comes from recent memory (masked layer, last N days) instead of
long-term train priors, with N in 7/14/28, alone and blended with the memory-free base model.

The recent-memory columns of the cached rows are recomputed for each N, so no row rebuild is needed.
Scored on Feb official-like windows, Feb pseudo and March pseudo windows (t2_pseudo.py).
"""
from __future__ import annotations

import argparse
import pickle

import lightgbm as lgb
import numpy as np

import t2_data
from common import OUT, days
from t2_data import Recent, Static, _shift
from t2_model import HOLD, MEMORY, PARAMS, decode, iou, predict_window, stack
from t2_onset_abl import LONG, onset_rows

COND = "queue_onset"
SHIFTED = ("recent_q", "recent_onset", "today_q_before")


def recent_cols(extra: dict) -> dict:
    out = {}
    for k, a in extra.items():
        out[k] = a
        if k in SHIFTED:
            for s in (-2, -1, 1, 2):
                out[f"{k}_{s:+d}"] = _shift(a, s)
    return out


def refresh(rows_by_split, names, n_days: int):
    """Overwrite the recent-memory columns for a window of n_days."""
    t2_data.RECENT_DAYS = n_days
    col = {n: i for i, n in enumerate(names)}
    for panel in sorted({r["panel"] for rs in rows_by_split.values() for r in rs}):
        rec = Recent(Static(panel))
        for split, rs in rows_by_split.items():
            di = {d: i for i, d in enumerate(days(panel, split))}
            for r in rs:
                if r["panel"] != panel:
                    continue
                for k, a in recent_cols(rec.features(split, di[r["day"]], r["slot"])).items():
                    r["X"][:, col[k]] = a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="+", type=int, default=[7, 14, 28])
    a = ap.parse_args()
    tr, ev, names = onset_rows()
    feb = [r for r in ev if r["day"] >= HOLD]
    ps = {src: pickle.load(open(OUT / f"t2_pseudo_{src}.pkl", "rb"))[0]["onset"] for src in ("feb", "mar")}
    sets = {"feb": feb, "feb_ps": ps["feb"], "mar_ps": ps["mar"]}
    base = lgb.Booster(model_file=str(OUT / "t2abl_base_onset.txt"))
    kb = [i for i, n in enumerate(names) if not n.startswith(MEMORY)]
    PB = {s: [predict_window(base, r["X"][:, kb], COND) for r in rs] for s, rs in sets.items()}
    keep = [i for i, n in enumerate(names) if not n.startswith(LONG)]
    nm = [names[i] for i in keep]
    for nd in a.days:
        refresh({"train": tr + feb + ps["feb"], "validation": ps["mar"]}, names, nd)
        sub = lambda rs: [dict(r, X=r["X"][:, keep]) for r in rs]
        X, y, w = stack(sub([r for r in tr if r["day"] < HOLD]), COND, (6,), nm, seed=1)
        Xv, yv, wv = stack(sub([r for r in tr if r["day"] >= HOLD]), COND, (6,), nm, seed=2)
        dt = lgb.Dataset(X, y, weight=w, feature_name=nm + ["h"])
        m = lgb.train(PARAMS, dt, 4000, valid_sets=[lgb.Dataset(Xv, yv, weight=wv, reference=dt)],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        m.save_model(str(OUT / f"t2rec{nd}_onset.txt"))
        for wb in (0.0, 0.3, 0.5):
            line = []
            for s, rs in sets.items():
                P = [wb * pb + (1 - wb) * predict_window(m, r["X"][:, keep], COND) for pb, r in zip(PB[s], rs)]
                res = {thr: np.mean([iou(decode(p, COND, thr, 3), r["y"], r["elig"]) for p, r in zip(P, rs)])
                       for thr in (0.2, 0.3, 0.45)}
                line.append(f"{s} " + " ".join(f"{res[t]:.4f}" for t in (0.2, 0.3, 0.45)))
            print(f"recent{nd:2d} w_base {wb:.1f} it {m.best_iteration:4d} | thr .2/.3/.45 | " + " | ".join(line), flush=True)


if __name__ == "__main__":
    main()
