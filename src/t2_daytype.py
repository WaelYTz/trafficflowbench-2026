"""Recent memory by day type: the per-hour memory columns (recent_q, recent_n, recent_onset and their
shifts) are recomputed from the last N days of the same type (weekday vs weekend) instead of the last 7
calendar days. Weekday and weekend demand differ, so same-type memory may point at the right bottleneck
more sharply. Scored like t2_onset_recent.py / t2_ongoing_recent.py.
"""
from __future__ import annotations

import argparse
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import OUT, days
from t2_data import CAL_SPLITS, Recent, Static
from t2_model import HOLD, MEMORY, PARAMS, aggregate, decode, iou, predict_window, stack
from t2_onset_abl import LONG, onset_rows
from t2_onset_recent import recent_cols, refresh

SUB = dict(window_stride=3, n_h=3, far_keep=0.15)


def refresh_daytype(rows_by_split, names, n_same: int):
    """Overwrite the recent-memory columns with same-day-type memory over the last n_same such days."""
    col = {n: i for i, n in enumerate(names)}
    wts = np.array([0.5, 1.0, 0.5])
    for panel in sorted({r["panel"] for rs in rows_by_split.values() for r in rs}):
        rec = Recent(Static(panel))
        dq, do, don = (np.diff(a, axis=0) for a in (rec.cq, rec.co, rec.con))
        cal = [d for s in CAL_SPLITS for d in days(panel, s)]
        wk = (pd.to_datetime(pd.Series(cal)).dt.dayofweek.to_numpy() >= 5)
        for split, rs in rows_by_split.items():
            di = {d: i for i, d in enumerate(days(panel, split))}
            for r in rs:
                if r["panel"] != panel:
                    continue
                c = rec.offset[split] + di[r["day"]]
                prev = np.flatnonzero(wk[:c] == wk[c])[-n_same:]
                h = (r["slot"] + 6) // 12 % 24
                hrs = [(h - 1) % 24, h, (h + 1) % 24]
                q = dq[prev][:, :, hrs].sum(axis=0) @ wts
                n = do[prev][:, :, hrs].sum(axis=0) @ wts
                on = don[prev][:, :, hrs].sum(axis=0) @ wts
                base = rec.features(split, di[r["day"]], r["slot"])  # today_q_before stays as is
                base.update(recent_q=q / np.maximum(n, 1), recent_n=n, recent_onset=on)
                for k, a in recent_cols(base).items():
                    r["X"][:, col[k]] = a


def evaluate(m, keep, sets, PB, cond, thr, fb):
    out = {}
    for s, rs in sets.items():
        P = [0.7 * predict_window(m, r["X"][:, keep], cond) + 0.3 * pb for r, pb in zip(rs, PB[s])]
        out[s] = aggregate([(r["panel"], cond, iou(decode(p, cond, thr, fb), r["y"], r["elig"])) for p, r in zip(P, rs)])["overall"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="onset", choices=("onset", "ongoing"))
    ap.add_argument("--n-same", nargs="+", type=int, default=[5, 8])
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
    groups = {"train": tr + feb + ps["feb"] + ps["feb_off"], "validation": ps["mar"] + ps["mar_off"]}
    sets = {"feb": feb, **{("feb_ps" if k == "feb" else k): v for k, v in ps.items()}}
    kb = [i for i, n in enumerate(names) if not n.startswith(MEMORY)]
    keep = [i for i, n in enumerate(names) if not n.startswith(LONG)]
    nm = [names[i] for i in keep]
    base = lgb.Booster(model_file=str(OUT / ("t2abl_base_onset.txt" if a.key == "onset" else "t2og_base_ongoing.txt")))
    PB = {s: [predict_window(base, r["X"][:, kb], cond) for r in rs] for s, rs in sets.items()}
    for var in ["box7"] + [f"same{n}" for n in a.n_same]:
        if var == "box7":
            refresh(groups, names, 7)
        else:
            refresh_daytype(groups, names, int(var[4:]))
        sub = lambda rs: [dict(r, X=r["X"][:, keep]) for r in rs]
        X, y, w = stack(sub([r for r in tr if r["day"] < HOLD]), cond, hz, nm, seed=1, **kw)
        Xv, yv, wv = stack(sub([r for r in tr if r["day"] >= HOLD]), cond, hz, nm, seed=2, **kw)
        dt = lgb.Dataset(X, y, weight=w, feature_name=nm + ["h"])
        m = lgb.train(PARAMS, dt, 6000, valid_sets=[lgb.Dataset(Xv, yv, weight=wv, reference=dt)],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        del X, y, w, Xv, yv, wv
        m.save_model(str(OUT / f"t2dt_{var}_{a.key}.txt"))
        res = evaluate(m, keep, sets, PB, cond, thr, fb)
        print(f"{a.key} {var:7s} it {m.best_iteration:5d} | blend 0.7/0.3 | " + " | ".join(f"{s} {v:.4f}" for s, v in res.items()), flush=True)


if __name__ == "__main__":
    main()
