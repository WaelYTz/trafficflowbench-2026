"""Onset training augmentation: add origins whose first scored queue appears at T+25 (one step earlier
than the official T+30), labelled at T+30, next to the regular onset windows. Same recent-7-day model;
scored with the official aggregation on Feb official-like and Feb / March pseudo-windows."""
from __future__ import annotations

import pickle
from concurrent.futures import ProcessPoolExecutor

import lightgbm as lgb
import numpy as np
import pandas as pd

import t2_data
from common import OUT, QUEUE_PANELS, T, days
from t2_data import H, Recent, Static, train_arrays, window_features
from t2_model import HOLD, MEMORY, PARAMS, aggregate, decode, iou, predict_window, stack
from t2_onset_abl import LONG, onset_rows
from t2_onset_recent import refresh

COND = "queue_onset"


def early_rows(panel: str):
    t2_data.RECENT_DAYS = 7
    st = Static(panel)
    v, q, e, vm, qm = train_arrays(st)
    dl = days(panel, "train")
    dow = pd.to_datetime(pd.Series(dl)).dt.dayofweek.to_numpy()
    zeros = np.zeros((st.L, 24))
    rec = Recent(st)
    qobs = np.isfinite(v) & (v <= st.cut[None, None, :])
    qtrue = qobs & e
    out = []
    for d in range(v.shape[0]):
        cov = np.convolve(np.isfinite(v[d]).mean(axis=1), np.ones(H) / H, mode="full")[H - 1:T]
        for t in range(H, T - 6):
            if cov[t - H] < 0.7 or qobs[d, t - H:t].any():
                continue
            hor = qtrue[d, t + 1:t + 7]
            if not (hor[4].any() and not hor[:4].any()):
                continue
            V = np.vstack([v[d, t - H:t], vm[d, t][None]]); Q = np.vstack([q[d, t - H:t], qm[d, t][None]])
            X, names = window_features(st, V, Q, t, dow[d], zeros, zeros, rec.features("train", d, t))
            lab = np.isfinite(v[d, t + 1:t + 7]) & (v[d, t + 1:t + 7] <= st.cut[None, :])
            el = e[d, t + 1:t + 7] & np.isfinite(v[d, t + 1:t + 7])
            out.append(dict(X=X, y=lab, elig=el, day=dl[d], slot=t, panel=panel))
    return out


def main():
    tr, ev, names = onset_rows()
    feb = [r for r in ev if r["day"] >= HOLD]
    ps = {s: pickle.load(open(OUT / f"t2_pseudo_{s}.pkl", "rb"))[0]["onset"] for s in ("feb", "mar", "feb_off", "mar_off")}
    refresh({"train": tr + feb + ps["feb"] + ps["feb_off"], "validation": ps["mar"] + ps["mar_off"]}, names, 7)
    with ProcessPoolExecutor(4) as ex:
        extra = [r for part in ex.map(early_rows, QUEUE_PANELS) for r in part]
    print("regular onset windows", len(tr), "| early (T+25) windows", len(extra), flush=True)
    keep = [i for i, n in enumerate(names) if not n.startswith(LONG)]
    kb = [i for i, n in enumerate(names) if not n.startswith(MEMORY)]
    nm = [names[i] for i in keep]
    base = lgb.Booster(model_file=str(OUT / "t2abl_base_onset.txt"))
    sets = {"feb": feb, **{("feb_ps" if k == "feb" else k): v for k, v in ps.items()}}
    PB = {s: [predict_window(base, r["X"][:, kb], COND) for r in rs] for s, rs in sets.items()}
    sub = lambda rs: [dict(r, X=r["X"][:, keep]) for r in rs]
    Xv, yv, wv = stack(sub([r for r in tr if r["day"] >= HOLD]), COND, (6,), nm, seed=2)
    for var, w_extra in (("rec7", 0.0), ("rec7_aug0.5", 0.5), ("rec7_aug1", 1.0)):
        X, y, w = stack(sub([r for r in tr if r["day"] < HOLD]), COND, (6,), nm, seed=1)
        if w_extra:
            Xe, ye, we = stack(sub([r for r in extra if r["day"] < HOLD]), COND, (6,), nm, seed=4)
            X, y, w = np.vstack([X, Xe]), np.concatenate([y, ye]), np.concatenate([w, we * w_extra])
        dt = lgb.Dataset(X, y, weight=w, feature_name=nm + ["h"])
        m = lgb.train(PARAMS, dt, 4000, valid_sets=[lgb.Dataset(Xv, yv, weight=wv, reference=dt)],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        m.save_model(str(OUT / f"t2aug_{var}_onset.txt"))
        line = []
        for s, rs in sets.items():
            P = [0.7 * predict_window(m, r["X"][:, keep], COND) + 0.3 * pb for r, pb in zip(rs, PB[s])]
            line.append(f"{s} " + aggregate([(r["panel"], COND, iou(decode(p, COND, 0.3, 3), r["y"], r["elig"])) for p, r in zip(P, rs)])["overall"].__format__(".4f"))
        print(f"{var:12s} it {m.best_iteration:4d} | blend thr .3 | " + " | ".join(line), flush=True)


if __name__ == "__main__":
    main()
