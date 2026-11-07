"""Diagnose the Task 2 local-vs-leaderboard gap on the only official train windows (June 2030).

A model is fitted without June (Jul-Jan, early-stopped on Feb), then June is scored two ways:
  A  our selector's June windows, features from the training path (build_train)
  B  the official June windows, features from the submission path (scored_windows logic, with the
     row at T from the masked layer and June left out of the priors)
Shared windows are compared feature by feature to catch differences between the two paths.
"""
from __future__ import annotations

import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import OUT, QUEUE_PANELS, REL, days, network
from t2_data import H, Recent, Static, priors, train_arrays, window_features
from t2_model import PARAMS, aggregate, decode, iou, predict_window, stack

JUNE = "2030-07"  # days before this are June


def official_june(panel: str):
    st = Static(panel)
    o = st.order
    wi = pd.read_csv(REL / "task2" / panel / "train" / "window_index.csv")
    hist = pd.read_parquet(REL / "task2" / panel / "train" / "window_history.parquet",
                           columns=["window_id", "timestamp", "link_id", "speed_kmh", "flow_vph"])
    li = {l: i for i, l in enumerate(network(panel).link_id)}
    rank = np.empty(st.L, int); rank[o] = np.arange(st.L)
    dl = days(panel, "train"); di = {d: i for i, d in enumerate(dl)}
    v, q, e, vm, qm = train_arrays(st)
    mcode = pd.factorize(pd.Index([x[:7] for x in dl]), sort=True)[0]
    onset_c, queued_c, seen_c = priors(st, v, e, mcode)
    op = (onset_c.sum(0) - onset_c[0]) / (len(mcode) - (mcode == 0).sum())
    qp = (queued_c.sum(0) - queued_c[0]) / np.maximum(seen_c.sum(0) - seen_c[0], 1)
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
        d = di[origin[:10]]
        V[H] = vm[d, slot]; Q[H] = qm[d, slot]
        X, names = window_features(st, V, Q, slot, pd.Timestamp(origin[:10]).dayofweek, op, qp, rec.features("train", d, slot))
        lab = np.isfinite(v[d, slot + 1:slot + 7]) & (v[d, slot + 1:slot + 7] <= st.cut[None, :])
        el = e[d, slot + 1:slot + 7] & np.isfinite(v[d, slot + 1:slot + 7])
        # the same window built the training way, for a feature diff
        Vt = np.vstack([v[d, slot - H:slot], vm[d, slot][None]]); Qt = np.vstack([q[d, slot - H:slot], qm[d, slot][None]])
        Xt, _ = window_features(st, Vt, Qt, slot, pd.Timestamp(origin[:10]).dayofweek, op, qp, rec.features("train", d, slot))
        out.append(dict(X=X, Xt=Xt, y=lab, elig=el, panel=panel, cond=w.condition, day=origin[:10], slot=slot,
                        hist_nan=float(np.isnan(V[:H]).mean()), hist_nan_t=float(np.isnan(Vt[:H]).mean())))
    return out, names


def main():
    tr, ev, names = pickle.load(open(OUT / "t2_rowse.pkl", "rb"))
    models = {}
    for cond, key, hz in (("queue_ongoing", "ongoing", (1, 2, 3, 4, 5, 6)), ("queue_onset", "onset", (6,))):
        kw = dict(window_stride=3, n_h=3, far_keep=0.15) if key == "ongoing" else {}
        Xt, yt, wt = stack([r for r in tr[key] if JUNE <= r["day"] < "2031-02"], cond, hz, names, seed=1, **kw)
        Xv, yv, wv = stack([r for r in tr[key] if r["day"] >= "2031-02"], cond, hz, names, seed=2, **kw)
        dt = lgb.Dataset(Xt, yt, weight=wt, feature_name=names + ["h"])
        models[key] = lgb.train(PARAMS, dt, 5000, valid_sets=[lgb.Dataset(Xv, yv, weight=wv, reference=dt)],
                                callbacks=[lgb.early_stopping(150), lgb.log_evaluation(1000)])
    thr = {"ongoing": (0.45, 1), "onset": (0.45, 3)}
    # A: our selector's June windows, training path
    sa = []
    for key, cond in (("ongoing", "queue_ongoing"), ("onset", "queue_onset")):
        for r in ev[key]:
            if r["day"] < JUNE:
                sa.append((r["panel"], cond, iou(decode(predict_window(models[key], r["X"], cond), cond, *thr[key]), r["y"], r["elig"])))
    print("A  our June windows (training path):", {k: round(v, 3) for k, v in aggregate(sa).items()},
          "n", len(sa), flush=True)
    # B: official June windows, submission path (and the same windows built the training way)
    sb, st_ = [], []
    diffs = []
    for p in QUEUE_PANELS:
        wins, nm = official_june(p)
        assert nm == names
        for w in wins:
            key = "onset" if w["cond"] == "queue_onset" else "ongoing"
            sb.append((p, w["cond"], iou(decode(predict_window(models[key], w["X"], w["cond"]), w["cond"], *thr[key]), w["y"], w["elig"])))
            st_.append((p, w["cond"], iou(decode(predict_window(models[key], w["Xt"], w["cond"]), w["cond"], *thr[key]), w["y"], w["elig"])))
            dif = ~np.isclose(w["X"], w["Xt"], equal_nan=True)
            diffs.append(dif.any(0))
            if len(diffs) <= 3:
                print(w["day"], w["slot"], w["cond"], "hist NaN share sub/train path", round(w["hist_nan"], 3), round(w["hist_nan_t"], 3))
    print("B  official June windows (submission path):", {k: round(v, 3) for k, v in aggregate(sb).items()}, "n", len(sb), flush=True)
    print("B' official June windows (training path):  ", {k: round(v, 3) for k, v in aggregate(st_).items()}, flush=True)
    share = np.mean(diffs, 0)
    print("features differing between paths (share of windows):",
          {names[i]: round(float(share[i]), 2) for i in np.argsort(-share)[:15] if share[i] > 0})
    on = [x for x in sb if x[1] == "queue_onset"]
    print("official onset IoU distribution:", np.round(np.percentile([x[2] for x in on], [10, 25, 50, 75, 90]), 2))


if __name__ == "__main__":
    main()
