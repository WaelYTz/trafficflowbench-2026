"""Feb holdout study: Task 1 predictions of the Jun-Jan models (t1h_*.json) at every Feb target cell,
scored with the official Task 1 rule and the local Task 3 evaluator.

Predictions are cached in out/t3s_<panel>.npz so post-processing ideas can be scored quickly.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from common import FAMILY, OUT, PANELS, days, load
from t1_features import Grid, build, profiles
from t1_train import bases
import t3_eval as E

MONTH = "2031-02"


def predict_month(panel: str, models: dict, keep_X: bool = False) -> dict:
    dl = days(panel, "train")
    idx = np.array([i for i, d in enumerate(dl) if d.startswith(MONTH)])
    g0 = Grid(panel, "train", d0=0, d1=1)
    prof = profiles(panel, g0.order)
    g = Grid(panel, "train", d0=int(idx[0]), d1=int(idx[-1]) + 1)
    d, t, l = np.nonzero(g.target)
    X, nm = build(g, d, t, l, prof, loo=True)
    bv, bq = bases(X, nm)
    dm = xgb.DMatrix(X)
    pv = np.clip((bv + models["v"].predict(dm)) * g.vf[l], 3, 140)
    pq = np.clip(bq + models["q"].predict(dm) * g.lanes[l], 60, None)
    sc = g.order[l]
    out = dict(idx=idx, d=d.astype(np.int16), t=t.astype(np.int16), l=sc.astype(np.int16), pv=pv.astype(np.float32), pq=pq.astype(np.float32),
               bv=(bv * g.vf[l]).astype(np.float32), bq=bq.astype(np.float32), regime=g.regime[d])
    np.savez_compressed(OUT / f"t3s_{panel}.npz", **out)
    if keep_X:
        out.update(X=X, lanes_t=g.lanes[l])
    return out


def fill(panel: str, s: dict, pv: np.ndarray, pq: np.ndarray):
    v = load(panel, "train", "v")[s["idx"]]
    q = load(panel, "train", "q")[s["idx"]]
    V, Q = v.copy(), q.copy()
    V[s["d"], s["t"], s["l"]] = pv
    Q[s["d"], s["t"], s["l"]] = pq
    return v, q, V, Q


def t1_score(panel: str, s: dict, pv, pq) -> float:
    from common import network
    v = load(panel, "train", "v")[s["idx"]][s["d"], s["t"], s["l"]]
    q = load(panel, "train", "q")[s["idx"]][s["d"], s["t"], s["l"]]
    lanes = network(panel).lanes.to_numpy(float)[s["l"]]
    out = []
    for r in range(3):
        m = s["regime"] == r
        if not m.any():
            continue
        rv = np.sqrt(np.mean((pv[m] - v[m]) ** 2)); rq = np.sqrt(np.mean(((pq[m] - q[m]) / lanes[m]) ** 2))
        out.append(0.54 * max(0, 1 - rv / 25) + 0.46 * max(0, 1 - rq / 600))
    return float(np.mean(out))


def score_variant(panel: str, s: dict, pv, pq) -> dict:
    v, q, V, Q = fill(panel, s, pv, pq)
    true = E.state_frame(panel, v, q, s["idx"])
    sub = E.state_frame(panel, V, Q, s["idx"])
    ph = E.evaluate(panel, sub, true)
    return dict(S_state=t1_score(panel, s, pv, pq), S_physics=float(np.mean([r["S_physics"] for r in ph])),
                S_LWR=float(np.mean([r["S_LWR"] for r in ph])), S_FD=float(np.mean([r["S_FD"] for r in ph])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="*", default=PANELS)
    a = ap.parse_args()
    models = {k: xgb.Booster(model_file=str(OUT / f"t1h_{k}.json")) for k in ("v", "q")}
    rows = []
    for p in a.panels:
        f = OUT / f"t3s_{p}.npz"
        s = dict(np.load(f)) if f.exists() else predict_month(p, models)
        for name, pv, pq in (("model", s["pv"], s["pq"]), ("base", s["bv"], s["bq"])):
            r = dict(panel=p, variant=name, **score_variant(p, s, pv, pq))
            print(r, flush=True)
            rows.append(r)
    df = pd.DataFrame(rows)
    df["family"] = df.panel.map(FAMILY)
    summ = df.groupby(["variant", "family"])[["S_state", "S_physics", "S_LWR", "S_FD"]].mean().groupby("variant").mean()
    summ["total_part"] = 0.35 * summ.S_state + 0.15 * summ.S_physics
    print(summ.round(4).to_string(), flush=True)
    df.to_csv(OUT / "t3_study.csv", index=False)


if __name__ == "__main__":
    main()
