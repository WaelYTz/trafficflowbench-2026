"""Physics-aware flow models, judged on every Feb target cell by the Task 1 rule and local Task 3.

S_LWR error at a free-flow target cell is about L * |q_err| / v, so long links dominate it while
the Task 1 flow RMSE weights every cell alike. Flow variants differ in sample weights and loss;
the speed model is shared. Writes out/t1_phys.csv and the models t1p_<variant>.json.
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import xgboost as xgb

from common import FAMILY, OUT, PANELS
from t1_train import bases, fit, panel_rows, PARAMS
from t3_study import predict_month, score_variant

VARIANTS = {
    "mse": dict(obj="reg:squarederror", w_len=False),
    "mse_len": dict(obj="reg:squarederror", w_len=True),
    "l1_len": dict(obj="reg:absoluteerror", w_len=True),
    "huber_len": dict(obj="reg:pseudohubererror", w_len=True, huber_slope=30.0),
}


def fit_variant(X, y, w, Xv, yv, rounds, device, obj, huber_slope=None):
    p = dict(PARAMS, device=device, objective=obj)
    if huber_slope:
        p["huber_slope"] = huber_slope
    dtr = xgb.DMatrix(X, label=y, weight=w)
    dva = xgb.DMatrix(Xv, label=yv)
    m = xgb.train(p, dtr, num_boost_round=rounds, evals=[(dva, "hold")], early_stopping_rounds=200, verbose_eval=1000)
    return m[: m.best_iteration + 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rounds", type=int, default=10000)
    ap.add_argument("--cap-train", type=int, default=1500)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--panels", nargs="*", default=PANELS)
    a = ap.parse_args()
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        parts = list(ex.map(panel_rows, [(p, a.cap_train, 8000, False) for p in a.panels]))
    names = parts[0]["names"]
    R = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "names"}
    del parts
    bv, bq = bases(R["X"], names)
    tv = R["yv"] / R["vf"] - bv
    tq = (R["yq"] - bq) / R["lanes"]
    w = np.zeros(len(tv))
    for p in np.unique(R["panel"]):
        m = R["panel"] == p
        w[m] = 1.0 / m.sum()
    h = R["hold"]; tr = ~h
    print(f"rows {len(tv):,} {time.time() - t0:.0f}s", flush=True)
    mv = fit(R["X"][tr], tv[tr], (w * len(w) / w.sum())[tr], R["X"][h], tv[h], a.rounds, a.device)
    mv = mv[: mv.best_iteration + 1]
    mv.save_model(str(OUT / "t1p_v.json"))
    print(f"v: {mv.num_boosted_rounds()} rounds {time.time() - t0:.0f}s", flush=True)
    qm = {}
    for name in a.variants:
        cfg = VARIANTS[name]
        # Per-lane residual target: a cell's total flow error is lanes * residual, its N error ~ L * that.
        wq = w * (R["len"] * R["lanes"] if cfg["w_len"] else 1.0)
        wq = wq * len(wq) / wq.sum()
        qm[name] = fit_variant(R["X"][tr], tq[tr], wq[tr], R["X"][h], tq[h], a.rounds, a.device, cfg["obj"],
                               cfg.get("huber_slope"))
        qm[name].save_model(str(OUT / f"t1p_q_{name}.json"))
        print(f"q {name}: {qm[name].num_boosted_rounds()} rounds {time.time() - t0:.0f}s", flush=True)
    del R
    rows = []
    for p in a.panels:
        s = predict_month(p, {"v": mv, "q": qm[a.variants[0]]}, keep_X=True)
        Xf = s.pop("X")
        dm = xgb.DMatrix(Xf)
        net_l = s["lanes_t"]
        for name, m in qm.items():
            pq = np.clip(s["bq"] + m.predict(dm) * net_l, 60, None)
            r = dict(panel=p, variant=name, **score_variant(p, s, s["pv"], pq))
            print(r, flush=True)
            rows.append(r)
    df = pd.DataFrame(rows)
    df["family"] = df.panel.map(FAMILY)
    summ = df.groupby(["variant", "family"])[["S_state", "S_physics", "S_LWR", "S_FD"]].mean().groupby("variant").mean()
    summ["total_part"] = 0.35 * summ.S_state + 0.15 * summ.S_physics
    print(summ.round(4).to_string(), flush=True)
    df.to_csv(OUT / "t1_phys.csv", index=False)
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
