"""Task 1 ensemble test: XGBoost (the submitted model) and CatBoost (GPU) on the same rows and targets,
trained on Jun-Jan and scored on the Feb holdout with the official rule, alone and blended."""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import xgboost as xgb
from catboost import CatBoostRegressor

from common import OUT, PANELS
from t1_train import bases, fit, panel_rows, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-train", type=int, default=4000)
    ap.add_argument("--cap-hold", type=int, default=8000)
    ap.add_argument("--rounds", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        parts = list(ex.map(panel_rows, [(p, a.cap_train, a.cap_hold, False) for p in PANELS]))
    names = parts[0]["names"]
    R = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "names"}
    del parts
    bv, bq = bases(R["X"], names)
    tv, tq = R["yv"] / R["vf"] - bv, (R["yq"] - bq) / R["lanes"]
    w = np.zeros(len(tv))
    for p in np.unique(R["panel"]):
        m = R["panel"] == p
        w[m] = 1.0 / m.sum()
    w *= len(w) / w.sum()
    h = R["hold"]; tr = ~h
    print(f"rows {len(tv):,} ({h.sum():,} holdout), {time.time() - t0:.0f}s", flush=True)
    pred = {}
    for key, y in (("v", tv), ("q", tq)):
        m = fit(R["X"][tr], y[tr], w[tr], R["X"][h], y[h], a.rounds, "cuda")
        pred["xgb", key] = m.predict(xgb.DMatrix(R["X"][h]), iteration_range=(0, m.best_iteration + 1))
        print(f"xgb {key}: best {m.best_iteration + 1}, {time.time() - t0:.0f}s", flush=True)
        c = CatBoostRegressor(iterations=a.rounds, depth=8, learning_rate=0.08, l2_leaf_reg=5, loss_function="RMSE",
                              task_type="GPU", random_seed=2026, border_count=254, early_stopping_rounds=300, verbose=2000)
        c.fit(R["X"][tr], y[tr], sample_weight=w[tr], eval_set=(R["X"][h], y[h]), use_best_model=True)
        pred["cat", key] = c.predict(R["X"][h])
        c.save_model(str(OUT / f"t1h_cat_{key}.cbm"))
        print(f"cat {key}: best {c.get_best_iteration()}, {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(OUT / "t1_cat_holdout_preds.npz", **{f"{a_}_{k}": v for (a_, k), v in pred.items()})
    res = {}
    for wx in (1.0, 0.8, 0.7, 0.6, 0.5, 0.0):
        pv = (bv[h] + wx * pred["xgb", "v"] + (1 - wx) * pred["cat", "v"]) * R["vf"][h]
        pq = bq[h] + (wx * pred["xgb", "q"] + (1 - wx) * pred["cat", "q"]) * R["lanes"][h]
        s, _ = score(pv, pq, R["yv"][h], R["yq"][h], R["lanes"][h], R["panel"][h], R["regime"][h])
        rv = float(np.sqrt(np.mean((pv - R["yv"][h]) ** 2))); rq = float(np.sqrt(np.mean(((pq - R["yq"][h]) / R["lanes"][h]) ** 2)))
        res[wx] = dict(S_state=s, rmse_v=rv, rmse_q_lane=rq)
        print(f"w_xgb {wx}: S_state {s:.5f} rmse_v {rv:.3f} rmse_q {rq:.2f}", flush=True)
    (OUT / "t1_cat_report.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
