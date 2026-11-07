"""XGBoost (GPU) versions of the Task 2 models, trained on the same rows and splits as t2_model.py,
for blending with the LightGBM probabilities in t2_decode.py."""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import xgboost as xgb

from common import OUT
from t2_model import HOLD, stack

PARAMS = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda", max_depth=8,
              learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, min_child_weight=20, reg_lambda=5.0,
              max_bin=256, seed=11)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--rows-tag", default=None, help="read t2_rows<rows-tag>.pkl")
    ap.add_argument("--rounds", type=int, default=8000)
    a = ap.parse_args()
    tr, _, names = pickle.load(open(OUT / f"t2_rows{a.tag if a.rows_tag is None else a.rows_tag}.pkl", "rb"))
    feat = names + ["h"]
    for cond, key, hz in (("queue_ongoing", "ongoing", (1, 2, 3, 4, 5, 6)), ("queue_onset", "onset", (6,))):
        kw = dict(window_stride=3, n_h=3, far_keep=0.15) if key == "ongoing" else {}
        Xt, yt, wt = stack([r for r in tr[key] if r["day"] < HOLD], cond, hz, names, seed=1, **kw)
        Xv, yv, wv = stack([r for r in tr[key] if r["day"] >= HOLD], cond, hz, names, seed=2, **kw)
        dt = xgb.QuantileDMatrix(Xt, yt, weight=wt, feature_names=feat)
        dv = xgb.QuantileDMatrix(Xv, yv, weight=wv, feature_names=feat, ref=dt)
        del Xt, Xv
        m = xgb.train(PARAMS, dt, a.rounds, evals=[(dv, "feb")], early_stopping_rounds=200, verbose_eval=500)
        print(f"{key}: best iteration {m.best_iteration + 1}", flush=True)
        m = m[: m.best_iteration + 1]
        m.save_model(str(OUT / f"t2{a.tag}_{key}_xgb.json"))


if __name__ == "__main__":
    main()
