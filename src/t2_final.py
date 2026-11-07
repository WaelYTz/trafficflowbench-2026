"""Refit the Task 2 LightGBM models on every train month (Feb included) and write the scored queue.

Round counts are the Feb-early-stopped ones from t2<tag>_<key>.txt plus 10% for the extra month;
thresholds come from t2<tag>_report.json. Feb cannot score these models, so they are judged by the
holdout version and the leaderboard.
"""
from __future__ import annotations

import argparse
import json
import pickle

import lightgbm as lgb
import numpy as np

from common import OUT, QUEUE_PANELS
from t2_data import scored_windows
from t2_model import MEMORY, PARAMS, decode, predict_window, stack

CONDS = (("ongoing", "queue_ongoing", (1, 2, 3, 4, 5, 6)), ("onset", "queue_onset", (6,)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--rows-tag", default=None)
    ap.add_argument("--drop-mem", action="store_true")
    a = ap.parse_args()
    tr, _, names = pickle.load(open(OUT / f"t2_rows{a.tag if a.rows_tag is None else a.rows_tag}.pkl", "rb"))
    keep = [i for i, n in enumerate(names) if not (a.drop_mem and n.startswith(MEMORY))]
    if a.drop_mem:
        for k in tr:
            for r in tr[k]:
                r["X"] = r["X"][:, keep]
    all_names, names = names, [names[i] for i in keep]
    best = json.loads((OUT / f"t2{a.tag}_report.json").read_text())["best"]
    feat = names + ["h"]
    models = {}
    for key, cond, hz in CONDS:
        n = lgb.Booster(model_file=str(OUT / f"t2{a.tag}_{key}.txt")).current_iteration()
        kw = dict(window_stride=3, n_h=3, far_keep=0.15) if key == "ongoing" else {}
        X, y, w = stack(tr[key], cond, hz, names, seed=3, **kw)
        print(f"{key}: {len(y):,} rows, {int(n * 1.1)} rounds", flush=True)
        m = lgb.train(PARAMS, lgb.Dataset(X, y, weight=w, feature_name=feat), int(n * 1.1))
        m.save_model(str(OUT / f"t2{a.tag}_{key}_final.txt"))
        models[key] = m
        del X, y, w
    queue = {}
    for p in QUEUE_PANELS:
        for split in ("validation", "private"):
            blocks, nm = scored_windows(p, split)
            assert nm == all_names
            for b in blocks:
                b["X"] = b["X"][:, keep]
                key = "onset" if b["condition"] == "queue_onset" else "ongoing"
                thr, fb = best[key][-2:]
                pred = decode(predict_window(models[key], b["X"], b["condition"]), b["condition"], thr, fb)
                out = np.zeros_like(pred, dtype=np.int8)
                out[:, b["order"]] = pred
                queue[b["window_id"]] = out
    pickle.dump(queue, open(OUT / f"t2{a.tag}_queue_final.pkl", "wb"))
    print("scored windows:", len(queue), "positive share", float(np.mean([q.mean() for q in queue.values()])), flush=True)


if __name__ == "__main__":
    main()
