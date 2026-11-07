"""Final Task 2 queue with recent-memory models.

Onset: 0.7 x recent-memory model (last N days of the masked layer, no long-term train priors) + 0.3 x the
memory-free model, decoded at thr 0.3 / top-3 fallback. Ongoing: the memory-free model, or with --ongoing
the same 0.7 / 0.3 blend at thr 0.45. Every input of a scored window is stamped <= T: recent memory reads
earlier days and the same day before T-12.
"""
from __future__ import annotations

import argparse
import json
import pickle

import lightgbm as lgb
import numpy as np

import t2_data
from common import OUT, QUEUE_PANELS
from t2_data import scored_windows
from t2_model import MEMORY, PARAMS, decode, predict_window, stack
from t2_onset_abl import LONG, onset_rows
from t2_onset_recent import refresh

COND = "queue_onset"
SUB = dict(window_stride=3, n_h=3, far_keep=0.15)  # ongoing subsampling, as in t2_final.py


def fit(rows, names, keep, cond, hz, holdout_model, out, **kw):
    """Refit on every train month with the holdout model's rounds + 10%, or load it if already saved."""
    if out.exists():
        return lgb.Booster(model_file=str(out))
    nm = [names[i] for i in keep]
    n = int(lgb.Booster(model_file=str(holdout_model)).current_iteration() * 1.1)
    X, y, w = stack([dict(r, X=r["X"][:, keep]) for r in rows], cond, hz, nm, seed=3, **kw)
    print(f"{out.name}: {len(y):,} rows, {n} rounds", flush=True)
    m = lgb.train(PARAMS, lgb.Dataset(X, y, weight=w, feature_name=nm + ["h"]), n)
    m.save_model(str(out))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--w-base", type=float, default=0.3)
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--base", default="f", help="memory-free models t2<base>_<key>_final.txt")
    ap.add_argument("--name", default="rec7")
    ap.add_argument("--ongoing", action="store_true", help="recent-memory blend for ongoing windows too")
    a = ap.parse_args()
    tr, _, names = onset_rows()
    keep = [i for i, n in enumerate(names) if not n.startswith(LONG)]
    kb = [i for i, n in enumerate(names) if not n.startswith(MEMORY)]
    refresh({"train": tr}, names, a.days)
    rec = fit(tr, names, keep, COND, (6,), OUT / f"t2rec{a.days}_onset.txt", OUT / f"t2rec{a.days}_onset_final.txt")
    del tr
    rec_og = None
    if a.ongoing:
        tr_all, _, _ = pickle.load(open(OUT / "t2_rowse.pkl", "rb"))
        tr = tr_all["ongoing"]; del tr_all
        refresh({"train": tr}, names, a.days)
        rec_og = fit(tr, names, keep, "queue_ongoing", (1, 2, 3, 4, 5, 6), OUT / f"t2og_rec{a.days}_ongoing.txt",
                     OUT / f"t2rec{a.days}_ongoing_final.txt", **SUB)
        del tr
    base = {k: lgb.Booster(model_file=str(OUT / f"t2{a.base}_{k}_final.txt")) for k in ("onset", "ongoing")}
    thr_on = json.loads((OUT / f"t2{a.base}_report.json").read_text())["best"]["ongoing"]
    t2_data.RECENT_DAYS = a.days
    queue, links = {}, []
    for p in QUEUE_PANELS:
        for split in ("validation", "private"):
            blocks, bn = scored_windows(p, split)
            assert bn == names
            for b in blocks:
                if b["condition"] == COND:
                    P = (1 - a.w_base) * predict_window(rec, b["X"][:, keep], COND) + a.w_base * predict_window(base["onset"], b["X"][:, kb], COND)
                    pred = decode(P, COND, a.thr, 3)
                    links.append((split, int(pred[5].sum())))
                elif rec_og is not None:
                    P = (1 - a.w_base) * predict_window(rec_og, b["X"][:, keep], b["condition"]) + a.w_base * predict_window(base["ongoing"], b["X"][:, kb], b["condition"])
                    pred = decode(P, b["condition"], 0.45)
                else:
                    pred = decode(predict_window(base["ongoing"], b["X"][:, kb], b["condition"]), b["condition"], *thr_on)
                out = np.zeros_like(pred, dtype=np.int8)
                out[:, b["order"]] = pred
                queue[b["window_id"]] = out
    pickle.dump(queue, open(OUT / f"t2{a.name}_queue_final.pkl", "wb"))
    for s in ("validation", "private"):
        print(s, "onset links/window", round(float(np.mean([k for sp, k in links if sp == s])), 2), flush=True)
    print("scored windows:", len(queue), flush=True)


if __name__ == "__main__":
    main()
