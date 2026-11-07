"""Compare Task 2 decoders (and LightGBM / XGBoost blends) on the Feb windows, then write the queue
file with the best choice per condition.

Decoders turn the (6, L) probabilities of a window into 0/1 cells:
  thr    fixed probability threshold (onset: step 6 only, top-k fallback)
  eiou   per window, the top-k cells by probability with k maximising the expected IoU
         sum_{S} p / (|S| + sum_{not S} p), after scaling the probabilities and with a floor
"""
from __future__ import annotations

import argparse
import json
import pickle

import lightgbm as lgb
import numpy as np
import xgboost as xgb

from common import OUT, QUEUE_PANELS
from t2_data import scored_windows
from t2_model import HOLD, aggregate, decode, iou

CONDS = (("ongoing", "queue_ongoing"), ("onset", "queue_onset"))
BLENDS = {"lgb": (1.0, 0.0), "xgb": (0.0, 1.0), "mix": (0.5, 0.5)}


def load_models(tag: str) -> dict:
    out = {}
    for k, _ in CONDS:
        out[k, "lgb"] = lgb.Booster(model_file=str(OUT / f"t2{tag}_{k}.txt"))
        f = OUT / f"t2{tag}_{k}_xgb.json"
        if f.exists():
            out[k, "xgb"] = xgb.Booster(model_file=str(f))
    return out


def probs(models: dict, key: str, X: np.ndarray, cond: str) -> dict:
    """{'lgb': (6, L), 'xgb': (6, L)} probabilities in the block's own link order."""
    L = len(X)
    hs = (6,) if cond == "queue_onset" else range(1, 7)
    out = {}
    for src in ("lgb", "xgb"):
        if (key, src) not in models:
            continue
        m = models[key, src]
        P = np.zeros((6, L))
        for h in hs:
            Xh = np.column_stack([X, np.full(L, h, np.float32)]).astype(np.float32)
            P[h - 1] = m.predict(Xh) if src == "lgb" else m.predict(xgb.DMatrix(Xh, feature_names=m.feature_names))
        out[src] = P
    return out


def blend(Ps: dict, src: str) -> np.ndarray:
    if src not in BLENDS or (src != "lgb" and "xgb" not in Ps):
        return None
    a, b = BLENDS[src]
    return Ps["lgb"] if b == 0 else Ps["xgb"] if a == 0 else a * Ps["lgb"] + b * Ps["xgb"]


def eiou(P: np.ndarray, cond: str, floor: float = 0.0, scale: float = 1.0) -> np.ndarray:
    Q = np.clip(P * scale, 0, 1)
    if cond == "queue_onset":
        Q = Q.copy(); Q[:5] = 0
    p = Q.ravel()
    o = np.argsort(-p)
    ps = p[o]
    cs = np.cumsum(ps)
    k = np.arange(1, len(ps) + 1)
    e = cs / (k + ps.sum() - cs)
    e[1:][ps[1:] < floor] = -1
    pred = np.zeros(p.shape, bool)
    pred[o[:int(np.argmax(e)) + 1]] = True
    return pred.reshape(P.shape)


def apply(d: tuple, P: np.ndarray, cond: str) -> np.ndarray:
    return decode(P, cond, d[2], d[3]) if d[1] == "thr" else eiou(P, cond, d[2], d[3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--rows-tag", default=None, help="read t2_rows<rows-tag>.pkl")
    a = ap.parse_args()
    models = load_models(a.tag)
    fp = OUT / f"t2{a.tag}_febP.pkl"
    have_xgb = all((k, "xgb") in models for k, _ in CONDS)
    feb = pickle.load(open(fp, "rb")) if fp.exists() else None
    if feb is None or (have_xgb and "xgb" not in feb["onset"][0]["P"]):
        _, ev, _ = pickle.load(open(OUT / f"t2_rows{a.tag if a.rows_tag is None else a.rows_tag}.pkl", "rb"))
        feb = {k: [dict(P=probs(models, k, r["X"], c), y=r["y"], elig=r["elig"], panel=r["panel"])
                   for r in ev[k] if r["day"] >= HOLD] for k, c in CONDS}
        del ev
        pickle.dump(feb, open(fp, "wb"))
    best = {}
    for k, c in CONDS:
        res = {}
        for src in BLENDS:
            Ps = [blend(r["P"], src) for r in feb[k]]
            if Ps[0] is None:
                continue
            for thr in np.arange(0.25, 0.66, 0.05):
                for fb in ((1, 2, 3) if k == "onset" else (1,)):
                    d = (src, "thr", round(float(thr), 2), fb)
                    res[d] = [iou(apply(d, P, c), r["y"], r["elig"]) for P, r in zip(Ps, feb[k])]
            for fl in (0.0, 0.1, 0.2, 0.3):
                for sc in (0.8, 1.0, 1.2, 1.5):
                    d = (src, "eiou", fl, sc)
                    res[d] = [iou(apply(d, P, c), r["y"], r["elig"]) for P, r in zip(Ps, feb[k])]
        m = {d: float(np.mean(v)) for d, v in res.items()}
        top = sorted(m, key=m.get, reverse=True)
        print(k, "top decoders:", [(t, round(m[t], 4)) for t in top[:8]], flush=True)
        for src in BLENDS:
            s = [t for t in top if t[0] == src]
            if s:
                print(f"  best {src}: {s[0]} {m[s[0]]:.4f}", flush=True)
        best[k] = (top[0], res[top[0]])
    sc = []
    for k, c in CONDS:
        sc += [(r["panel"], c, s) for r, s in zip(feb[k], best[k][1])]
    rep = dict(model=aggregate(sc), best={k: list(v[0]) for k, v in best.items()})
    print("best:", rep, flush=True)
    (OUT / f"t2{a.tag}_decode_report.json").write_text(json.dumps(rep, indent=1))
    queue = {}
    for p in QUEUE_PANELS:
        for split in ("validation", "private"):
            blocks, _ = scored_windows(p, split)
            for b in blocks:
                k = "onset" if b["condition"] == "queue_onset" else "ongoing"
                d = best[k][0]
                pred = apply(d, blend(probs(models, k, b["X"], b["condition"]), d[0]), b["condition"])
                out = np.zeros_like(pred, dtype=np.int8)
                out[:, b["order"]] = pred
                queue[b["window_id"]] = out
    pickle.dump(queue, open(OUT / f"t2{a.tag}_queue_dec.pkl", "wb"))
    print("scored windows:", len(queue), flush=True)


if __name__ == "__main__":
    main()
