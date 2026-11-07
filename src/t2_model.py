"""Task 2 models: LightGBM for ongoing windows (link x horizon) and onset windows (link at T+30).

Trained on Jun-Jan train windows, early-stopped on Feb, and judged by the official window IoU on
Feb windows cut with the official-like selector. Uses only train data, so it is causal for every
validation and private window.
"""
from __future__ import annotations

import argparse
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import lightgbm as lgb
import numpy as np

from common import FAMILY, OUT, QUEUE_PANELS
from t2_data import build_train, scored_windows

HOLD = "2031-02"
MEMORY = ("onset_prior", "queued_prior", "recent_q", "recent_n", "recent_onset", "today_q_before")


def collect(mode: str):
    with ProcessPoolExecutor(4) as ex:
        parts = list(ex.map(partial(build_train, mode=mode), QUEUE_PANELS))
    names = parts[0][1]
    rows = {"onset": [], "ongoing": []}
    for r, _ in parts:
        for k in rows:
            rows[k] += r[k]
    return rows, names


def stack(wins, cond: str, horizons=(1, 2, 3, 4, 5, 6), names=None, window_stride=1, n_h=None,
          far_keep=1.0, seed=0):
    """Rows of (link, horizon) cells. Easy negatives far from any visible queue are subsampled
    (and re-weighted) so that dense ongoing windows fit in memory."""
    rng = np.random.default_rng(seed)
    X, y, w = [], [], []
    i_dn, i_up = names.index("km_to_q_dn"), names.index("km_to_q_up")
    for r in wins[::window_stride]:
        hs = horizons if n_h is None else rng.choice(horizons, n_h, replace=False)
        n_el = r["elig"][[h - 1 for h in hs]].sum()
        far = (r["X"][:, i_dn] > 8) & (r["X"][:, i_up] > 8)
        for h in hs:
            el = r["elig"][h - 1].copy()
            lab = r["y"][h - 1]
            wt = np.full(len(el), 1.0 / max(n_el, 1), np.float32)
            if far_keep < 1:
                drop = far & ~lab & (rng.random(len(el)) > far_keep)
                el &= ~drop
                wt[far & ~lab] /= far_keep
            if not el.any():
                continue
            X.append(np.column_stack([r["X"][el], np.full(el.sum(), h, np.float32)]))
            y.append(lab[el])
            w.append(wt[el])
    w = np.concatenate(w)
    # Mean-one weights: raw window weights (~1e-3) would make the hessian floors block most splits.
    return np.vstack(X).astype(np.float32), np.concatenate(y).astype(np.float32), w / w.mean()


def predict_window(model, X: np.ndarray, cond: str) -> np.ndarray:
    """Probability (6, L) in the block's own link order."""
    L = len(X)
    P = np.zeros((6, L))
    hs = (6,) if cond == "queue_onset" else range(1, 7)
    for h in hs:
        P[h - 1] = model.predict(np.column_stack([X, np.full(L, h, np.float32)]))
    return P


def decode(P: np.ndarray, cond: str, thr: float, fallback: int = 1) -> np.ndarray:
    pred = P > thr
    if cond == "queue_onset":
        pred[:5] = False
        if not pred[5].any():
            pred[5, np.argsort(-P[5])[:fallback]] = True
    return pred


def iou(pred, y, el) -> float:
    t, p = y & el, pred & el
    u = (t | p).sum()
    return 1.0 if u == 0 else (t & p).sum() / u


def aggregate(scores: list[tuple[str, str, float]]) -> dict:
    """Official order: window -> panel x condition -> panel -> family -> overall."""
    pc = {}
    for p, c, s in scores:
        pc.setdefault((p, c), []).append(s)
    panel = {}
    for (p, c), v in pc.items():
        panel.setdefault(p, []).append(np.mean(v))
    fam = {}
    for p, v in panel.items():
        fam.setdefault(FAMILY[p], []).append(np.mean(v))
    by_cond = {c: float(np.mean([np.mean(v) for (p, cc), v in pc.items() if cc == c])) for c in ("queue_onset", "queue_ongoing")}
    return dict(overall=float(np.mean([np.mean(v) for v in fam.values()])), **by_cond)


PARAMS = dict(objective="binary", learning_rate=0.04, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=255, verbosity=-1, num_threads=12, seed=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--tag", default="", help="suffix for output files")
    ap.add_argument("--rows-tag", default=None, help="reuse t2_rows<rows-tag>.pkl")
    ap.add_argument("--leaves", type=int, default=PARAMS["num_leaves"])
    ap.add_argument("--min-data", type=int, default=PARAMS["min_data_in_leaf"])
    ap.add_argument("--stride", type=int, default=3, help="ongoing window stride")
    ap.add_argument("--far-keep", type=float, default=0.15)
    ap.add_argument("--drop-mem", action="store_true", help="leave out place-time memory features (priors, recent counts)")
    ap.add_argument("--eval-only", action="store_true", help="load t2<tag>_<key>.txt instead of training")
    a = ap.parse_args()
    PARAMS.update(num_leaves=a.leaves, min_data_in_leaf=a.min_data)
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / f"t2_rows{a.tag if a.rows_tag is None else a.rows_tag}.pkl"
    if a.rebuild or not cache.exists():
        tr, names = collect("train")
        ev, _ = collect("eval")
        pickle.dump((tr, ev, names), open(cache, "wb"), protocol=4)
    tr, ev, names = pickle.load(open(cache, "rb"))
    keep = np.arange(len(names))
    if a.drop_mem:
        keep = np.array([i for i, n in enumerate(names) if not n.startswith(MEMORY)])
        for rows in (tr, ev):
            for k in rows:
                for r in rows[k]:
                    r["X"] = r["X"][:, keep]
        names = [names[i] for i in keep]
        print("memory-free features:", len(names), flush=True)
    feat = names + ["h"]
    for k in ("onset", "ongoing"):
        print(k, "train windows", sum(r["day"] < HOLD for r in tr[k]), "feb windows", sum(r["day"] >= HOLD for r in tr[k]),
              "eval windows (feb)", sum(r["day"] >= HOLD for r in ev[k]), flush=True)
    models = {}
    for cond, key, hz in (("queue_ongoing", "ongoing", (1, 2, 3, 4, 5, 6)), ("queue_onset", "onset", (6,))):
        if a.eval_only:
            models[key] = lgb.Booster(model_file=str(OUT / f"t2{a.tag}_{key}.txt"))
            continue
        kw = dict(window_stride=a.stride, n_h=3, far_keep=a.far_keep) if key == "ongoing" else {}
        Xt, yt, wt = stack([r for r in tr[key] if r["day"] < HOLD], cond, hz, names, seed=1, **kw)
        Xv, yv, wv = stack([r for r in tr[key] if r["day"] >= HOLD], cond, hz, names, seed=2,
                           **(dict(kw, window_stride=3) if kw else kw))
        print(f"{key}: train rows {len(yt):,} (pos {yt.mean():.3f}), feb rows {len(yv):,}", flush=True)
        dt = lgb.Dataset(Xt, yt, weight=wt, feature_name=feat, free_raw_data=True)
        dv = lgb.Dataset(Xv, yv, weight=wv, reference=dt)
        m = lgb.train(PARAMS, dt, a.rounds, valid_sets=[dv], callbacks=[lgb.early_stopping(150), lgb.log_evaluation(250)])
        models[key] = m
        m.save_model(str(OUT / f"t2{a.tag}_{key}.txt"))
    # Window IoU on official-like Feb windows, against the baselines.
    feb = {k: [r for r in ev[k] if r["day"] >= HOLD] for k in ev}
    P = {k: [predict_window(models[k], r["X"], c) for r in feb[k]] for k, c in (("ongoing", "queue_ongoing"), ("onset", "queue_onset"))}
    if "onset_prior" in names:
        i_q = names.index("q_last"); i_mem = names.index("onset_prior")
        base = []
        for r in feb["ongoing"]:
            base.append((r["panel"], "queue_ongoing", iou(np.broadcast_to(r["X"][:, i_q] > 0.5, (6, len(r["X"]))), r["y"], r["elig"])))
        for r in feb["onset"]:
            pr = np.zeros((6, len(r["X"])), bool); pr[5, np.argsort(-r["X"][:, i_mem])[:3]] = True
            base.append((r["panel"], "queue_onset", iou(pr, r["y"], r["elig"])))
        print("baseline (persistence / memory top-3):", aggregate(base), flush=True)
    best = {}
    for key, cond in (("ongoing", "queue_ongoing"), ("onset", "queue_onset")):
        res = {}
        for thr in np.arange(0.2, 0.71, 0.05):
            for fb in ((1, 2, 3) if key == "onset" else (1,)):
                s = [iou(decode(p, cond, thr, fb), r["y"], r["elig"]) for p, r in zip(P[key], feb[key])]
                res[(round(float(thr), 2), fb)] = float(np.mean(s))
        best[key] = max(res, key=res.get)
        print(key, "IoU by (threshold, fallback):", {k: round(v, 3) for k, v in sorted(res.items())}, flush=True)
    sc = []
    for key, cond in (("ongoing", "queue_ongoing"), ("onset", "queue_onset")):
        thr, fb = best[key]
        sc += [(r["panel"], cond, iou(decode(p, cond, thr, fb), r["y"], r["elig"])) for p, r in zip(P[key], feb[key])]
    rep = dict(model=aggregate(sc), best={k: list(v) for k, v in best.items()})
    print("model:", rep, flush=True)
    (OUT / f"t2{a.tag}_report.json").write_text(json.dumps(rep, indent=1))
    # Scored windows.
    queue = {}
    for p in QUEUE_PANELS:
        for split in ("validation", "private"):
            blocks, nm = scored_windows(p, split)
            assert [nm[i] for i in keep] == names
            for b in blocks:
                b["X"] = b["X"][:, keep]
                key = "onset" if b["condition"] == "queue_onset" else "ongoing"
                thr, fb = best[key]
                pred = decode(predict_window(models[key], b["X"], b["condition"]), b["condition"], thr, fb)
                out = np.zeros_like(pred, dtype=np.int8)
                out[:, b["order"]] = pred
                queue[b["window_id"]] = out
    pickle.dump(queue, open(OUT / f"t2{a.tag}_queue.pkl", "wb"))
    print("scored windows:", len(queue), flush=True)


if __name__ == "__main__":
    main()
