"""Public anubisyy onset recipe, retrained with seed bagging (LB: their single LGBM + CatBoost pair reached
onset IoU 0.776, our best). Same features (ext/anubis_assets/queue_features.py, 12 history steps before T),
same training origins (first origin of each contiguous run of onset candidates), same targets (T+30).

  collect   -> out/pubbag_rows.npz (all train days; Feb flagged)
  eval      -> LGBM / CatBoost bags fitted on Jun-Jan, scored on the raw-history Feb official-like windows
  final     -> bags refitted on Jun-Feb with the eval's round counts; queue files for the scored windows
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import OUT, QUEUE_PANELS, REL, ROOT

ASSETS = ROOT / "ext" / "anubis_assets"
os.environ.setdefault("TFB_DATA_ROOT", str(REL))
os.environ.setdefault("TFB_OUTPUT_ROOT", str(OUT / "anubis_tmp"))
os.environ.setdefault("TFB_THREADS", "8")
sys.path.insert(0, str(ASSETS))
import queue_features as QF  # noqa: E402

ROWS = OUT / "pubbag_rows.npz"
TAG = ""          # "T": history = the 12 rows ending AT T (row T from the masked layer), see collect_T
# "TP": as T without the place-identity columns (panel number, link index, milepost), which encode where
# train-month bottlenecks were; March onsets moved (D12_I5_S, D7_I405_S), as memory showed.
PLACE_COLS = (0, 1, 57)
KEEP = None       # column subset used by every model (None = all 83)


def cols(X):
    return X if KEEP is None else X[:, KEEP]
LGB_PARAMS = dict(objective="binary", metric="binary_logloss", learning_rate=0.04, num_leaves=15, min_data_in_leaf=100,
                  lambda_l2=10, feature_fraction=0.9, num_threads=8, verbosity=-1)


def collect():
    """Training rows: X at T+30 for eligible cells; a 5%-history-dropout copy for the CatBoost bags."""
    rng = np.random.default_rng(20261005)
    out = {k: [] for k in ("X", "y", "Xd", "yd", "feb", "febd")}
    for pi, name in enumerate(QF.panel_names()):
        p = QF.Panel(name, pi)
        for f in sorted((REL / "corridors" / name / "train" / "mainline_states").glob("**/*.parquet")):
            times, v, q, e = p.train_day(f)
            cand = QF.candidate_origins(p, times, v, e)["queue_onset"]
            cs = set(cand)
            feb = f.parent.name.endswith("2031-02")
            for t in [a for a in cand if a - 1 not in cs]:
                y = (v[t + QF.HORIZON] <= p.cut) & np.isfinite(v[t + QF.HORIZON])
                el = e[t + QF.HORIZON] & np.isfinite(v[t + QF.HORIZON])
                x = p.features(v[t - QF.HISTORY:t], q[t - QF.HISTORY:t], e[t - QF.HISTORY:t], pd.Timestamp(times[t]), "queue_onset")[-len(p.links):]
                out["X"].append(x[el]); out["y"].append(y[el]); out["feb"].append(np.full(el.sum(), feb))
                hv, hq, he = v[t - QF.HISTORY:t].copy(), q[t - QF.HISTORY:t].copy(), e[t - QF.HISTORY:t].copy()
                drop = rng.random(hv.shape) < 0.05
                hv[drop] = np.nan; hq[drop] = np.nan; he[drop] = False
                xd = p.features(hv, hq, he, pd.Timestamp(times[t]), "queue_onset")[-len(p.links):]
                out["Xd"].append(xd[el]); out["yd"].append(y[el]); out["febd"].append(np.full(el.sum(), feb))
        print(name, "rows", sum(len(a) for a in out["y"]), flush=True)
    np.savez_compressed(ROWS, **{k: np.concatenate(v) for k, v in out.items()})


def _pub_index(name):
    """Cache (scorer) order -> the public model's link order."""
    from common import network
    p = QF.Panel(name, QF.panel_names().index(name))
    sc = {l: i for i, l in enumerate(network(name).link_id)}
    return p, np.array([sc[l] for l in p.links])


def collect_T():
    """As collect, but the 12 history rows are T-55 .. T: rows before T from the full layer (window_history
    is complete) and the row at T from the train masked layer, which is how the row at T is released on
    the scored splits (Task 1 masks 20-50% of it). The row at T is released data stamped T, so it is causal."""
    from common import days, load
    rng = np.random.default_rng(20261006)
    out = {k: [] for k in ("X", "y", "Xd", "yd", "feb", "febd")}
    for name in QF.panel_names():
        p, ix = _pub_index(name)
        A = {k: load(name, "train", k, mmap=True) for k in ("v", "q", "elig", "vm", "qm")}
        for d, day in enumerate(days(name, "train")):
            v = np.asarray(A["v"][d], np.float32)[:, ix]; q = np.asarray(A["q"][d], np.float32)[:, ix]
            e = np.asarray(A["elig"][d])[:, ix].astype(bool)
            vm = np.asarray(A["vm"][d], np.float32)[:, ix]; qm = np.asarray(A["qm"][d], np.float32)[:, ix]
            times = pd.date_range(f"{day}T00:00:00Z", periods=v.shape[0], freq="5min")
            cand = QF.candidate_origins(p, times, v, e)["queue_onset"]
            cs = set(cand)
            feb = day.startswith("2031-02")
            for t in [a for a in cand if a - 1 not in cs]:
                y = (v[t + QF.HORIZON] <= p.cut) & np.isfinite(v[t + QF.HORIZON])
                el = e[t + QF.HORIZON] & np.isfinite(v[t + QF.HORIZON])
                hv = np.vstack([v[t - QF.HISTORY + 1:t], vm[t:t + 1]]); hq = np.vstack([q[t - QF.HISTORY + 1:t], qm[t:t + 1]])
                he = np.isfinite(hv)
                x = p.features(hv, hq, he, times[t], "queue_onset")[-len(p.links):]
                out["X"].append(x[el]); out["y"].append(y[el]); out["feb"].append(np.full(el.sum(), feb))
                hv, hq = hv.copy(), hq.copy()
                drop = rng.random(hv.shape) < 0.05
                hv[drop] = np.nan; hq[drop] = np.nan
                xd = p.features(hv, hq, np.isfinite(hv), times[t], "queue_onset")[-len(p.links):]
                out["Xd"].append(xd[el]); out["yd"].append(y[el]); out["febd"].append(np.full(el.sum(), feb))
        print(name, "rows", sum(len(a) for a in out["y"]), flush=True)
    np.savez_compressed(OUT / "pubbag_rows_T.npz", **{k: np.concatenate(v) for k, v in out.items()})


def history_T_train(panel, day, slot):
    """Travel-order (12, L) arrays T-55 .. T for a train day: full layer before T, masked layer at T."""
    from common import days, load
    from t2_data import Static
    st = Static(panel); o = st.order
    d = days(panel, "train").index(day)
    g = lambda k, a, b: np.asarray(load(panel, "train", k, mmap=True)[d, a:b], float)[:, o]
    v = np.vstack([g("v", slot - 11, slot), g("vm", slot, slot + 1)]); q = np.vstack([g("q", slot - 11, slot), g("qm", slot, slot + 1)])
    return v, q, np.isfinite(v).astype(float)


def history_T_scored(panel, split):
    """Scored windows: window_history rows T-55 .. T-5 plus the released (masked-layer) row at T."""
    from common import days, load
    from t2_data import Static
    from t2_pub import history_scored
    st = Static(panel); o = st.order
    di = {d: i for i, d in enumerate(days(panel, split))}
    V0, Q0 = load(panel, split, "v", mmap=True), load(panel, split, "q", mmap=True)
    out = {}
    for wid, (V, Q, E, origin) in history_scored(panel, split).items():
        d = di[str(origin.date())]; slot = origin.hour * 12 + origin.minute // 5
        vT = np.asarray(V0[d, slot], float)[o]; qT = np.asarray(Q0[d, slot], float)[o]
        v = np.vstack([V[1:], vT[None]]); q = np.vstack([Q[1:], qT[None]])
        out[wid] = (v, q, np.isfinite(v).astype(float), origin)
    return out


def fit_lgb(X, y, Xv, yv, seed, rounds=None):
    p = dict(LGB_PARAMS, seed=seed, bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8)
    if rounds:
        return lgb.train(p, lgb.Dataset(X, label=y), num_boost_round=rounds)
    return lgb.train(p, lgb.Dataset(X, label=y), num_boost_round=3000, valid_sets=[lgb.Dataset(Xv, label=yv)],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def fit_cat(X, y, Xv, yv, seed, rounds=None):
    from catboost import CatBoostClassifier
    kw = dict(depth=7, learning_rate=0.05, l2_leaf_reg=10, task_type="GPU", random_seed=seed, loss_function="Logloss",
              verbose=0, train_dir=str(OUT / "catboost_info"))
    if rounds:
        m = CatBoostClassifier(iterations=rounds, **kw); m.fit(X, y)
    else:
        m = CatBoostClassifier(iterations=4000, early_stopping_rounds=150, **kw); m.fit(X, y, eval_set=(Xv, yv), use_best_model=True)
    return m


class Bag:
    """Average of LightGBM and CatBoost members on the public features; predicts (6, L) in travel order."""
    def __init__(self, lgbs, cats):
        self.lgbs, self.cats = lgbs, cats
        from t2_pub import Public
        self.pub = Public()

    def probs(self, name, Vh, Qh, Eh, origin):
        onset_p, _, pos = self.pub.panel(name)
        v, q, e = Vh[:, pos].astype(np.float32), Qh[:, pos].astype(np.float32), Eh[:, pos]
        X = cols(onset_p.features(v, q, e.astype(np.float32), origin, "queue_onset")[-len(pos):])
        pl = np.mean([m.predict(X) for m in self.lgbs], 0)
        pc = np.mean([m.predict_proba(X)[:, 1] for m in self.cats], 0)
        P = np.zeros((6, len(pos)))
        P[5, pos] = 0.5 * pl + 0.5 * pc
        return P, np.zeros(1)


def raw_feb_eval(bag_probs, hist=None):
    """Official aggregate on the raw-history Feb onset windows (the faithful proxy; pub single pair 0.879)."""
    from t2_model import HOLD, aggregate, decode, iou
    from t2_onset_abl import onset_rows
    from t2_pub import history_from_cache
    tr, ev, names = onset_rows()
    feb = [r for r in ev if r["day"] >= HOLD]
    P = []
    for r in feb:
        v, q, e = (hist or history_from_cache)(r["panel"], r["day"], r["slot"])
        origin = pd.Timestamp(f"{r['day']}T{r['slot'] // 12:02d}:{r['slot'] % 12 * 5:02d}:00Z")
        P.append(bag_probs(r["panel"], v, q, e.astype(float), origin))
    res = {}
    for thr in (0.45, 0.5, 0.55, 0.6):
        sc = [iou(decode(p, "queue_onset", thr, 1), r["y"], r["elig"]) for p, r in zip(P, feb)]
        res[thr] = aggregate([(r["panel"], "queue_onset", x) for r, x in zip(feb, sc)])["overall"]
    return res, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("collect", "eval", "final"))
    ap.add_argument("--n-lgb", type=int, default=5)
    ap.add_argument("--n-cat", type=int, default=3)
    ap.add_argument("--tag", default="", choices=("", "T", "TP"))
    a = ap.parse_args()
    sfx = f"_{a.tag}" if a.tag else ""
    hist = history_T_train if a.tag in ("T", "TP") else None
    global KEEP
    if a.tag == "TP":
        KEEP = [i for i in range(83) if i not in PLACE_COLS]
    if a.step == "collect":
        collect_T() if a.tag == "T" else collect()
        return
    R = dict(np.load(OUT / f"pubbag_rows{'_T' if a.tag in ('T', 'TP') else ''}.npz"))
    R["X"], R["Xd"] = cols(R["X"]), cols(R["Xd"])
    tr, va = ~R["feb"], R["feb"]
    trd, vad = ~R["febd"], R["febd"]
    if a.step == "eval":
        lgbs = [fit_lgb(R["X"][tr], R["y"][tr], R["X"][va], R["y"][va], 100 + s) for s in range(a.n_lgb)]
        print("lgb best rounds", [m.best_iteration for m in lgbs], flush=True)
        # CatBoost: dropout copy (augmented) plus the clean rows, as their recipe trains on both reps
        Xc = np.concatenate([R["X"][tr], R["Xd"][trd]]); yc = np.concatenate([R["y"][tr], R["yd"][trd]])
        cats = [fit_cat(Xc, yc, R["X"][va], R["y"][va], 200 + s) for s in range(a.n_cat)]
        print("cat best rounds", [m.get_best_iteration() for m in cats], flush=True)
        pickle.dump(dict(lgb=[m.best_iteration for m in lgbs], cat=[m.get_best_iteration() for m in cats]),
                    open(OUT / f"pubbag_rounds{sfx}.pkl", "wb"))
        bag = Bag(lgbs, cats)
        for nl, nc, tag in ((1, 1, "single pair"), (a.n_lgb, a.n_cat, "full bag"), (a.n_lgb, 0, "lgb bag"), (0, a.n_cat, "cat bag")):
            b = Bag(lgbs[:nl], cats[:nc]) if nl and nc else None
            if not nc:
                f = lambda n, V, Q, E, o, L=lgbs[:nl]: _single(bag, n, V, Q, E, o, L, [])
            elif not nl:
                f = lambda n, V, Q, E, o, C=cats[:nc]: _single(bag, n, V, Q, E, o, [], C)
            else:
                f = lambda n, V, Q, E, o, b=b: b.probs(n, V, Q, E, o)[0]
            res, P = raw_feb_eval(f, hist)
            print(tag, {k: round(v, 4) for k, v in res.items()}, flush=True)
            if tag == "full bag":
                pickle.dump(P, open(OUT / f"pubbag_feb_probs{sfx}.pkl", "wb"))
        return
    rounds = pickle.load(open(OUT / f"pubbag_rounds{sfx}.pkl", "rb"))
    lr = int(np.mean(rounds["lgb"]) * 1.1); cr = int(np.mean(rounds["cat"]) * 1.1)
    lgbs = [fit_lgb(R["X"], R["y"], None, None, 100 + s, rounds=lr) for s in range(a.n_lgb)]
    Xc = np.concatenate([R["X"], R["Xd"]]); yc = np.concatenate([R["y"], R["yd"]])
    cats = [fit_cat(Xc, yc, None, None, 200 + s, rounds=cr) for s in range(a.n_cat)]
    for i, m in enumerate(lgbs):
        m.save_model(str(OUT / f"pubbag{sfx}_lgb_{i}.txt"))
    for i, m in enumerate(cats):
        m.save_model(str(OUT / f"pubbag{sfx}_cat_{i}.cbm"))
    bag = Bag(lgbs, cats)
    from t2_pub import history_scored
    probs = {}
    for p in QUEUE_PANELS:
        for split in ("validation", "private"):
            wi = pd.read_csv(REL / "task2" / p / split / "window_index.csv")
            cond = dict(zip(wi.window_id, wi.condition))
            for wid, (V, Q, E, origin) in (history_T_scored(p, split) if a.tag in ("T", "TP") else history_scored(p, split)).items():
                if cond[wid] == "queue_onset":
                    probs[wid] = bag.probs(p, V, Q, E, origin)[0]
    pickle.dump(probs, open(OUT / f"pubbag_scored_probs{sfx}.pkl", "wb"))
    print("final bag: scored onset windows", len(probs), flush=True)


def _single(bag, name, Vh, Qh, Eh, origin, lgbs, cats):
    onset_p, _, pos = bag.pub.panel(name)
    v, q, e = Vh[:, pos].astype(np.float32), Qh[:, pos].astype(np.float32), Eh[:, pos]
    X = cols(onset_p.features(v, q, e.astype(np.float32), origin, "queue_onset")[-len(pos):])
    parts = [m.predict(X) for m in lgbs] + [m.predict_proba(X)[:, 1] for m in cats]
    P = np.zeros((6, len(pos)))
    P[5, pos] = np.mean(parts, 0)
    return P


if __name__ == "__main__":
    main()
