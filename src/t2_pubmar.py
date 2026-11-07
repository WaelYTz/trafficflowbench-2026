"""Public onset recipe refitted on Jun-Feb + ALL of March (validation masked layer, Task 1-filled history,
labels on observed cells; rows from scripts/onset_march_aug.py collect), for the PRIVATE (April) onset
windows only. March data is stamped before every April window, so this is causal for them; the March
(validation) windows keep the public onset, since a March fit would use days after their origin.

Evidence (out/onset_march_aug.log): LightGBM bags fitted on Jun-Feb + Mar 1-15 vs Jun-Feb only, scored on
the filled-history March bench onset windows of Mar 16-31.

  python t2_pubmar.py W [N_LGB N_CAT]   -> out/pubmar_scored_probs.pkl, kaggle_pkg/src_ds/queue_pubM_<og>.pkl
                                          (private onset replaced; validation = pub)
"""
from __future__ import annotations

import pickle
import sys

import numpy as np
import pandas as pd

from common import OUT, QUEUE_PANELS, REL, ROOT
from t2_model import decode
from t2_pub import history_scored
from t2_pubbag import Bag

DS = ROOT / "kaggle_pkg" / "src_ds"
THR = 0.55


def main():
    w = float(sys.argv[1])
    n_lgb = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    n_cat = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    R = dict(np.load(OUT / "pubbag_rows.npz"))
    M = dict(np.load(OUT / "pubbag_rows_mar.npz"))
    rounds = pickle.load(open(OUT / "pubbag_rounds.pkl", "rb"))
    lr, cr = int(np.mean(rounds["lgb"]) * 1.1), int(np.mean(rounds["cat"]) * 1.1)
    X = np.concatenate([R["X"], M["X"]]); y = np.concatenate([R["y"], M["y"]])
    wt = np.concatenate([np.ones(len(R["y"])), np.full(len(M["y"]), w)])
    import lightgbm as lgb
    from t2_pubbag import LGB_PARAMS
    lgbs = [lgb.train(dict(LGB_PARAMS, seed=100 + s, bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8),
                      lgb.Dataset(X, label=y, weight=wt), num_boost_round=lr) for s in range(n_lgb)]
    cats = []
    if n_cat:
        from catboost import CatBoostClassifier, Pool
        Xc = np.concatenate([R["X"], R["Xd"], M["X"]]); yc = np.concatenate([R["y"], R["yd"], M["y"]])
        wc = np.concatenate([np.ones(len(R["y"]) + len(R["yd"])), np.full(len(M["y"]), w)])
        for s in range(n_cat):
            m = CatBoostClassifier(iterations=cr, depth=7, learning_rate=0.05, l2_leaf_reg=10, task_type="GPU",
                                   random_seed=200 + s, loss_function="Logloss", verbose=0, train_dir=str(OUT / "catboost_info"))
            m.fit(Pool(Xc, yc, weight=wc)); cats.append(m)
    for i, m in enumerate(lgbs):
        m.save_model(str(OUT / f"pubmar_final_lgb_{i}.txt"))
    for i, m in enumerate(cats):
        m.save_model(str(OUT / f"pubmar_final_cat_{i}.cbm"))
    bag = Bag(lgbs, cats)
    rec = pickle.load(open(OUT / "t2rec7g2_queue_final_probs.pkl", "rb"))   # wid -> (P, travel->scorer order, cond)
    probs, onset = {}, {}
    for p in QUEUE_PANELS:
        wi = pd.read_csv(REL / "task2" / p / "private" / "window_index.csv")
        cond = dict(zip(wi.window_id, wi.condition))
        for wid, (V, Q, E, origin) in history_scored(p, "private").items():
            if cond[wid] != "queue_onset":
                continue
            P = bag.probs(p, V, Q, E, origin)[0] if cats else _lgb_only(bag, p, V, Q, E, origin, lgbs)
            probs[wid] = P
            out = np.zeros(P.shape, np.int8); out[:, rec[wid][1]] = decode(P, "queue_onset", THR, 1)
            onset[wid] = out
    pickle.dump(probs, open(OUT / "pubmar_scored_probs.pkl", "wb"))
    ref = pickle.load(open(DS / "queue_pub_c8.pkl", "rb"))
    print(f"private onset windows {len(onset)}; changed vs pub {sum(not (onset[w] == ref[w]).all() for w in onset)}", flush=True)
    for og in ("c8", "c8e"):
        f = DS / f"queue_pub_{og}.pkl"
        if f.exists():
            q = pickle.load(open(f, "rb")); q.update(onset)
            pickle.dump(q, open(DS / f"queue_pubM_{og}.pkl", "wb"))


def _lgb_only(bag, name, V, Q, E, origin, lgbs):
    onset_p, _, pos = bag.pub.panel(name)
    X = onset_p.features(V[:, pos].astype(np.float32), Q[:, pos].astype(np.float32), E[:, pos].astype(np.float32), origin, "queue_onset")[-len(pos):]
    P = np.zeros((6, len(pos)))
    P[5, pos] = np.mean([m.predict(X) for m in lgbs], 0)
    return P


if __name__ == "__main__":
    main()
