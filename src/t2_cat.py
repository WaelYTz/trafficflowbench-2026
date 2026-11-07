"""CatBoost (GPU) onset model on the memory-free features, with feature-dropout augmentation, and its
blends with our LightGBM onset model and the public onset model on the Feb windows."""
from __future__ import annotations

import pickle

import lightgbm as lgb
import numpy as np
from catboost import CatBoostClassifier

from common import OUT
from t2_model import HOLD, MEMORY, decode, iou, predict_window, stack

COND = "queue_onset"


def main():
    tr, ev, names = pickle.load(open(OUT / "t2_rowse.pkl", "rb"))
    keep = [i for i, n in enumerate(names) if not n.startswith(MEMORY)]
    rows_tr = [dict(r, X=r["X"][:, keep]) for r in tr["onset"] if r["day"] < HOLD]
    rows_fv = [dict(r, X=r["X"][:, keep]) for r in tr["onset"] if r["day"] >= HOLD]
    feb = [dict(r, X=r["X"][:, keep]) for r in ev["onset"] if r["day"] >= HOLD]
    del tr, ev
    nm = [names[i] for i in keep]
    X, y, w = stack(rows_tr, COND, (6,), nm, seed=1)
    Xv, yv, wv = stack(rows_fv, COND, (6,), nm, seed=2)
    rng = np.random.default_rng(20261023)
    Xa = X.copy(); Xa[rng.random(Xa.shape) < 0.05] = np.nan          # a second copy with 5% of values missing
    X2, y2, w2 = np.vstack([X, Xa]), np.concatenate([y, y]), np.concatenate([w, w])
    m = CatBoostClassifier(iterations=4000, depth=7, learning_rate=0.05, l2_leaf_reg=10, task_type="GPU",
                           random_seed=20261023, loss_function="Logloss", early_stopping_rounds=150, verbose=500)
    m.fit(X2, y2, sample_weight=w2, eval_set=(Xv, yv), use_best_model=True)
    m.save_model(str(OUT / "t2f_onset_cat.cbm"))
    lg = lgb.Booster(model_file=str(OUT / "t2f_onset.txt"))
    pub = pickle.load(open(OUT / "t2pub_feb_f.pkl", "rb"))["onset"]   # (ours, public) probabilities per Feb window
    res = {}
    for i, r in enumerate(feb):
        L = len(r["X"])
        pc = np.zeros((6, L)); pc[5] = m.predict_proba(np.column_stack([r["X"], np.full(L, 6, np.float32)]))[:, 1]
        pl = predict_window(lg, r["X"], COND)
        pp = pub[i][1]
        for name, P in (("cat", pc), ("lgb", pl), ("pub", pp), ("cat+lgb", (pc + pl) / 2), ("cat+pub", (pc + pp) / 2),
                        ("all3", (pc + pl + pp) / 3)):
            for thr in (0.35, 0.45, 0.55):
                for fb in (1, 3):
                    res.setdefault((name, thr, fb), []).append(iou(decode(P, COND, thr, fb), r["y"], r["elig"]))
    m_ = {k: float(np.mean(v)) for k, v in res.items()}
    for name in ("cat", "lgb", "pub", "cat+lgb", "cat+pub", "all3"):
        best = max((k for k in m_ if k[0] == name), key=m_.get)
        print(f"{name:8s} best {best[1:]} Feb onset IoU {m_[best]:.4f}", flush=True)


if __name__ == "__main__":
    main()
