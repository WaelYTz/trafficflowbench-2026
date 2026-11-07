"""Score Task 2 models on the masked-layer pseudo-windows (t2_pseudo.py): Feb vs March, ours vs public."""
from __future__ import annotations

import argparse
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import OUT
from t2_model import MEMORY, decode, iou, predict_window
from t2_pub import Public

CONDS = (("onset", "queue_onset"), ("ongoing", "queue_ongoing"))
THR = {"onset": (0.2, 0.3, 0.4, 0.45, 0.55), "ongoing": (0.35, 0.45, 0.55)}


def probs(src: str, tags: list[str]):
    f = OUT / f"t2_pseudo_{src}_probs.pkl"
    cache = pickle.load(open(f, "rb")) if f.exists() else {}
    rows, names = pickle.load(open(OUT / f"t2_pseudo_{src}.pkl", "rb"))
    pub = None
    for key, cond in CONDS:
        for tag in tags:
            if (tag, key) in cache:
                continue
            if tag == "cat":
                if key != "onset":
                    continue
                from catboost import CatBoostClassifier
                m = CatBoostClassifier(); m.load_model(str(OUT / "t2f_onset_cat.cbm"))
                keep = [i for i, n in enumerate(names) if not n.startswith(MEMORY)]
                out = []
                for r in rows[key]:
                    L = len(r["X"]); p = np.zeros((6, L))
                    p[5] = m.predict_proba(np.column_stack([r["X"][:, keep], np.full(L, 6, np.float32)]))[:, 1]
                    out.append(p)
            elif tag == "pub":
                pub = pub or Public()
                out = []
                for r in rows[key]:
                    origin = pd.Timestamp(f"{r['day']}T{r['slot'] // 12:02d}:{r['slot'] % 12 * 5:02d}:00Z")
                    out.append(pub.probs(r["panel"], *r["hist"], origin, cond))
            else:
                m = lgb.Booster(model_file=str(OUT / f"t2{tag}_{key}.txt"))
                keep = [i for i, n in enumerate(names) if not (n.startswith(MEMORY) and m.num_feature() < len(names) + 1)]
                out = [predict_window(m, r["X"][:, keep], cond) for r in rows[key]]
            cache[(tag, key)] = out
            pickle.dump(cache, open(f, "wb"))
            print("probs", src, tag, key, len(out), flush=True)
    return rows, cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", default=["feb", "mar"])
    ap.add_argument("--tags", nargs="+", default=["f", "e", "pub"])
    a = ap.parse_args()
    for src in a.src:
        rows, P = probs(src, a.tags)
        for key, cond in CONDS:
            R = rows[key]
            truth = np.mean([r["y"].sum(1)[5] if key == "onset" else r["y"].sum() / 6 for r in R])
            print(f"\n== {src} {key}: {len(R)} windows, truth links/window {truth:.2f}", flush=True)
            cands = {t: P[(t, key)] for t in a.tags if (t, key) in P}
            if "pub" in a.tags:
                for t in list(cands):
                    if t != "pub":
                        cands[f"{t}+pub"] = [(x + y) / 2 for x, y in zip(P[(t, key)], P[("pub", key)])]
            for name, PP in cands.items():
                res = []
                for thr in THR[key]:
                    for fb in ((1, 3) if key == "onset" else (1,)):
                        preds = [decode(p, cond, thr, fb) for p in PP]
                        s = np.mean([iou(d, r["y"], r["elig"]) for d, r in zip(preds, R)])
                        n = np.mean([d[5].sum() if key == "onset" else d.sum() / 6 for d in preds])
                        res.append((s, thr, fb, n))
                res.sort(reverse=True)
                print(f"  {name:8s} " + "  ".join(f"{s:.4f}@{thr},{fb} ({n:.1f})" for s, thr, fb, n in res[:4]), flush=True)


if __name__ == "__main__":
    main()
