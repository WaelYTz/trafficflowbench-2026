"""Memory-free ongoing on the scored windows: 0.2 base LightGBM (no MEMORY features) + 0.8 memory-free CNN x3.

Public windows (Mar 1-12) see last-split memory; on the raw-history Feb set the memory-free blend scores
0.9467 against 0.9482 for c8 (out/ongoing_nn_nomem.log), so it trades a little fresh-memory accuracy for
immunity to stale memory. Writes queue_<onset>_nm.pkl into kaggle_pkg/src_ds, with the onset taken from
the existing queue_<onset>_g.pkl files.
"""
from __future__ import annotations

import pickle

import lightgbm as lgb
import numpy as np
import torch

import t2_nnlib
from common import OUT, QUEUE_PANELS, ROOT
from t2_data import scored_windows
from t2_model import MEMORY, decode, predict_window

DS = ROOT / "kaggle_pkg" / "src_ds"
W_NN = 0.8


def main():
    base = lgb.Booster(model_file=str(OUT / "t2final_base_ongoing.txt"))
    nets = [torch.load(OUT / f"t2nn_ongoing_nm_s{s}.pt", weights_only=False) for s in (7, 11, 23)]
    ong, probs = {}, {}
    for p in QUEUE_PANELS:
        for split in ("validation", "private"):
            blocks, names = scored_windows(p, split)
            kb = [i for i, n in enumerate(names) if not n.startswith(MEMORY)]
            bl = [b for b in blocks if b["condition"] == "queue_ongoing"]
            pn = [t2_nnlib.predict(n, bl) for n in nets]
            for i, b in enumerate(bl):
                P = (1 - W_NN) * predict_window(base, b["X"][:, kb], b["condition"]) + W_NN * np.mean([x[i] for x in pn], 0)
                out = np.zeros((6, len(b["order"])), dtype=np.int8)
                out[:, b["order"]] = decode(P, b["condition"], 0.45)
                ong[b["window_id"]] = out
                probs[b["window_id"]] = P.astype(np.float32)
    pickle.dump(probs, open(OUT / "t2_ongoing_nm_scored_probs.pkl", "wb"))
    ref = pickle.load(open(DS / "queue_g_c8.pkl", "rb"))
    print(f"ongoing windows {len(ong)}; identical to c8 {sum((ong[w] == ref[w]).all() for w in ong)}; "
          f"March cells/window nm {np.mean([ong[w].sum() for w in ong if 'validation' in w]):.1f} "
          f"vs c8 {np.mean([ref[w].sum() for w in ong if 'validation' in w]):.1f}", flush=True)
    for on in ("pub", "g"):
        q = pickle.load(open(DS / f"queue_{on}_g.pkl", "rb"))
        q.update(ong)
        pickle.dump(q, open(DS / f"queue_{on}_nm.pkl", "wb"))


if __name__ == "__main__":
    main()
