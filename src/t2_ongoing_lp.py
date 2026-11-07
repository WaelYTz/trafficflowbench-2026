"""Ongoing c8 with the CNNs retrained only on official-like windows (persistence IoU <= 0.9; Jun-Jan fit, best
filtered-Feb epoch; out/t2nn_ongoing_lp0.9_s{7,11,23}.pt): P = 0.2 P_lgb + 0.8 P_cnn_lp, thr 0.45.
Raw Feb pers<=0.9: c8 0.8844 -> 0.9019; March sets ~0 (out/ongoing_lowpers_check.log) -> LB decides.
P_lgb is recovered from the c8 probabilities and the old CNN average: P_lgb = (P_c8 - 0.8 P_cnn_old) / 0.2.

  python t2_ongoing_lp.py   -> kaggle_pkg/src_ds/queue_{pub,gate}_lp.pkl
"""
from __future__ import annotations

import pickle

import numpy as np
import torch

import t2_data
import t2_nnlib
from common import OUT, QUEUE_PANELS, ROOT
from t2_data import scored_windows
from t2_model import decode
from t2_onset_abl import LONG

DS = ROOT / "kaggle_pkg" / "src_ds"
t2_data.RECENT_DAYS = 7


def main():
    c8 = pickle.load(open(OUT / "t2rec7gc8_queue_final_probs.pkl", "rb"))
    old = [torch.load(OUT / f"t2final_nn_ongoing_s{s}.pt", weights_only=False) for s in (7, 11, 23)]
    new = [torch.load(OUT / f"t2nn_ongoing_lp0.9_s{s}.pt", weights_only=False) for s in (7, 11, 23)]
    out, lo, hi = {}, [], []
    for p in QUEUE_PANELS:
        for split in ("validation", "private"):
            blocks, names = scored_windows(p, split)
            kr = [i for i, n in enumerate(names) if not n.startswith(LONG)]
            B = [b for b in blocks if b["condition"] == "queue_ongoing"]
            po = np.mean([t2_nnlib.predict(m, B) for m in old], 0)
            pn = np.mean([t2_nnlib.predict(m, [dict(b, X=b["X"][:, kr]) for b in B]) for m in new], 0)
            for b, a, n in zip(B, po, pn):
                P, order, cond = c8[b["window_id"]]
                pl = (P - 0.8 * a) / 0.2
                lo.append(pl.min()); hi.append(pl.max())
                out[b["window_id"]] = (0.2 * np.clip(pl, 0, 1) + 0.8 * n, order)
    print(f"recovered LGBM probs range [{min(lo):.3f}, {max(hi):.3f}] (must be ~[0, 1])", flush=True)
    for src in ("pub", "gate"):
        q = pickle.load(open(DS / f"queue_{src}_c8.pkl", "rb"))
        ch = 0
        for w, (P, order) in out.items():
            new_q = np.zeros_like(q[w]); new_q[:, order] = decode(P, "queue_ongoing", 0.45)
            ch += not (new_q == q[w]).all(); q[w] = new_q
        pickle.dump(q, open(DS / f"queue_{src}_lp.pkl", "wb"))
        print(f"queue_{src}_lp.pkl: {ch} ongoing windows changed vs c8", flush=True)


if __name__ == "__main__":
    main()
