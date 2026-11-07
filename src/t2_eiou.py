"""Expected-IoU decoding for ongoing windows: per window, the top-k cells of p**a with k maximising
sum_top / (k + sum_rest), instead of a fixed 0.45 threshold. Future eligibility is unknown on scored windows,
so all cells count (on the March bench this matches decoding with the true eligibility for ongoing).
Evidence (a = 1.3; scripts/eiou_decode.py, eiou_ongoing_feb.py): March bench c8 0.9149 -> 0.9181;
raw Feb 0.9482 -> 0.9492, low-persistence subset (pers <= 0.7) 0.7578 -> 0.7680. Onset keeps its
threshold decoder (eIoU without eligibility loses on raw Feb: 0.8791 -> 0.872).

  python t2_eiou.py SRC_QUEUE PROBS OUT_QUEUE   (e.g. queue_pub_c8.pkl t2rec7gc8_queue_final_probs.pkl queue_pub_c8e.pkl)
"""
from __future__ import annotations

import pickle
import sys

import numpy as np

from common import OUT, ROOT
from t2_model import decode

DS = ROOT / "kaggle_pkg" / "src_ds"
A = 1.3


def eiou(P: np.ndarray, a: float = A, kmin: int = 1) -> np.ndarray:
    f = (P ** a).ravel()
    o = np.argsort(-f)
    c = np.cumsum(f[o])
    k = np.arange(1, len(f) + 1)
    kk = max(int(np.argmax(c / (k + f.sum() - c))) + 1, kmin)
    out = np.zeros(f.shape, bool)
    out[o[:kk]] = True
    return out.reshape(P.shape)


def main():
    src, probs, dst = sys.argv[1:4]
    q = pickle.load(open(DS / src, "rb"))
    P = pickle.load(open(OUT / probs, "rb"))
    same = changed = 0
    for wid, (p, order, cond) in P.items():
        if cond != "queue_ongoing":
            continue
        old = np.zeros_like(q[wid]); old[:, order] = decode(p, cond, 0.45)
        same += (old == q[wid]).all()
        new = np.zeros_like(q[wid]); new[:, order] = eiou(p)
        changed += not (new == q[wid]).all()
        q[wid] = new
    print(f"ongoing windows: thr-0.45 decode reproduces source on {same}; eIoU changes {changed}", flush=True)
    pickle.dump(q, open(DS / dst, "wb"))


if __name__ == "__main__":
    main()
