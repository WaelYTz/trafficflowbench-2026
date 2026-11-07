"""Queue files with the onset of a retrained public-recipe bag (t2_pubbag.py final [--tag T]) and our ongoing
variants: queue_<name>_<ongoing>.pkl in kaggle_pkg/src_ds. Usage: t2_pubbag_queue.py NAME TAG THR"""
from __future__ import annotations

import pickle
import sys

import numpy as np

from common import OUT, ROOT
from t2_model import decode

DS = ROOT / "kaggle_pkg" / "src_ds"


def main():
    name, tag, thr = sys.argv[1], sys.argv[2], float(sys.argv[3])
    sfx = f"_{tag}" if tag else ""
    bag = pickle.load(open(OUT / f"pubbag_scored_probs{sfx}.pkl", "rb"))            # wid -> (6, L) travel order
    order = {w: v[1] for w, v in pickle.load(open(OUT / "t2rec7g2_queue_final_probs.pkl", "rb")).items()}
    onset = {}
    for wid, P in bag.items():
        out = np.zeros((6, len(order[wid])), dtype=np.int8)
        out[:, order[wid]] = decode(P, "queue_onset", thr, 1)
        onset[wid] = out
    pub = pickle.load(open(DS / "queue_pub_g.pkl", "rb"))
    same = sum((onset[w] == pub[w]).all() for w in onset)
    links = np.mean([onset[w].sum() for w in onset if "validation" in w])
    print(f"{name}: {len(onset)} onset windows, identical to public onset {same}, March links/window {links:.2f}", flush=True)
    for og in ("c8", "nm", "g"):
        f = DS / f"queue_pub_{og}.pkl"
        if f.exists():
            q = pickle.load(open(f, "rb"))
            q.update(onset)
            pickle.dump(q, open(DS / f"queue_{name}_{og}.pkl", "wb"))
    pickle.dump(onset, open(DS / f"probeq_{name}.pkl", "wb"))


if __name__ == "__main__":
    main()
