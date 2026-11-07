"""Onset regime gate: the public anubisyy onset by default, our memory onset (rec7g) where the window's recent
onset memory disagrees with the long-term train place prior, i.e. where the active bottleneck has moved.

gate = cosine(recent_onset, onset_prior) over the window's links (recent_onset: onsets in the last 7 days of
the masked layer at that hour, causal; empty memory counts as agreement). gate < 0.8 -> rec7g (0.3, top-3),
else pub (0.55, top-1).

Evidence (out/onset_gate*.log): on March pseudo-windows with Task 1-filled history (new regime; memory fresh)
pub 0.715, rec7g 0.822, gate 0.810, with the gain on the panels whose bottlenecks moved (D7_I405_S, D12_I5_S);
on the raw-history Feb windows (old regime) pub 0.879, gate 0.870. The gate changes 2/40 public and 9/40 private
onset windows vs pub. March data is evidence for the April windows only; every input of a window is <= T.

Writes queue_<name>_<ongoing>.pkl (onset replaced in queue_pub_<ongoing>.pkl) and probeq_<name>.pkl into
kaggle_pkg/src_ds. Usage: t2_gate.py [TAU NAME] (default 0.8 gate). At 0.9 the private windows are the same 9,
while 8 public windows change instead of 2, so the public LB tests the mechanism on more windows.
"""
from __future__ import annotations

import pickle
import sys

import numpy as np

import t2_data
from common import OUT, QUEUE_PANELS, ROOT
from t2_data import scored_windows
from t2_model import decode

DS = ROOT / "kaggle_pkg" / "src_ds"
TAU = 0.8


def gate(X: np.ndarray, names: list[str]) -> float:
    a, b = X[:, names.index("recent_onset")], X[:, names.index("onset_prior")]
    if a.sum() == 0:
        return 1.0
    if b.sum() == 0:
        return 0.0
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def main():
    global TAU
    name = "gate"
    if len(sys.argv) > 2:
        TAU, name = float(sys.argv[1]), sys.argv[2]
    t2_data.RECENT_DAYS = 7
    rec = pickle.load(open(OUT / "t2rec7g2_queue_final_probs.pkl", "rb"))     # wid -> (P travel order, order, cond)
    pub = pickle.load(open(OUT / "t2pub_scored_probs.pkl", "rb"))             # wid -> P travel order
    onset, fired = {}, []
    for p in QUEUE_PANELS:
        for split in ("validation", "private"):
            blocks, names = scored_windows(p, split)
            for b in blocks:
                if b["condition"] != "queue_onset":
                    continue
                wid = b["window_id"]
                g = gate(b["X"], names)
                pred = decode(rec[wid][0], "queue_onset", 0.3, 3) if g < TAU else decode(pub[wid], "queue_onset", 0.55, 1)
                out = np.zeros_like(pred, dtype=np.int8)
                out[:, rec[wid][1]] = pred
                onset[wid] = out
                if g < TAU:
                    fired.append((wid, round(g, 2)))
    ref = pickle.load(open(DS / "queue_pub_c8.pkl", "rb"))
    changed = [w for w in onset if not (onset[w] == ref[w]).all()]
    print(f"gate < {TAU} on {len(fired)} windows; changed vs pub onset: "
          f"{sum('validation' in w for w in changed)} public, {sum('private' in w for w in changed)} private", flush=True)
    print("changed:", changed, flush=True)
    for og in ("c8", "g", "nm"):
        f = DS / f"queue_pub_{og}.pkl"
        if f.exists():
            q = pickle.load(open(f, "rb"))
            q.update(onset)
            pickle.dump(q, open(DS / f"queue_{name}_{og}.pkl", "wb"))
    pickle.dump(onset, open(DS / f"probeq_{name}.pkl", "wb"))


if __name__ == "__main__":
    main()
