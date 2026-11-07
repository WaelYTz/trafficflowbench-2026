"""Queue files mixing rec7g with the public anubisyy queue models on the scored windows.

The public yeash notebook found on the LB that a 50/50 average of ongoing probabilities with the public
LightGBM (threshold 0.45) beat either model, and that the public onset model localises first breakdowns
well on the March windows. Our pseudo-windows had ranked the public ongoing model lower, but they also
mis-ranked the CNN-only onset, so these are LB tests. Public models read only the 12 history steps
before T (causal).

Writes queue_<onset>_<ongoing>.pkl into kaggle_pkg/src_ds for the assemble kernel.
"""
from __future__ import annotations

import pickle

import numpy as np

from common import OUT, QUEUE_PANELS, ROOT
from t2_model import decode
from t2_pub import Public, history_scored

DS = ROOT / "kaggle_pkg" / "src_ds"


def main():
    ours = pickle.load(open(OUT / "t2rec7g2_queue_final_probs.pkl", "rb"))       # wid -> (P travel order, order, cond)
    t2f = pickle.load(open(OUT / "t2f_queue_final.pkl", "rb"))                    # wid -> (6, L) scorer order
    pub = Public()
    pp = {}
    for p in QUEUE_PANELS:
        for split in ("validation", "private"):
            for wid, (V, Q, E, origin) in history_scored(p, split).items():
                pp[wid] = pub.probs(p, V, Q, E, origin, ours[wid][2])
    pickle.dump(pp, open(OUT / "t2pub_scored_probs.pkl", "wb"))

    def scorer(wid, pred):
        out = np.zeros_like(pred, dtype=np.int8)
        out[:, ours[wid][1]] = pred
        return out

    onset = {"g": lambda wid, P, Pp: scorer(wid, decode(P, "queue_onset", 0.3, 3)),
             "pub": lambda wid, P, Pp: scorer(wid, decode(Pp, "queue_onset", 0.55, 1)),
             "omix": lambda wid, P, Pp: scorer(wid, decode(0.5 * P + 0.5 * Pp, "queue_onset", 0.4, 2)),
             "t2f": lambda wid, P, Pp: t2f[wid].astype(np.int8)}
    ongoing = {"g": lambda wid, P, Pp: scorer(wid, decode(P, "queue_ongoing", 0.45)),
               "mix": lambda wid, P, Pp: scorer(wid, decode(0.5 * P + 0.5 * Pp, "queue_ongoing", 0.45))}
    # The ongoing mix lost 0.02 on the raw-history Feb windows (out/raw_feb_ongoing_pub2.log), so the queue files
    # for the LB keep our ongoing. queue_t2fp_g (t2f onset exactly as probed at 0.728) is cut from that probe file.
    combos = [("g", "g"), ("pub", "g"), ("omix", "g"), ("g", "mix")]
    ref = pickle.load(open(OUT / "t2rec7g_queue_final.pkl", "rb"))
    for on, og in combos:
        q = {}
        for wid, (P, order, cond) in ours.items():
            f = onset[on] if cond == "queue_onset" else ongoing[og]
            q[wid] = f(wid, P, pp[wid])
        pickle.dump(q, open(DS / f"queue_{on}_{og}.pkl", "wb"))
        val = [w for w in q if "validation" in w]
        on_links = np.mean([q[w][5].sum() for w in val if ours[w][2] == "queue_onset"])
        og_cells = np.mean([q[w].sum() for w in val if ours[w][2] == "queue_ongoing"])
        same = sum((q[w] == ref[w]).all() for w in q)
        print(f"queue_{on}_{og}: March onset links/window {on_links:.2f}, ongoing cells/window {og_cells:.1f}, "
              f"identical to rec7g {same}/{len(q)}", flush=True)


if __name__ == "__main__":
    main()
