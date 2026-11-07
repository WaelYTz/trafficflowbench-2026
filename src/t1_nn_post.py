"""Neural Task 1 imputer (src/t1_nn.py, 4 nets: 2 Jun-Jan + 2 Jun-Feb) in submission-key order, plus the all-link gap
flag. Writes kaggle_pkg/src_ds/t1nn_post.npz with vn, qn (NaN off the state rows) and gap: True where every mainline link
of the panel is hidden in the split's masked layer at that slot (the blanked Task 2 windows; train has none, so the
GBDT never learned them: Feb synthetic gaps rmse_v 7.25 GBDT vs 5.01 NN). The assembler blends the NN into our state
and, on gap cells, into the final blend (subkey.write_submission state_nn)."""
from __future__ import annotations

import numpy as np

import common
import subkey
from common import CACHE, PANELS, ROOT


def main(src=ROOT / "out_t1nn" / "pred4"):
    if not (CACHE / "subkey.npz").exists():
        subkey.build()
    k = subkey.key()
    n = len(k["task"])
    vn = np.full(n, np.nan, np.float32); qn = np.full(n, np.nan, np.float32); gap = np.zeros(n, bool)
    for pi, p in enumerate(PANELS):
        for si, s in ((1, "validation"), (2, "private")):
            z = np.load(src / f"t1_{p}_{s}.npz")
            m = (k["task"] == 0) & (k["panel"] == pi) & (k["split"] == si)
            d, t, l = k["day"][m], k["slot"][m], k["link"][m]
            vn[m] = z["V"][d, t, l]; qn[m] = z["Q"][d, t, l]
            v = common.load(p, s, "v")                         # masked layer of the scored split
            allgap = ~np.isfinite(v).any(axis=2)              # (day, slot): no link observed
            gap[m] = allgap[d, t]
            print(p, s, int(m.sum()), f"nn finite {np.isfinite(vn[m]).mean():.4f} gap {gap[m].mean():.4f}", flush=True)
    np.savez_compressed(ROOT / "kaggle_pkg" / "src_ds" / "t1nn_post.npz", vn=vn, qn=qn, gap=gap)


if __name__ == "__main__":
    main()
