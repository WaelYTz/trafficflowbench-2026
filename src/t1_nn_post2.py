"""Same as t1_nn_post.py (submission-key order vn / qn / gap flag) for another prediction folder, written to
kaggle_pkg/src_ds/t1nn_post2.npz (t1nn_post.npz stays untouched).  python t1_nn_post2.py [pred8]"""
from __future__ import annotations

import sys

import numpy as np

import common
import subkey
from common import CACHE, PANELS, ROOT


def main(src=ROOT / "out_t1nn" / "pred8", dst=ROOT / "kaggle_pkg" / "src_ds" / "t1nn_post2.npz"):
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
            allgap = ~np.isfinite(common.load(p, s, "v")).any(axis=2)
            gap[m] = allgap[d, t]
            print(p, s, int(m.sum()), f"nn finite {np.isfinite(vn[m]).mean():.4f} gap {gap[m].mean():.4f}", flush=True)
    st = k["task"] == 0
    print(f"state rows {int(st.sum()):,}, non-finite NN v {int((~np.isfinite(vn[st])).sum())} q {int((~np.isfinite(qn[st])).sum())}", flush=True)
    np.savez_compressed(dst, vn=vn, qn=qn, gap=gap)
    print("wrote", dst, flush=True)


if __name__ == "__main__":
    main(ROOT / "out_t1nn" / sys.argv[1]) if len(sys.argv) > 1 else main()
