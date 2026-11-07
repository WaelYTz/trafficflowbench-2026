"""Post-hoc free-flow speed correction for Task 1 (offline imputation: reads only the scored split's own masked
layer). Observed free-flow speed is generated as level_l * (1 + c_t + e) with c_t one shock per (day, slot)
shared by the whole corridor, so at a hidden free-flowing cell the best guess is E[min(120.75, level_l (1 + c_t)
(1 + e))], with level_l and c_t read from the same month's / same timestamp's visible cells
(t1_features.shock_grid). Writes t1_shock_ps.npz with, in submission-key order, ps (NaN off the state rows)
and gate = 0.92 vf: the assembler blends final = (1 - w) blend + w ps where blend > gate.

March pseudo-test on our Jun-Jan model (out/audit_shock_gain_mar.log): S_state 0.94054 -> 0.94350 at w 0.8.
"""
from __future__ import annotations

import numpy as np

import subkey
from common import CACHE, PANELS, ROOT
from t1_features import Grid, shock_grid


def main():
    if not (CACHE / "subkey.npz").exists():
        subkey.build()
    k = subkey.key()
    n = len(k["task"])
    ps = np.full(n, np.nan, np.float32)
    gate = np.full(n, np.inf, np.float32)
    for pi, p in enumerate(PANELS):
        for si, s in ((1, "validation"), (2, "private")):
            g = Grid(p, s, use_masked=False)
            _, _, psn = shock_grid(g)
            m = (k["task"] == 0) & (k["panel"] == pi) & (k["split"] == si)
            d, t, sc = k["day"][m], k["slot"][m], k["link"][m]
            l = g.rank[sc]                                   # scorer index -> travel position
            ps[m] = psn[d, t, l] * g.vf[l]
            gate[m] = 0.92 * g.vf[l]
            print(p, s, int(m.sum()), f"ps mean {np.nanmean(ps[m]):.2f} vs vf {g.vf[l].mean():.2f}", flush=True)
    np.savez_compressed(ROOT / "kaggle_pkg" / "src_ds" / "t1_shock_ps.npz", ps=ps, gate=gate)


if __name__ == "__main__":
    main()
