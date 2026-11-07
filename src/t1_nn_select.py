"""Pick the Feb-best neural-imputer ensembles and build out_t1nn/pred8 (validation/private, cache layout).

Families (Jun-Jan nets scored on the Feb holdout -> prediction folder with their Jun-Jan + Jun-Feb nets):
  old: c48, c48s3        -> pred4     (width 48, 8 epochs; current submission)
  d64: d64h              -> pred_d64  (width 64, 10 epochs, V2 channels: ramp demand ratio + congestion frequency)
  g48: g48h              -> pred_g48  (gap specialist: synthetic all-link gaps on every day, gap loss x2, V2)
Normal cells: family weights in {0,1,2}, scored as the best blend with the local GBDT (speed and flow weights separately,
the S_state parts are separable). All-link gap cells (the deployed blend takes 100% NN there): NN alone on the
synthetic-gap cells. Composite = gap cells at 1.9% of targets (queue panels), as in validation/private."""
from __future__ import annotations

import itertools
import json

import numpy as np

from common import FAMILY, PANELS, ROOT, load

OUTD = ROOT / "out_t1nn"
FAM = {"old": ["c48", "c48s3"], "d64": ["d64h"], "g48": ["g48h"]}
PRED = {"old": "pred4", "d64": "pred_d64", "g48": "pred_g48"}
F_GAP = 0.019 * 10 / 8
QP = np.array([p not in ("D12_I405_N", "D12_I405_S") for p in PANELS])


def complete(f):
    ok = all((OUTD / f"feb_nn_{t}.npz").exists() and (OUTD / f"feb_nn_gap_{t}.npz").exists() for t in FAM[f])
    return ok and len(list((OUTD / PRED[f]).glob("t1_*.npz"))) == 20


class Scorer:
    """Official S_state structure with per (panel, regime) MSEs from bincount (fast)."""

    def __init__(self, ref, m=None):
        m = np.ones(len(ref["yv"]), bool) if m is None else m
        self.g = (ref["panel"][m] * 3 + ref["regime"][m]).astype(int)
        self.n = np.bincount(self.g, minlength=30).astype(float)
        self.yv, self.yq, self.lanes, self.m = ref["yv"][m], ref["yq"][m], ref["lanes"][m], m

    def mse(self, pv, pq):
        pv, pq = pv[self.m], pq[self.m]
        with np.errstate(invalid="ignore"):
            return (np.bincount(self.g, (pv - self.yv) ** 2, 30) / self.n,
                    np.bincount(self.g, ((pq - self.yq) / self.lanes) ** 2, 30) / self.n)


def S_from(mv, mq, parts=(1, 1)):
    """S_state from per-group MSEs; parts selects speed / flow parts (for separable weight search)."""
    per = {}
    for p in range(len(PANELS)):
        s = []
        for r in range(3):
            k = p * 3 + r
            if np.isfinite(mv[k]):
                s.append(parts[0] * 0.54 * max(0, 1 - np.sqrt(mv[k]) / 25) + parts[1] * 0.46 * max(0, 1 - np.sqrt(mq[k]) / 600))
        if s:
            per[PANELS[p]] = np.mean(s)
    fam = {}
    for p, s in per.items():
        fam.setdefault(FAMILY[p], []).append(s)
    return float(np.mean([np.mean(v) for v in fam.values()]))


def main():
    ref = dict(np.load(OUTD / "feb_gbdt.npz")); gref = dict(np.load(OUTD / "feb_gbdt_gap.npz"))
    gm = gref["ingap"].astype(bool)
    fams = [f for f in FAM if complete(f)]
    print("families available:", fams, flush=True)
    P, G = {}, {}
    for f in fams:
        c = [dict(np.load(OUTD / f"feb_nn_{t}.npz")) for t in FAM[f]]; g = [dict(np.load(OUTD / f"feb_nn_gap_{t}.npz")) for t in FAM[f]]
        P[f] = (np.mean([x["pv"] for x in c], 0), np.mean([x["pq"] for x in c], 0))
        G[f] = (np.mean([x["pv"] for x in g], 0), np.mean([x["pq"] for x in g], 0))
    sc, sg = Scorer(ref), Scorer(gref, gm)
    ws = np.round(np.arange(0, 1.0001, 0.05), 2)
    gb = sc.mse(ref["pv"], ref["pq"])
    print(f"GBDT clean S {S_from(*gb):.5f}; GBDT gap-cell S {S_from(*sg.mse(gref['pv'], gref['pq'])):.5f}", flush=True)

    def mix(D, w):
        t = sum(w.values())
        return (sum(w[f] * D[f][0] for f in w) / t, sum(w[f] * D[f][1] for f in w) / t)

    def best_blend(pv, pq):
        bv = max(ws, key=lambda a: S_from(*sc.mse(a * pv + (1 - a) * ref["pv"], pq), parts=(1, 0)))
        bq = max(ws, key=lambda a: S_from(*sc.mse(pv, a * pq + (1 - a) * ref["pq"]), parts=(0, 1)))
        return bv, bq, S_from(*sc.mse(bv * pv + (1 - bv) * ref["pv"], bq * pq + (1 - bq) * ref["pq"]))

    rows_n, rows_g = [], []
    for combo in itertools.product((0, 1, 2), repeat=len(fams)):
        if not any(combo) or (max(combo) == 2 and min(c for c in combo if c) == 2):
            continue                                            # (2,2) duplicates (1,1)
        w = {f: c for f, c in zip(fams, combo) if c}
        pv, pq = mix(P, w)
        bv, bq, sb = best_blend(pv, pq)
        rows_n.append((sb, S_from(*sc.mse(pv, pq)), w, bv, bq))
        gv, gq = mix(G, w)
        mv, mq = sg.mse(gv, gq)
        rows_g.append((S_from(mv, mq), w, float(np.sqrt(np.nanmean((gv[gm] - gref["yv"][gm]) ** 2))),
                       float(np.sqrt(np.nanmean(((gq[gm] - gref["yq"][gm]) / gref["lanes"][gm]) ** 2)))))
    rows_n.sort(key=lambda r: -r[0]); rows_g.sort(key=lambda r: -r[0])
    print("\nNORMAL cells (clean Feb holdout): S blended with GBDT at best (w_v, w_q) | NN alone", flush=True)
    for r in rows_n:
        print(f"  {str(r[2]):40s} blend {r[0]:.5f} (w_v {r[3]:.2f}, w_q {r[4]:.2f}) | NN alone {r[1]:.5f}")
    print("\nGAP cells (synthetic all-link gaps, NN alone): S | rmse_v | rmse_q_lane", flush=True)
    for r in rows_g:
        print(f"  {str(r[1]):40s} {r[0]:.5f} | {r[2]:.3f} | {r[3]:.2f}")
    bn, bg = rows_n[0], rows_g[0]
    base_n = next((r for r in rows_n if r[2] == {"old": 1}), None)
    base_g = next((r for r in rows_g if r[1] == {"old": 1}), None)

    def composite(wn, wv, wq, wg):
        pv, pq = mix(P, wn); gv, gq = mix(G, wg)
        a = sc.mse(wv * pv + (1 - wv) * ref["pv"], wq * pq + (1 - wq) * ref["pq"])
        b = sg.mse(gv, gq)
        f = np.where(np.repeat(QP, 3) & np.isfinite(b[0]), F_GAP, 0.0)
        return S_from((1 - f) * a[0] + f * np.nan_to_num(b[0]), (1 - f) * a[1] + f * np.nan_to_num(b[1]))

    res = dict(normal=bn[2], gap=bg[1], w_v=bn[3], w_q=bn[4], S_clean_blend=bn[0], S_clean_nn=bn[1], S_gap=bg[0],
               gap_rmse_v=bg[2], gap_rmse_q=bg[3], composite=composite(bn[2], bn[3], bn[4], bg[1]))
    if base_n and base_g:
        res.update(old_w_v=base_n[3], old_w_q=base_n[4], old_S_clean_blend=base_n[0], old_S_gap=base_g[0],
                   old_gap_rmse_v=base_g[2], old_gap_rmse_q=base_g[3],
                   old_composite=composite({"old": 1}, base_n[3], base_n[4], {"old": 1}))
    print("\nSELECTED:", json.dumps(res, indent=1, default=float), flush=True)
    (OUTD / "select.json").write_text(json.dumps(res, indent=1, default=float))

    # pred8: normal cells from the normal ensemble, all-link gap cells from the gap ensemble
    od = OUTD / "pred8"
    od.mkdir(exist_ok=True)
    for p in PANELS:
        for s in ("validation", "private"):
            Z = {f: np.load(OUTD / PRED[f] / f"t1_{p}_{s}.npz") for f in set(bn[2]) | set(bg[1])}
            out = {}
            for k in ("V", "Q"):
                nrm = sum(bn[2][f] * Z[f][k] for f in bn[2]) / sum(bn[2].values())
                gp = sum(bg[1][f] * Z[f][k] for f in bg[1]) / sum(bg[1].values())
                allgap = ~np.isfinite(load(p, s, "v")).any(axis=2)          # (day, slot): every mainline link hidden
                out[k] = np.where(allgap[:, :, None], gp, nrm).astype(np.float32)
            np.savez_compressed(od / f"t1_{p}_{s}.npz", **out)
            print(f"pred8 {p} {s}: gap slots {int(allgap.sum())}", flush=True)


if __name__ == "__main__":
    main()
