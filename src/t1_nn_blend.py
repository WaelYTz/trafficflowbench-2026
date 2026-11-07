"""Feb holdout: GBDT (t1_nn_gbdt_feb.py) vs neural imputer (t1_nn.py) vs blends, official S_state rule.
  python t1_nn_blend.py h48 [h64 ...]   (several tags are averaged as an NN ensemble)
  python t1_nn_blend.py --gap h48        (synthetic Task 2 gaps)"""
import sys

import numpy as np

from common import ROOT
from t1_nn import report, s_state

OUTD = ROOT / "out_t1nn"
gap = "--gap" in sys.argv
tags = [x for x in sys.argv[1:] if not x.startswith("--")]
ref = dict(np.load(OUTD / ("feb_gbdt_gap.npz" if gap else "feb_gbdt.npz")))
nn = [dict(np.load(OUTD / (f"feb_nn_gap_{t}.npz" if gap else f"feb_nn_{t}.npz"))) for t in tags]
pv_n = np.mean([x["pv"] for x in nn], 0); pq_n = np.mean([x["pq"] for x in nn], 0)
for t, x in zip(tags, nn):
    report(f"NN {t}", x["pv"], x["pq"], ref)
report("NN ens" if len(nn) > 1 else "NN", pv_n, pq_n, ref)
report("GBDT", ref["pv"], ref["pq"], ref)
print("err corr v %.3f q %.3f" % (np.corrcoef(pv_n - ref["yv"], ref["pv"] - ref["yv"])[0, 1],
                                  np.corrcoef(pq_n - ref["yq"], ref["pq"] - ref["yq"])[0, 1]))
args = lambda wv, wq, m=slice(None): ((wv * pv_n + (1 - wv) * ref["pv"])[m], (wq * pq_n + (1 - wq) * ref["pq"])[m],
                                      ref["yv"][m], ref["yq"][m], ref["lanes"][m], ref["panel"][m], ref["regime"][m])
ws = np.round(np.arange(0, 1.0001, 0.05), 2)
grid = {(wv, wq): s_state(*args(wv, wq)) for wv in ws for wq in ws}
best = max(grid, key=grid.get)
same = {w: grid[(w, w)] for w in ws}
bs = max(same, key=same.get)
print("single weight: " + " ".join(f"{w:.2f}:{s:.5f}" for w, s in same.items()))
print(f"best single w_nn {bs:.2f}: S {same[bs]:.5f} (GBDT {same[0.0]:.5f}, NN {same[1.0]:.5f}, gain {same[bs] - same[0.0]:+.5f})")
print(f"best (w_v, w_q) = {best}: S {grid[best]:.5f} (gain {grid[best] - same[0.0]:+.5f})")
report(f"blend w={bs:.2f}", bs * pv_n + (1 - bs) * ref["pv"], bs * pq_n + (1 - bs) * ref["pq"], ref)
report(f"blend {best}", best[0] * pv_n + (1 - best[0]) * ref["pv"], best[1] * pq_n + (1 - best[1]) * ref["pq"], ref)
# Fit-half / test-half check of the weight (odd vs even days)
for half in (0, 1):
    fit, tst = (ref["d"] % 2) == half, (ref["d"] % 2) != half
    g = {w: s_state(*args(w, w, fit)) for w in ws}
    w = max(g, key=g.get)
    print(f"half {half}: w fit {w:.2f} -> test gain {s_state(*args(w, w, tst)) - s_state(*args(0, 0, tst)):+.5f}")
if gap and "ingap" in ref:
    m = ref["ingap"].astype(bool)
    for name, pv, pq in (("NN", pv_n, pq_n), ("GBDT", ref["pv"], ref["pq"]),
                         (f"blend {bs:.2f}", bs * pv_n + (1 - bs) * ref["pv"], bs * pq_n + (1 - bs) * ref["pq"])):
        rv = np.sqrt(np.mean((pv[m] - ref["yv"][m]) ** 2)); rq = np.sqrt(np.mean(((pq[m] - ref["yq"][m]) / ref["lanes"][m]) ** 2))
        rv2 = np.sqrt(np.mean((pv[~m] - ref["yv"][~m]) ** 2)); rq2 = np.sqrt(np.mean(((pq[~m] - ref["yq"][~m]) / ref["lanes"][~m]) ** 2))
        print(f"{name:12s} gap cells rmse_v {rv:.3f} rmse_q {rq:.2f} (n={m.sum():,}) | other cells {rv2:.3f} / {rq2:.2f}")
    gw = {w: np.mean([s_state(*((w * pv_n + (1 - w) * ref["pv"])[m], (w * pq_n + (1 - w) * ref["pq"])[m], ref["yv"][m],
                               ref["yq"][m], ref["lanes"][m], ref["panel"][m], ref["regime"][m]))]) for w in ws}
    print("gap-cell S by w_nn: " + " ".join(f"{w:.2f}:{s:.4f}" for w, s in gw.items()))


def composite(tags, f_gap=0.019 * 10 / 8):
    """Deployment-like Feb estimate: clean holdout cells plus synthetic-gap cells weighted as the share of scored
    validation/private targets inside Task 2 gaps (1.9% overall, queue panels only). Separate NN weights for gap
    cells (known at prediction time: every link of the slot hidden) and the rest."""
    from common import FAMILY, PANELS
    c = dict(np.load(OUTD / "feb_gbdt.npz")); gr = dict(np.load(OUTD / "feb_gbdt_gap.npz"))
    nc = [dict(np.load(OUTD / f"feb_nn_{t}.npz")) for t in tags]; ng = [dict(np.load(OUTD / f"feb_nn_gap_{t}.npz")) for t in tags]
    cv, cq = np.mean([x["pv"] for x in nc], 0), np.mean([x["pq"] for x in nc], 0)
    gv, gq = np.mean([x["pv"] for x in ng], 0), np.mean([x["pq"] for x in ng], 0)
    gm = gr["ingap"].astype(bool)

    def mse(pv, pq, ref, m):
        return np.mean((pv[m] - ref["yv"][m]) ** 2), np.mean(((pq[m] - ref["yq"][m]) / ref["lanes"][m]) ** 2)

    def S(w, wg):
        per = {}
        for p in range(len(PANELS)):
            s = []
            for r in range(3):
                mc = (c["panel"] == p) & (c["regime"] == r)
                mg = gm & (gr["panel"] == p) & (gr["regime"] == r)
                if not mc.any():
                    continue
                a = mse(w * cv + (1 - w) * c["pv"], w * cq + (1 - w) * c["pq"], c, mc)
                f = f_gap if PANELS[p] not in ("D12_I405_N", "D12_I405_S") and mg.any() else 0.0
                if f:
                    b = mse(wg * gv + (1 - wg) * gr["pv"], wg * gq + (1 - wg) * gr["pq"], gr, mg)
                    a = ((1 - f) * a[0] + f * b[0], (1 - f) * a[1] + f * b[1])
                s.append(0.54 * max(0, 1 - np.sqrt(a[0]) / 25) + 0.46 * max(0, 1 - np.sqrt(a[1]) / 600))
            per[PANELS[p]] = np.mean(s)
        fam = {}
        for p, s in per.items():
            fam.setdefault(FAMILY[p], []).append(s)
        return float(np.mean([np.mean(v) for v in fam.values()]))

    ws = np.round(np.arange(0, 1.0001, 0.1), 2)
    base = S(0, 0)
    g = {(w, wg): S(w, wg) for w in ws for wg in ws}
    b = max(g, key=g.get)
    print(f"composite (gap share {f_gap:.3f}): GBDT {base:.5f}, NN {S(1, 1):.5f}, best (w, w_gap) {b}: {g[b]:.5f} "
          f"(gain {g[b] - base:+.5f}); same w: " + " ".join(f"{w:.1f}:{g[(w, w)]:.5f}" for w in ws))


if not gap and all((OUTD / f"feb_nn_gap_{t}.npz").exists() for t in tags) and (OUTD / "feb_gbdt_gap.npz").exists():
    composite(tags)
