"""Post-T deletion check for the space-time ongoing inputs (v1 path of st_scored.py and the v2 context grid):
for each scored ONGOING window, blank every released masked-layer / ramp cell stamped after its origin T (rest of the
day, later days, later splits on train -> March -> April; as src/t2_post_t_check.py), rebuild from scratch
  X  (t2_data.scored_windows; compared on all columns and on the kr = non-LONG columns the ST nets use),
  G  (t2_stnet.attach_grids: window_history + masked row T + ramps T-12..T),
  C  (t2_stnet2.attach_ctx: masked layer + ramps at T-36..T-13),
and assert bit-identity with the inputs built from the full release. Negative control: cutting at T-20 instead must
change C and G.

  TFB_RAMPS=1 TFB_FILL_BLANKS=1 python scripts/st2_post_t_check.py [N_PER_PANEL_SPLIT]
"""
import os, sys, time
assert os.environ.get("TFB_RAMPS") == "1" and os.environ.get("TFB_FILL_BLANKS") == "1"
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
import common, t2_data, t2_stnet, t2_stnet2
from common import QUEUE_PANELS, days
from t2_data import scored_windows
from t2_pipeline import LONG
from t2_post_t_check import truncated_load
t2_data.RECENT_DAYS = 7
_load = common.load
same = lambda a, b: np.array_equal(np.nan_to_num(np.asarray(a, float), nan=-9e9), np.nan_to_num(np.asarray(b, float), nan=-9e9))


def rebuild(p, split, wid, cut):
    common.load = t2_data.load = truncated_load(split, *cut)
    try:
        blocks, names = scored_windows(p, split)
        b = [x for x in blocks if x["window_id"] == wid]
        t2_stnet.attach_grids(p, split, b, names)
        t2_stnet2.attach_ctx(p, split, b)
    finally:
        common.load = t2_data.load = _load
    return b[0]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    t0 = time.time()
    cnt = dict(checked=0, x=0, xkr=0, g=0, c=0)
    neg_done = False
    for p in QUEUE_PANELS:
        for split in ("validation", "private"):
            ref, names = scored_windows(p, split)
            kr = [i for i, nm in enumerate(names) if not nm.startswith(LONG)]
            og = [b for b in ref if b["condition"] == "queue_ongoing"]
            if not og:
                continue
            t2_stnet.attach_grids(p, split, og, names)
            t2_stnet2.attach_ctx(p, split, og)
            wi = pd.read_csv(common.REL / "task2" / p / split / "window_index.csv").set_index("window_id")
            dl = {d: i for i, d in enumerate(days(p, split))}
            for k, b in enumerate(og):
                if n is not None and k >= n:
                    break
                o = str(wi.loc[b["window_id"], "forecast_origin"])
                d, slot = dl[o[:10]], int(o[11:13]) * 12 + int(o[14:16]) // 5
                nb = rebuild(p, split, b["window_id"], (d, slot))
                r = dict(x=same(b["X"], nb["X"]), xkr=same(b["X"][:, kr], nb["X"][:, kr]), g=same(b["G"], nb["G"]), c=same(b["C"], nb["C"]))
                cnt["checked"] += 1
                for key, ok in r.items():
                    cnt[key] += ok
                if not all(r.values()):
                    print(f"DIFF {b['window_id']}: {r}", flush=True)
                if not neg_done:                                   # negative control: cut 20 slots before T
                    nn_ = rebuild(p, split, b["window_id"], (d, slot - 20))
                    print(f"negative control {b['window_id']} (cut at T-20): C identical {same(b['C'], nn_['C'])}, "
                          f"G identical {same(b['G'], nn_['G'])}, X identical {same(b['X'], nn_['X'])}", flush=True)
                    neg_done = True
            print(f"{p} {split}: {cnt} ({time.time() - t0:.0f}s)", flush=True)
    print(f"DONE: {cnt['checked']} ongoing windows; identical X {cnt['x']}, X[kr] {cnt['xkr']}, G {cnt['g']}, C {cnt['c']} "
          f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
