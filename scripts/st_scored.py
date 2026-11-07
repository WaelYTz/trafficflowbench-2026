"""Space-time net probabilities for the 160 scored windows (validation + private), blended into a yr queue.

Ongoing windows: P = (1 - W) P_yr + W P_st, P_yr from <yr>_probs.pkl (t2_pipeline output, travel order), P_st the mean
of the ST seeds on scored_windows blocks + grids (t2_stnet.attach_grids). Onset windows keep the yr queue. Writes
out_st/st_scored_probs.pkl (window_id -> (6, L) travel order) and out_st/<out> in t2_pipeline's queue format.

  TFB_RAMPS=1 [TFB_FILL_BLANKS=1] python scripts/st_scored.py --yr out_yr/t2yr_queue_final.pkl --w 0.5 --thr 0.5 \
      --tag st --out t2yr_st50_queue_final.pkl
"""
import argparse, os, pickle, sys
assert os.environ.get("TFB_RAMPS") == "1"
import numpy as np, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
import t2_data, t2_stnet
from common import QUEUE_PANELS, ROOT
from t2_data import scored_windows
from t2_model import decode
t2_data.RECENT_DAYS = 7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yr", default="out_yr/t2yr_queue_final.pkl")
    ap.add_argument("--tag", nargs="+", default=["st"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 23, 31, 47])
    ap.add_argument("--w", type=float, default=0.5)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--out", default="t2yr_st_queue_final.pkl")
    ap.add_argument("--probs-out", default="st_scored_probs.pkl")
    a = ap.parse_args()
    yq = pickle.load(open(ROOT / a.yr, "rb")); yp = pickle.load(open(ROOT / a.yr.replace(".pkl", "_probs.pkl"), "rb"))
    bundles = [torch.load(ROOT / "out_st" / f"{t}_ongoing_s{s}.pt", weights_only=False) for t in a.tag for s in a.seeds
               if (ROOT / "out_st" / f"{t}_ongoing_s{s}.pt").exists()]
    print(f"{len(bundles)} ST bundles", flush=True)
    models = [t2_stnet.load(b) for b in bundles]
    pst, queue = {}, dict(yq)
    for p in QUEUE_PANELS:
        for split in ("validation", "private"):
            blocks, names = scored_windows(p, split)
            assert names == bundles[0]["names"]
            og = [b for b in blocks if b["condition"] == "queue_ongoing"]
            if not og:
                continue
            t2_stnet.attach_grids(p, split, og, names)
            P = np.mean([t2_stnet.predict(b, og, model=m) for b, m in zip(bundles, models)], axis=0)
            for b, ps in zip(og, P):
                Py, order, cond = yp[b["window_id"]]
                assert (np.asarray(order) == b["order"]).all() and cond == "queue_ongoing"
                pst[b["window_id"]] = ps.astype(np.float32)
                pred = decode((1 - a.w) * Py + a.w * ps, cond, a.thr)
                out = np.zeros_like(pred, dtype=np.int8); out[:, b["order"]] = pred
                queue[b["window_id"]] = out
            print(p, split, len(og), "ongoing windows", flush=True)
    pickle.dump(pst, open(ROOT / "out_st" / a.probs_out, "wb"))
    pickle.dump(queue, open(ROOT / "out_st" / a.out, "wb"))
    same = sum((queue[k] == yq[k]).all() for k in queue)
    print(f"{len(queue)} windows ({len(pst)} ongoing re-blended); identical to yr: {same}", flush=True)


if __name__ == "__main__":
    main()
