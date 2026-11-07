"""Train the space-time ongoing net (src/t2_stnet.py) on Jun-Jan, best epoch by Feb loss, from out_st/rows/.

  python scripts/st_train.py --seeds 7 11 23 [--tag st] [--epochs 24] [--tw 32] [--no-x]
Bundles: out_st/<tag>_ongoing_s<seed>.pt; also saves Feb val probabilities (out_st/<tag>_feb_s<seed>.npz).
"""
import argparse, json, sys, time
import numpy as np, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
import t2_stnet
from common import QUEUE_PANELS, ROOT
from t2_model import HOLD
from t2_pipeline import LONG
D = ROOT / "out_st" / "rows"


def load_packs():
    fit, val, names = [], [], None
    for p in QUEUE_PANELS:
        meta = json.loads((D / f"meta_{p}.json").read_text())
        names = names or meta["names"]
        assert meta["names"] == names
        X = np.load(D / f"X_{p}.npy", mmap_mode="r"); G = np.load(D / f"G_{p}.npy", mmap_mode="r")
        y = np.load(D / f"y_{p}.npy"); e = np.load(D / f"e_{p}.npy")
        dd = np.array(meta["day"]); nf = int((dd < HOLD).sum())
        assert (dd[:nf] < HOLD).all() and (dd[nf:] >= HOLD).all()
        first = np.array(meta["first"])
        for lst, sl in ((fit, slice(0, nf)), (val, slice(nf, len(dd)))):
            lst.append(dict(panel=p, idx=np.arange(len(dd))[sl], X=X[sl], G=G[sl], y=y[sl], e=e[sl], first=first[sl]))
    return fit, val, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 23])
    ap.add_argument("--tag", default="st")
    ap.add_argument("--epochs", type=int, default=t2_stnet.CFG["epochs"])
    ap.add_argument("--tw", type=int, default=t2_stnet.CFG["tw"])
    ap.add_argument("--width", type=int, default=t2_stnet.CFG["width"])
    ap.add_argument("--lr", type=float, default=t2_stnet.CFG["lr"])
    ap.add_argument("--no-x", action="store_true", help="grid only (no engineered per-link features)")
    ap.add_argument("--no-g", action="store_true", help="engineered features only (no grid)")
    ap.add_argument("--no-place", action="store_true", help="drop the panel / pos inputs (place-free)")
    ap.add_argument("--pers-w", type=float, default=0.0, help="window weight 1 + a * (1 - persistence IoU)")
    a = ap.parse_args()
    fit, val, names = load_packs()
    keep = [i for i, n in enumerate(names) if not n.startswith(LONG)]
    if a.no_place:
        keep = [i for i in keep if names[i] not in ("panel", "pos")]
    if a.pers_w:
        iql = names.index("q_last")
        for pk in fit + val:
            ql = np.nan_to_num(np.asarray(pk["X"][:, :, iql])) > 0.5            # (N, L)
            t, pr = pk["y"] & pk["e"], ql[:, None, :] & pk["e"]
            u = (t | pr).sum((1, 2)); pers = np.where(u == 0, 1.0, (t & pr).sum((1, 2)) / np.maximum(u, 1))
            pk["w"] = 1 + a.pers_w * (1 - pers)
        t2_stnet.trim()
        print(f"persistence weights: mean {np.mean(np.concatenate([p['w'] for p in fit])):.3f}", flush=True)
    print(f"fit {sum(len(p['y']) for p in fit)} / val {sum(len(p['y']) for p in val)} windows, {len(keep)} features", flush=True)
    for s in a.seeds:
        f = ROOT / "out_st" / f"{a.tag}_ongoing_s{s}.pt"
        if f.exists():
            print(f"{f.name} exists", flush=True); continue
        t0 = time.time()
        b = t2_stnet.train(fit, val, keep, "ongoing", s, log=lambda m: print(m, flush=True),
                           cfg=dict(epochs=a.epochs, tw=a.tw, width=a.width, lr=a.lr), use_x=not a.no_x, use_g=not a.no_g)
        b["names"] = names
        torch.save(b, f)
        P = t2_stnet.predict(b, val)
        np.savez_compressed(ROOT / "out_st" / f"{a.tag}_feb_s{s}.npz", **{p["panel"]: x.astype(np.float16) for p, x in zip(val, P)})
        print(f"seed {s}: {time.time() - t0:.0f}s, Feb iou@.45 {t2_stnet.panel_iou(P, val):.4f}", flush=True)


if __name__ == "__main__":
    main()
