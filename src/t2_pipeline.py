"""Final Task 2 pipeline (the reproduction entry point): train-split windows -> four LightGBM models refit
on every train month -> queue for the scored windows.

Models (LightGBM, PARAMS in t2_model.py):
  memory-free   no place-time memory              onset 424 rounds, ongoing 4243
  recent        + last-7-day memory (masked layer) onset 441 rounds, ongoing 4359
Round counts are the Feb early-stopping results of the holdout runs (t2_model.py --drop-mem;
t2_onset_recent.py; t2_ongoing_recent.py) plus 10% for the extra month.
Decoding: P = (1 - w) recent + w memory-free. Ongoing: w = 0.3, thr 0.45. Onset: w = 0.5 x cosine(recent onset
memory, long-term train onset prior) over the window's links (0.3 when either is empty), so the memory-free
model loses its say where the active bottleneck has moved; thr 0.3 with a top-3 fallback at T+30.

Both conditions then average that with corridor CNNs (t2_nnlib.py, trained on Jun-Jan and stopped at the
best Feb epoch; the onset CNNs are trained on the T+30 step only): P = (1 - NN_W) P_lgb + NN_W P_cnn, NN_W 0.5.
P_cnn averages CNN groups, each a seed average: ongoing uses the full-feature CNN (3 seeds); onset averages
the full-feature CNN (5 seeds) and a place-free CNN without the panel / pos inputs (3 seeds). All scored
windows lie in the first days of their month, where recent memory can still describe the previous split's
bottlenecks, and there the CNNs (convolutions shared along the corridor) carry the onset blend.

Every input of a scored window is stamped <= T: the 12-step history, the masked-layer row at T, recent
memory from earlier days and from the same day before T-12, and static network data. Models are fitted
on train months only.
"""
from __future__ import annotations

import argparse
import os
import pickle
import time

import lightgbm as lgb
import numpy as np

import t2_data
from common import OUT, QUEUE_PANELS
from t2_data import scored_windows
import t2_nnlib
from t2_model import HOLD, MEMORY, PARAMS, collect, decode, predict_window, stack
from t2_onset_recent import refresh

LONG = ("onset_prior", "queued_prior")
RECENT_DAYS = int(os.environ.get("TFB_RECENT_DAYS", 7))   # recent-memory window (days); 7 = final
W_BASE = 0.3
ROUNDS = {("base", "onset"): 424, ("base", "ongoing"): 4243, ("rec", "onset"): 441, ("rec", "ongoing"): 4359}
PLACE = ("panel", "pos")
NN_W = {"onset": 0.5, "ongoing": 0.5}
NN_GROUPS = {"onset": {"cnn": (7, 11, 23, 31, 47), "cnn_np": (7, 11, 23)}, "ongoing": {"cnn": (7, 11, 23)}}
CONDS = {"onset": ("queue_onset", (6,), {}),
         "ongoing": ("queue_ongoing", (1, 2, 3, 4, 5, 6), dict(window_stride=3, n_h=3, far_keep=0.15))}


def onset_weight(X: np.ndarray, names: list[str]) -> float:
    """Weight of the memory-free model: 0.5 x agreement between recent and long-term onset memory."""
    a, b = X[:, names.index("recent_onset")], X[:, names.index("onset_prior")]
    if a.sum() == 0 or b.sum() == 0:
        return W_BASE
    return 0.5 * float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def fit(rows, names, keep, key, kind):
    out = OUT / f"t2final_{kind}_{key}.txt"
    if out.exists():
        return lgb.Booster(model_file=str(out))
    cond, hz, kw = CONDS[key]
    nm = [names[i] for i in keep]
    X, y, w = stack([dict(r, X=r["X"][:, keep]) for r in rows[key]], cond, hz, nm, seed=3, **kw)
    n = ROUNDS[(kind, key)]
    print(f"{kind} {key}: {len(y):,} rows, {n} rounds", flush=True)
    m = lgb.train(PARAMS, lgb.Dataset(X, y, weight=w, feature_name=nm + ["h"]), n)
    m.save_model(str(out))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nn-w", nargs=2, type=float, metavar=("ONSET", "ONGOING"), help="CNN blend weights (default: NN_W)")
    ap.add_argument("--onset-cnn-seeds", nargs="+", type=int, help="full-feature onset CNN seeds")
    ap.add_argument("--onset-np-seeds", nargs="*", type=int, help="place-free onset CNN seeds (none: no such group)")
    ap.add_argument("--ongoing-cnn-seeds", nargs="+", type=int, help="ongoing CNN seeds")
    ap.add_argument("--out", default="t2_queue_final.pkl")
    a = ap.parse_args()
    if a.nn_w:
        NN_W.update(onset=a.nn_w[0], ongoing=a.nn_w[1])
    if a.onset_cnn_seeds:
        NN_GROUPS["onset"]["cnn"] = tuple(a.onset_cnn_seeds)
    if a.onset_np_seeds is not None:
        NN_GROUPS["onset"]["cnn_np"] = tuple(a.onset_np_seeds)
    if a.ongoing_cnn_seeds:
        NN_GROUPS["ongoing"]["cnn"] = tuple(a.ongoing_cnn_seeds)
    groups = {key: {g: s for g, s in NN_GROUPS[key].items() if s} for key in NN_W if NN_W[key] > 0}
    t0 = time.time()
    nn_files = {(key, g, s): OUT / f"t2final_{'nn' if g == 'cnn' else 'nnnp'}_{key}_s{s}.pt"
                for key in groups for g in groups[key] for s in groups[key][g]}
    trained = [OUT / f"t2final_{k}_{key}.txt" for k in ("base", "rec") for key in ("onset", "ongoing")] + list(nn_files.values())
    if all(f.exists() for f in trained):
        rows, names = None, scored_windows(QUEUE_PANELS[0], "validation")[1]   # every model is on disk
    else:
        rows, names = collect("train")
        print(f"rows: onset {len(rows['onset'])}, ongoing {len(rows['ongoing'])} windows ({time.time() - t0:.0f}s)", flush=True)
    kb = [i for i, n in enumerate(names) if not n.startswith(MEMORY)]
    kr = [i for i, n in enumerate(names) if not n.startswith(LONG)]
    knp = [i for i in kr if names[i] not in PLACE]
    models = {(k, key): fit(rows, names, kb, key, k) for k in ("base",) for key in ("onset", "ongoing")}
    if not all((OUT / f"t2final_rec_{key}.txt").exists() for key in ("onset", "ongoing")) or not all(f.exists() for f in nn_files.values()):
        refresh({"train": rows["onset"] + rows["ongoing"]}, names, RECENT_DAYS)
    models.update({("rec", key): fit(rows, names, kr, key, "rec") for key in ("onset", "ongoing")})
    nets = {}
    for (key, g, seed), f in nn_files.items():
        if not f.exists():
            b = t2_nnlib.train([r for r in rows[key] if r["day"] < HOLD], [r for r in rows[key] if r["day"] >= HOLD],
                               kr if g == "cnn" else knp, key, seed, log=lambda m: print(m, flush=True))
            t2_nnlib.torch.save(b, f)
        nets[(key, g, seed)] = t2_nnlib.torch.load(f, weights_only=False)
    del rows
    t2_data.RECENT_DAYS = RECENT_DAYS
    queue, probs = {}, {}
    for p in QUEUE_PANELS:
        for split in ("validation", "private"):
            blocks, bn = scored_windows(p, split)
            assert bn == names
            pnn = {}
            for key in groups:
                kb_ = [b for b in blocks if (b["condition"] == "queue_onset") == (key == "onset")]
                if kb_:
                    pg = [[t2_nnlib.predict(nets[(key, g, s_)], kb_) for s_ in seeds] for g, seeds in groups[key].items()]
                    pnn.update({b["window_id"]: np.mean([np.mean([p_[i] for p_ in ps_], axis=0) for ps_ in pg], axis=0)
                                for i, b in enumerate(kb_)})
            for b in blocks:
                key = "onset" if b["condition"] == "queue_onset" else "ongoing"
                w = onset_weight(b["X"], names) if key == "onset" else W_BASE
                P = (1 - w) * predict_window(models[("rec", key)], b["X"][:, kr], b["condition"]) \
                    + w * predict_window(models[("base", key)], b["X"][:, kb], b["condition"])
                if b["window_id"] in pnn:
                    P = (1 - NN_W[key]) * P + NN_W[key] * pnn[b["window_id"]]
                pred = decode(P, b["condition"], 0.3, 3) if key == "onset" else decode(P, b["condition"], 0.45)
                out = np.zeros_like(pred, dtype=np.int8)
                out[:, b["order"]] = pred
                queue[b["window_id"]] = out
                probs[b["window_id"]] = (P.astype(np.float32), np.asarray(b["order"]), b["condition"])
    pickle.dump(queue, open(OUT / a.out, "wb"))
    pickle.dump(probs, open(OUT / a.out.replace(".pkl", "_probs.pkl"), "wb"))     # pre-decode, travel order
    print(f"scored windows: {len(queue)} ({time.time() - t0:.0f}s)", flush=True)
    for ref in (OUT / "t2rec7e_queue_final.pkl", OUT / "t2rec7f_queue_final.pkl", OUT / "t2rec7g_queue_final.pkl"):
        if ref.exists():
            r = pickle.load(open(ref, "rb"))
            same = sum((queue[k] == r[k]).all() for k in queue)
            print(f"identical to {ref.name}: {same}/{len(queue)} windows", flush=True)


if __name__ == "__main__":
    main()
