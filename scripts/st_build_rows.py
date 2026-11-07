"""Ongoing training windows for the space-time net (src/t2_stnet.py), per panel on disk in out_st/rows/.

Same windows and features as t2_pipeline's ongoing rows with TFB_YOUNG_STEP=1 TFB_RAMPS=1 (t2_data.build_train
+ refresh to 7-day memory), plus the raw (5, 13, L) grid of slots T-12..T: history from the train layer, row T
from the masked layer (as window_features gets it), ramp flows from ramp_states (stamped <= T).

  TFB_YOUNG_STEP=1 TFB_RAMPS=1 python scripts/st_build_rows.py [panel ...]
"""
import json, os, sys, time
assert os.environ.get("TFB_YOUNG_STEP") == "1" and os.environ.get("TFB_RAMPS") == "1"
import numpy as np, pandas as pd
from numpy.lib.format import open_memmap
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
import t2_data
from common import QUEUE_PANELS, ROOT, days
from t2_data import H, Recent, Static, priors, train_arrays, training_origins, window_features
import t2_stnet
t2_data.RECENT_DAYS = 7          # = refresh(rows, names, 7) in t2_pipeline
D = ROOT / "out_st" / "rows"; D.mkdir(parents=True, exist_ok=True)


def build(panel):
    t0 = time.time()
    st = Static(panel)
    v, q, e, vm, qm = train_arrays(st)
    dl = days(panel, "train")
    mcode = pd.factorize(pd.Index([x[:7] for x in dl]), sort=True)[0]
    dow = pd.to_datetime(pd.Series(dl)).dt.dayofweek.to_numpy()
    onset_c, queued_c, seen_c = priors(st, v, e, mcode)
    wins = [w for w in training_origins(v, e, st.cut[None, :]) if w[2] == "queue_ongoing"]
    rec = Recent(st)
    a_or, a_fr = t2_stnet.ramp_maps(st)
    rq = rec.rq["train"]
    N, L = len(wins), st.L
    names = None
    X = G = None
    Y = np.zeros((N, 6, L), bool); E = np.zeros((N, 6, L), bool)
    meta = dict(day=[], slot=[], first=[])
    for i, (d, t, _) in enumerate(wins):
        m = mcode[d]
        op = (onset_c.sum(0) - onset_c[m]) / (len(mcode) - (mcode == m).sum())
        qp = (queued_c.sum(0) - queued_c[m]) / np.maximum(seen_c.sum(0) - seen_c[m], 1)
        V = np.vstack([v[d, t - H:t], vm[d, t][None]])
        Q = np.vstack([q[d, t - H:t], qm[d, t][None]])
        x, names = window_features(st, V, Q, t, dow[d], op, qp, rec.features("train", d, t))
        if X is None:
            X = open_memmap(D / f"X_{panel}.npy", "w+", np.float32, (N, L, x.shape[1]))
            G = open_memmap(D / f"G_{panel}.npy", "w+", np.float16, (N, 5, H + 1, L))
        X[i] = x
        G[i] = t2_stnet.make_grid(st, V, Q, np.asarray(rq[d, t - H:t + 1], np.float32), a_or, a_fr)
        Y[i] = np.isfinite(v[d, t + 1:t + 7]) & (v[d, t + 1:t + 7] <= st.cut[None, :])
        E[i] = e[d, t + 1:t + 7] & np.isfinite(v[d, t + 1:t + 7])
        hq = (np.isfinite(v[d, t - H:t]) & (v[d, t - H:t] <= st.cut[None, :])).any(1)
        meta["day"].append(dl[d]); meta["slot"].append(int(t)); meta["first"].append(int(np.argmax(hq)) if hq.any() else -1)
    X.flush(); G.flush(); del X, G
    np.save(D / f"y_{panel}.npy", Y); np.save(D / f"e_{panel}.npy", E)
    meta["names"] = names
    (D / f"meta_{panel}.json").write_text(json.dumps(meta))
    print(f"{panel}: {N} ongoing windows, L {L} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    for p in sys.argv[1:] or QUEUE_PANELS:
        if not (D / f"meta_{p}.json").exists():
            build(p)
