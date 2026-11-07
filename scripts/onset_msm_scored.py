"""Month-pure memory onset gate on the SCORED windows -> kaggle_pkg/src_ds/queue_<name>_stF.pkl.
Memory = onset footprints (masked layer, links queued within 30 min of a corridor onset start) on the same split's
days BEFORE the window day, hour band +-1 h; fires where the public model's pick has <= rho * max memory mass and
memory holds >= minev counts; then P[T+30] = (1-w) P_pub + w m/max(m). Every input is a released observation of an
earlier day of the same split (allowed: 'earlier days of the same split')."""
import pickle, sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
from common import OUT, REL, QUEUE_PANELS, ROOT, load, days, T
from t2_data import Static
from t2_model import decode
RHO, MINEV, W, SLOT0, NAME = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
DS = ROOT / "kaggle_pkg" / "src_ds"
pub = pickle.load(open(OUT / "t2pub_scored_probs.pkl", "rb"))
q = pickle.load(open(DS / "queue_pub_stF.pkl", "rb"))
changed = []
for pn in QUEUE_PANELS:
    st = Static(pn)
    for split in ("validation", "private"):
        v = load(pn, split, "v")[:, :, st.order]
        qd = np.isfinite(v) & (v <= st.cut[None, None, :]); act = qd.any(axis=2)
        ev = np.zeros((v.shape[0], 24, st.L), np.float32)
        for d in range(v.shape[0]):
            for t in range(12, T - 6):
                if act[d, t] and not act[d, t - 12:t].any():
                    ev[d, t // 12] += qd[d, t:t + 6].any(axis=0)
        di = {x: i for i, x in enumerate(days(pn, split))}
        wi = pd.read_csv(REL / "task2" / pn / split / "window_index.csv")
        for w in wi.itertuples(index=False):
            if w.condition != "queue_onset":
                continue
            o = str(w.forecast_origin); d = di[o[:10]]; slot = int(o[11:13]) * 12 + int(o[14:16]) // 5
            h = (slot + 6) // 12 % 24
            m = (ev[:d][:, [(h - 1) % 24, h, (h + 1) % 24]] * np.array([0.5, 1.0, 0.5])[None, :, None]).sum((0, 1))
            P = pub[w.window_id].copy(); b = decode(P, "queue_onset", 0.55, 1)
            if slot >= SLOT0 and m.sum() >= MINEV and m[b[5]].max() <= RHO * m.max():
                P[5] = (1 - W) * P[5] + W * m / m.max()
                pred = decode(P, "queue_onset", 0.55, 1)
                out = np.zeros_like(pred, dtype=np.int8); out[:, st.order] = pred
                if not (out == q[w.window_id]).all():
                    changed.append((w.window_id, o))
                q[w.window_id] = out
print(f"{NAME}: changed {sum('validation' in c[0] for c in changed)} public, {sum('private' in c[0] for c in changed)} private")
for c in changed:
    print("  ", c)
pickle.dump(q, open(DS / f"queue_{NAME}_stF.pkl", "wb"))
