"""Post-T deletion check + reproduction check for the month-pure memory onset (msmU, final pick v51).

For every scored ONSET window (validation + private), rebuild the memory-blended prediction of onset_msm_scored.py
(RHO 1.0, MINEV 1, W 0.5, SLOT0 0) twice:
  full   - from the release as shipped,
  cut    - with every masked-layer cell after the window origin T blanked (rest of the day, later days, later splits;
           src/t2_post_t_check.truncated_load, the organizers' rerun),
and assert (1) cut == full (no post-T dependence) and (2) full == the queue in kaggle_pkg/src_ds/queue_msmU_stF.pkl
(the file assembled into v51). Negative control: cutting at the start of the previous day must change some windows.

  python scripts/msm_post_t_check.py
"""
import pickle, sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
import common
from common import OUT, REL, QUEUE_PANELS, ROOT, days, T
from t2_data import Static
from t2_model import decode
from t2_post_t_check import truncated_load

RHO, MINEV, W, SLOT0 = 1.0, 1.0, 0.5, 0
DS = ROOT / "kaggle_pkg" / "src_ds"
pub = pickle.load(open(OUT / "t2pub_scored_probs.pkl", "rb"))
ref = pickle.load(open(DS / "queue_msmU_stF.pkl", "rb"))
base = pickle.load(open(DS / "queue_pub_stF.pkl", "rb"))


def events(load, pn, split, st):
    v = load(pn, split, "v")[:, :, st.order]
    qd = np.isfinite(v) & (v <= st.cut[None, None, :]); act = qd.any(axis=2)
    ev = np.zeros((v.shape[0], 24, st.L), np.float32)
    for d in range(v.shape[0]):
        for t in range(12, T - 6):
            if act[d, t] and not act[d, t - 12:t].any():
                ev[d, t // 12] += qd[d, t:t + 6].any(axis=0)
    return ev


def predict(ev, wid, d, slot, st):
    h = (slot + 6) // 12 % 24
    m = (ev[:d][:, [(h - 1) % 24, h, (h + 1) % 24]] * np.array([0.5, 1.0, 0.5])[None, :, None]).sum((0, 1))
    P = pub[wid].copy(); b = decode(P, "queue_onset", 0.55, 1)
    if slot >= SLOT0 and m.sum() >= MINEV and m[b[5]].max() <= RHO * m.max():
        P[5] = (1 - W) * P[5] + W * m / m.max()
        pred = decode(P, "queue_onset", 0.55, 1)
        out = np.zeros_like(pred, dtype=np.int8); out[:, st.order] = pred
        return out
    return base[wid]


def main():
    n = same_cut = same_ref = neg_changed = 0
    for pn in QUEUE_PANELS:
        st = Static(pn)
        for split in ("validation", "private"):
            ev_full = events(common.load, pn, split, st)
            di = {x: i for i, x in enumerate(days(pn, split))}
            wi = pd.read_csv(REL / "task2" / pn / split / "window_index.csv")
            for w in wi.itertuples(index=False):
                if w.condition != "queue_onset" or w.window_id not in ref:
                    continue
                o = str(w.forecast_origin); d = di[o[:10]]; slot = int(o[11:13]) * 12 + int(o[14:16]) // 5
                full = predict(ev_full, w.window_id, d, slot, st)
                cut = predict(events(truncated_load(split, d, slot), pn, split, st), w.window_id, d, slot, st)
                neg = predict(events(truncated_load(split, max(d - 1, 0), -1), pn, split, st), w.window_id, d, slot, st)
                n += 1; same_cut += int((cut == full).all()); same_ref += int((full == ref[w.window_id]).all())
                neg_changed += int(not (neg == full).all())
            print(f"{pn} {split}: checked {n}, cut==full {same_cut}, full==v51 file {same_ref}, neg control changed {neg_changed}",
                  flush=True)
    print(f"DONE: {n} onset windows; identical after post-T deletion {same_cut}/{n}; reproduce v51 queue {same_ref}/{n}; "
          f"negative control (cut a day early) changed {neg_changed}", flush=True)


if __name__ == "__main__":
    main()
