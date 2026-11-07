"""Previous-month lift for the public onset model (Nov 6).

lift = P_num / P_den from two LightGBM bags on the public onset recipe rows:
  March windows (validation, public LB): den = Jun-Jan, num = Jun-Jan + Feb at weight W   (Feb = month before March)
  April windows (private):               den = Jun-Feb, num = Jun-Feb + all March at W    (March = month before April)
New onset prob at T+30 = clip(pub * (num + eps) / (den + eps)), decoded at 0.55 / top-1 like the public onset.
Causal: every training row is from a month before the scored window. The public LB then tests the same method
(previous-month lift) that the private rows use.

  python onset_lift.py            -> bench check + kaggle_pkg/src_ds/queue_lift_stF.pkl
"""
import pickle
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
from common import OUT, QUEUE_PANELS, REL, ROOT  # noqa: E402
from t2_model import aggregate, decode, iou  # noqa: E402
from t2_pub import Public, history_scored  # noqa: E402
from t2_pubbag import LGB_PARAMS  # noqa: E402
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from mar_fill_eval import origin as bench_origin  # noqa: E402

C = "queue_onset"; THR = 0.55; W = 5.0; EPS = 1e-3; NB = 5
DS = ROOT / "kaggle_pkg" / "src_ds"


def fit(X, y, w, tag):
    fs = [OUT / f"lift_{tag}_{s}.txt" for s in range(NB)]
    if all(f.exists() for f in fs):
        return [lgb.Booster(model_file=str(f)) for f in fs]
    rounds = int(np.mean(pickle.load(open(OUT / "pubbag_rounds.pkl", "rb"))["lgb"]) * 1.1)
    _ms = [lgb.train(dict(LGB_PARAMS, seed=100 + s, bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, verbose=-1),
                      lgb.Dataset(X, label=y, weight=w), num_boost_round=rounds) for s in range(NB)]
    ms = _ms
    for m, f in zip(ms, fs):
        m.save_model(str(f))
    return ms


def lifted(pub, name, P, V, Q, E, origin, num, den):
    onset_p, _, pos = pub.panel(name)
    X = onset_p.features(V[:, pos].astype(np.float32), Q[:, pos].astype(np.float32), E[:, pos].astype(np.float32), origin, C)[-len(pos):]
    pn = np.mean([m.predict(X) for m in num], 0); pd_ = np.mean([m.predict(X) for m in den], 0)
    out = P.copy(); out[5, pos] = np.clip(P[5, pos] * (pn + EPS) / (pd_ + EPS), 0, 1)
    return out


def main():
    R = dict(np.load(OUT / "pubbag_rows.npz")); M = dict(np.load(OUT / "pubbag_rows_mar.npz"))
    feb = R["feb"].astype(bool)
    print(f"train rows {len(R['y'])}, Feb {feb.sum()}, March {len(M['y'])}", flush=True)
    den_v = fit(R["X"][~feb], R["y"][~feb], None, "denv")
    num_v = fit(R["X"], R["y"], np.where(feb, W, 1.0), "numv")
    den_p = fit(R["X"], R["y"], None, "denp")
    num_p = fit(np.concatenate([R["X"], M["X"]]), np.concatenate([R["y"], M["y"]]),
                np.concatenate([np.ones(len(R["y"])), np.full(len(M["y"]), W)]), "nump")
    print("fitted", flush=True)
    pub = Public()

    # 1) local evidence: Feb->March lift on every March bench onset window (filled history, pseudo windows)
    rows, _ = pickle.load(open(OUT / "t2_pseudo_marfill.pkl", "rb"))
    PB = pickle.load(open(OUT / "t2_pseudo_marfill_probs.pkl", "rb"))
    Wn = rows["onset"]; pp = PB[("onset", "pub")]
    s0, s1 = [], []
    for r, p in zip(Wn, pp):
        hv, hq, he = r["hist"]
        pl = lifted(pub, r["panel"], p, hv, hq, he, bench_origin(r), num_v, den_v)
        s0.append(iou(decode(p, C, THR, 1), r["y"], r["elig"])); s1.append(iou(decode(pl, C, THR, 1), r["y"], r["elig"]))
    s0, s1 = np.array(s0), np.array(s1)
    dd = np.array([int(r["day"][8:10]) for r in Wn])
    for nm, m in (("all", dd > 0), ("d<=5", dd <= 5), ("d6-15", (dd > 5) & (dd <= 15)), ("d16+", dd >= 16)):
        a0 = aggregate([(r["panel"], C, x) for r, x, k in zip(Wn, s0, m) if k])["overall"]
        a1 = aggregate([(r["panel"], C, x) for r, x, k in zip(Wn, s1, m) if k])["overall"]
        print(f"bench {nm:6s} n {m.sum():3d}: pub {a0:.4f} -> Feb-lift {a1:.4f} ({a1 - a0:+.4f})  "
              f"win/loss {(s1[m] > s0[m] + 1e-9).sum()}/{(s1[m] < s0[m] - 1e-9).sum()}", flush=True)

    # 2) scored windows: validation = Feb lift, private = March lift
    rec = pickle.load(open(OUT / "t2rec7g2_queue_final_probs.pkl", "rb"))
    q = pickle.load(open(DS / "queue_pub_stF.pkl", "rb"))
    same = changed = 0; new = {}
    for split, (num, den) in (("validation", (num_v, den_v)), ("private", (num_p, den_p))):
        ch = 0
        for p in QUEUE_PANELS:
            wi = pd.read_csv(REL / "task2" / p / split / "window_index.csv")
            cond = dict(zip(wi.window_id, wi.condition))
            for wid, (V, Q, E, origin) in history_scored(p, split).items():
                if cond[wid] != C or wid not in q:
                    continue
                P = pub.probs(p, V, Q, E, origin, C)
                base = np.zeros(P.shape, np.int8); base[:, rec[wid][1]] = decode(P, C, THR, 1)
                same += int((base == q[wid]).all())
                out = np.zeros(P.shape, np.int8); out[:, rec[wid][1]] = decode(lifted(pub, p, P, V, Q, E, origin, num, den), C, THR, 1)
                new[wid] = out; ch += int(not (out == base).all())
        print(f"{split}: onset windows changed by the lift {ch}", flush=True); changed += ch
    print(f"sanity: pub decode reproduces queue_pub_stF onset in {same}/{len(new)} windows", flush=True)
    q2 = dict(q); q2.update(new)
    pickle.dump(q2, open(DS / "queue_lift_stF.pkl", "wb"))
    qv = dict(q); qv.update({k: v for k, v in new.items() if "_private_" in k})
    pickle.dump(qv, open(DS / "queue_liftP_stF.pkl", "wb"))   # private-only lift (public rows = pub)
    print("wrote queue_lift_stF.pkl (both splits) and queue_liftP_stF.pkl (private only)", flush=True)


if __name__ == "__main__":
    main()
