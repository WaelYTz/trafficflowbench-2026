"""March official-style pseudo-windows with the history gaps filled by our Task 1 imputations, so the history is
complete like window_history.parquet (the masked-history pseudo sets overstate memory). Selection, labels, the
row at T and recent memory stay on the masked layer, exactly as for the official windows. All models are the
final Jun-Feb fits, so March is out of sample. Question: does this in-regime proxy reproduce the public LB
(onset pub 0.776 > omix 0.743 ~ pT 0.739 > rec7g 0.718; ongoing nm = c8 - 0.042, c10 = c8 - 0.009, g = c8 - 0.0035)?

  python mar_fill_eval.py [--mask]     (--mask: the same windows with the masked history, for comparison)
"""
import argparse, os, pickle, sys
import numpy as np, pandas as pd, torch
SRC = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src")
sys.path.insert(0, SRC)
import lightgbm as lgb
import t2_data, t2_nnlib
from common import OUT, QUEUE_PANELS, ROOT, days, load
from t2_data import H, Recent, Static, priors, train_arrays, window_features
from t2_model import MEMORY, aggregate, decode, iou, predict_window
from t2_pseudo import _official, masked
from t2_pipeline import LONG, PLACE, onset_weight
t2_data.RECENT_DAYS = 7                 # the final models' memory window (t2_pipeline sets it the same way)

NEW = ("today_q_before_ph", "today_q_before_age", "today_q_before_corr")
T1 = {p: OUT / "kg_t1m_npz" / "out" / f"t1_{p}_validation.npz" for p in QUEUE_PANELS}
T1["D7_I405_S"] = OUT / "kg_t1m_fix" / "out" / "t1_D7_I405_S_validation.npz"


def build(panel, fill):
    st = Static(panel); o = st.order
    v, q, e = masked(st, "validation")
    el = load(panel, "validation", "elig")[:, :, o].astype(bool)
    if fill:
        z = np.load(T1[panel])
        vf = np.where(np.isfinite(v), v, z["V"][:, :, o]); qf = np.where(np.isfinite(q), q, z["Q"][:, :, o])
    else:
        vf, qf = v, q
    dl = days(panel, "validation")
    vt, _, et, _, _ = train_arrays(st)
    mcode = pd.factorize(pd.Index([x[:7] for x in days(panel, "train")]), sort=True)[0]
    onset_c, queued_c, seen_c = priors(st, vt, et, mcode)
    del vt, et
    op = onset_c.sum(0) / len(mcode); qp = queued_c.sum(0) / np.maximum(seen_c.sum(0), 1)
    rec = Recent(st)
    dow = pd.to_datetime(pd.Series(dl)).dt.dayofweek.to_numpy()
    rows = {"onset": [], "ongoing": []}
    names = None
    for d, t, cond in _official(v, e, st.cut):
        V = np.vstack([vf[d, t - H:t], v[d, t:t + 1]]); Q = np.vstack([qf[d, t - H:t], q[d, t:t + 1]])
        t2_data.TODAY2 = True
        f = rec.features("validation", d, t)
        t2_data.TODAY2 = False
        base = {k: f[k] for k in f if k not in NEW}
        X, names = window_features(st, V, Q, t, dow[d], op, qp, base)
        hv = vf[d, t - H:t]
        hT = np.vstack([vf[d, t - H + 1:t], v[d, t:t + 1]]); qT = np.vstack([qf[d, t - H + 1:t], q[d, t:t + 1]])
        key = "onset" if cond == "queue_onset" else "ongoing"
        rows[key].append(dict(X=X, X2=np.column_stack([X] + [np.asarray(f[k], np.float32) for k in NEW]),
                              y=e[d, t + 1:t + 7] & (v[d, t + 1:t + 7] <= st.cut[None, :]), elig=e[d, t + 1:t + 7].copy(),
                              day=dl[d], slot=t, panel=panel,
                              hist=(hv, qf[d, t - H:t], (el[d, t - H:t] & np.isfinite(hv)).astype(float)),
                              histT=(hT, qT, np.isfinite(hT).astype(float))))
    return rows, names


def origin(r):
    return pd.Timestamp(f"{r['day']}T{r['slot'] // 12:02d}:{r['slot'] % 12 * 5:02d}:00Z")


def probs(rows, names):
    kb = [i for i, n in enumerate(names) if not n.startswith(MEMORY)]
    kr = [i for i, n in enumerate(names) if not n.startswith(LONG)]
    L = lambda f: lgb.Booster(model_file=str(OUT / f))
    N = lambda f: torch.load(f, weights_only=False)
    P = {}
    # onset
    R = rows["onset"]
    rec, base = L("t2final_rec_onset.txt"), L("t2final_base_onset.txt")
    plgb = []
    for r in R:
        w = onset_weight(r["X"], names)
        plgb.append((1 - w) * predict_window(rec, r["X"][:, kr], "queue_onset") + w * predict_window(base, r["X"][:, kb], "queue_onset"))
    g1 = [t2_nnlib.predict(N(OUT / f"t2final_nn_onset_s{s}.pt"), R) for s in (7, 11, 23, 31, 47)]
    g2 = [t2_nnlib.predict(N(OUT / f"t2final_nnnp_onset_s{s}.pt"), R) for s in (7, 11, 23)]
    pnn = [0.5 * np.mean([g[i] for g in g1], 0) + 0.5 * np.mean([g[i] for g in g2], 0) for i in range(len(R))]
    P[("onset", "rec7g")] = [0.5 * a + 0.5 * b for a, b in zip(plgb, pnn)]
    from t2_pub import Public
    pub = Public()
    P[("onset", "pub")] = [pub.probs(r["panel"], *r["hist"], origin(r), "queue_onset") for r in R]
    from catboost import CatBoostClassifier
    import t2_pubbag
    cats = []
    for i in range(3):
        m = CatBoostClassifier(); m.load_model(str(OUT / f"pubbag_T_cat_{i}.cbm")); cats.append(m)
    bag = t2_pubbag.Bag([L(f"pubbag_T_lgb_{i}.txt") for i in range(5)], cats)
    P[("onset", "pT")] = [bag.probs(r["panel"], *r["histT"], origin(r))[0] for r in R]
    # ongoing
    R = rows["ongoing"]
    rec, base = L("t2final_rec_ongoing.txt"), L("t2final_base_ongoing.txt")
    pb = [predict_window(base, r["X"][:, kb], "queue_ongoing") for r in R]
    pl = [0.7 * predict_window(rec, r["X"][:, kr], "queue_ongoing") + 0.3 * b for r, b in zip(R, pb)]
    mean = lambda ps: [np.mean([p[i] for p in ps], 0) for i in range(len(R))]
    pn = mean([t2_nnlib.predict(N(OUT / f"t2final_nn_ongoing_s{s}.pt"), R) for s in (7, 11, 23)])
    pnm = mean([t2_nnlib.predict(N(OUT / f"t2nn_ongoing_nm_s{s}.pt"), R) for s in (7, 11, 23)])
    R2 = [dict(r, X=r["X2"]) for r in R]
    pn2 = mean([t2_nnlib.predict(N(ROOT / "out_today2" / f"t2final_nn_ongoing_s{s}.pt"), R2) for s in (7, 11, 23)])
    P[("ongoing", "base")] = pb; P[("ongoing", "lgb")] = pl
    P[("ongoing", "cnn")] = pn; P[("ongoing", "cnn_nm")] = pnm; P[("ongoing", "cnn_t2")] = pn2
    return P


ONSET = {"pub": ("pub", 0.55, 1), "pT": ("pT", 0.5, 1), "rec7g": ("rec7g", 0.3, 3)}


def report(rows, P, tag):
    out = {}
    R = rows["onset"]
    V = {k: [decode(p, "queue_onset", thr, fb) for p in P[("onset", m)]] for k, (m, thr, fb) in ONSET.items()}
    V["omix"] = [decode(0.5 * a + 0.5 * b, "queue_onset", 0.4, 2) for a, b in zip(P[("onset", "rec7g")], P[("onset", "pub")])]
    V["pub+pT"] = [decode(0.5 * a + 0.5 * b, "queue_onset", 0.5, 1) for a, b in zip(P[("onset", "pub")], P[("onset", "pT")])]
    out["onset"] = (R, {k: np.array([iou(d, r["y"], r["elig"]) for d, r in zip(v, R)]) for k, v in V.items()})
    R = rows["ongoing"]
    pl, pn = P[("ongoing", "lgb")], P[("ongoing", "cnn")]
    mix = lambda a, wa, b: [wa * x + (1 - wa) * y for x, y in zip(a, b)]
    W = {"c8": mix(pl, 0.2, pn), "g": mix(pl, 0.5, pn), "c10": pn, "nm": mix(P[("ongoing", "base")], 0.2, P[("ongoing", "cnn_nm")]),
         "t2new": mix(pl, 0.2, P[("ongoing", "cnn_t2")]), "e6": mix(pl, 0.2, mix(pn, 0.5, P[("ongoing", "cnn_t2")])),
         "lgb": pl}
    out["ongoing"] = (R, {k: np.array([iou(decode(p, "queue_ongoing", 0.45), r["y"], r["elig"]) for p, r in zip(v, R)]) for k, v in W.items()})
    cond = {"onset": "queue_onset", "ongoing": "queue_ongoing"}
    for key, (R, S) in out.items():
        first5 = np.zeros(len(R), bool)
        for p in QUEUE_PANELS:
            ix = sorted([i for i, r in enumerate(R) if r["panel"] == p], key=lambda i: (R[i]["day"], R[i]["slot"]))[:5]
            first5[ix] = True
        dd = np.array([int(r["day"][8:10]) for r in R])
        subs = {"all": np.ones(len(R), bool), "d<=7": dd <= 7, "first5": first5, "d>=15": dd >= 15}
        print(f"\n== {tag} {key}: {len(R)} windows ({', '.join(f'{k} {m.sum()}' for k, m in subs.items())})", flush=True)
        for k, s in S.items():
            agg = {sn: aggregate([(r["panel"], cond[key], x) for r, x, m in zip(R, s, sm) if m])["overall"] for sn, sm in subs.items()}
            print(f"  {k:8s} " + "  ".join(f"{sn} {a:.4f}" for sn, a in agg.items()), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", action="store_true")
    a = ap.parse_args()
    tag = "marmask" if a.mask else "marfill"
    f = OUT / f"t2_pseudo_{tag}.pkl"
    if f.exists():
        rows, names = pickle.load(open(f, "rb"))
    else:
        rows, names = {"onset": [], "ongoing": []}, None
        for p in QUEUE_PANELS:
            r, names = build(p, not a.mask)
            for k in rows:
                rows[k] += r[k]
            print(tag, p, {k: len(v) for k, v in r.items()}, flush=True)
        pickle.dump((rows, names), open(f, "wb"))
    fp = OUT / f"t2_pseudo_{tag}_probs.pkl"
    if fp.exists():
        P = pickle.load(open(fp, "rb"))
    else:
        P = probs(rows, names)
        pickle.dump(P, open(fp, "wb"))
    report(rows, P, tag)


if __name__ == "__main__":
    main()
