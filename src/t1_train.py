"""Task 1: residual gradient-boosted models trained on the released train masks.

Inputs are the train split's masked layer (exactly the scored setting), labels are the unmasked
layer at the template's target cells. February 2031 is held out to score the method with the
official RMSE rule; the final models are refit on every train day and predict validation/private.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import xgboost as xgb

from common import FAMILY, OUT, PANELS, days, load
from t1_features import Grid, build, month_profiles, profiles, train_month_profiles

HOLDOUT_FROM = "2031-02-01"
CHUNK = 60
SEED = 2026
DROP = tuple(x for x in os.environ.get("T1_DROP", "").split(",") if x)   # feature names left out (experiments only)


def drop_cols(X: np.ndarray, names: list[str]) -> np.ndarray:
    return X[:, [i for i, n in enumerate(names) if n not in DROP]] if DROP else X


def bases(X: np.ndarray, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    c = {n: X[:, i] for i, n in enumerate(names)}
    bv = c["iv"].copy()
    for alt in ("v_w15", "prof_v"):
        bv = np.where(np.isfinite(bv), bv, c[alt])
    bv = np.where(np.isfinite(bv), bv, 1.0)
    bq = c["q_w06"].copy()
    for alt in ("q_w15", "iq", "prof_q"):
        bq = np.where(np.isfinite(bq), bq, c[alt])
    return bv, bq


def panel_rows(args) -> dict:
    panel, cap_train, cap_hold, month_prof = args
    rng = np.random.default_rng(SEED + PANELS.index(panel))
    g0 = Grid(panel, "train", d0=0, d1=1)
    if month_prof:
        prof, mid = train_month_profiles(panel)
    else:
        prof, mid = profiles(panel, g0.order), None
    vt = load(panel, "train", "v", mmap=True)
    qt = load(panel, "train", "q", mmap=True)
    dl = days(panel, "train")
    out = {k: [] for k in ("X", "yv", "yq", "hold", "regime", "lanes", "vf", "len")}
    names = None
    for d0 in range(0, len(dl), CHUNK):
        g = Grid(panel, "train", d0=d0, d1=min(d0 + CHUNK, len(dl)))
        d, t, l = np.nonzero(g.target)
        keep = np.zeros(len(d), bool)
        for dd in np.unique(d):
            idx = np.flatnonzero(d == dd)
            cap = cap_hold if g.days[dd] >= HOLDOUT_FROM else cap_train
            keep[rng.choice(idx, min(cap, len(idx)), replace=False)] = True
        d, t, l = d[keep], t[keep], l[keep]
        if month_prof:
            X, names = build(g, d, t, l, prof, loo=False, key=mid[d0:d0 + g.D])
        else:
            X, names = build(g, d, t, l, prof, loo=True)
        sc = g.order[l]
        out["X"].append(X)
        out["yv"].append(vt[d0 + d, t, sc]); out["yq"].append(qt[d0 + d, t, sc])
        out["hold"].append(np.array([g.days[x] >= HOLDOUT_FROM for x in d]))
        out["regime"].append(g.regime[d]); out["lanes"].append(g.lanes[l]); out["vf"].append(g.vf[l])
        out["len"].append(g.length[l])
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["panel"] = np.full(len(res["yv"]), PANELS.index(panel), np.int8)
    res["X"] = drop_cols(res["X"], names)
    res["names"] = [n for n in names if n not in DROP]
    print(f"{panel}: {len(res['yv']):,} rows ({res['hold'].sum():,} holdout)", flush=True)
    return res


def score(pv, pq, yv, yq, lanes, panel, regime) -> tuple[float, dict]:
    """Official rule: per panel x regime RMSE -> regime mean -> family mean -> overall mean."""
    per_panel = {}
    for p in np.unique(panel):
        s = []
        for r in range(3):
            m = (panel == p) & (regime == r)
            if not m.any():
                continue
            rv = np.sqrt(np.mean((pv[m] - yv[m]) ** 2))
            rq = np.sqrt(np.mean(((pq[m] - yq[m]) / lanes[m]) ** 2))
            s.append(0.54 * max(0, 1 - rv / 25) + 0.46 * max(0, 1 - rq / 600))
        per_panel[PANELS[p]] = float(np.mean(s))
    fam = {}
    for p, s in per_panel.items():
        fam.setdefault(FAMILY[p], []).append(s)
    return float(np.mean([np.mean(v) for v in fam.values()])), per_panel


PARAMS = dict(tree_method="hist", max_depth=9, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=50, reg_lambda=5.0, objective="reg:squarederror", max_bin=256, seed=SEED)


def fit(X, y, w, Xv=None, yv=None, rounds=6000, device="cuda"):
    p = dict(PARAMS, device=device)
    dtr = xgb.DMatrix(X, label=y, weight=w)
    if Xv is None:
        return xgb.train(p, dtr, num_boost_round=rounds)
    dva = xgb.DMatrix(Xv, label=yv)
    return xgb.train(p, dtr, num_boost_round=rounds, evals=[(dva, "hold")], early_stopping_rounds=200,
                     verbose_eval=500)


def self_rows(args) -> dict:
    """Training rows from a scored split's own observed cells. Task 1 is offline imputation, so the split's
    released observations are fair inputs (as for the month-own profiles). A fraction of the observed
    eligible non-target cells is hidden, and features use month-own profiles built after hiding."""
    panel, split, frac, seed = args
    g = Grid(panel, split, use_masked=False)
    rng = np.random.default_rng(seed + PANELS.index(panel))
    cand = g.obs & g.elig & ~g.target
    dd, tt, ll = [], [], []
    for d in range(g.D):
        c = np.flatnonzero(cand[d].ravel())
        t, l = np.unravel_index(rng.choice(c, int(round(frac * len(c))), replace=False), cand[d].shape)
        dd.append(np.full(len(t), d)); tt.append(t); ll.append(l)
    d, t, l = (np.concatenate(x) for x in (dd, tt, ll))
    yv, yq = np.asarray(g.v[d, t, l], float), np.asarray(g.q[d, t, l], float)
    g.v = np.array(g.v, np.float32); g.q = np.array(g.q, np.float32)
    g.v[d, t, l] = np.nan; g.q[d, t, l] = np.nan
    g.obs = np.isfinite(g.v) & np.isfinite(g.q)
    X, nm = build(g, d, t, l, month_profiles(g), loo=False)
    bv, bq = bases(X, nm)
    return dict(X=drop_cols(X, nm), tv=yv / g.vf[l] - bv, tq=(yq - bq) / g.lanes[l], w=np.full(len(d), 1.0 / len(d)))


def self_train(models, panels, split, frac, rounds, lr, device, workers):
    """Continue boosting the final models on the split's own re-hidden cells (tested on March pseudo:
    S_state 0.9405 -> 0.9423 at 400 rounds; Feb control flat)."""
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(self_rows, [(p, split, frac, SEED) for p in panels]))
    R = {k: np.concatenate([x[k] for x in parts]) for k in parts[0]}
    del parts
    R["w"] *= len(R["w"]) / R["w"].sum()
    p = dict(PARAMS, device=device, learning_rate=lr)
    out = {key: xgb.train(p, xgb.DMatrix(R["X"], label=R[f"t{key}"], weight=R["w"]), num_boost_round=rounds,
                          xgb_model=models[key]) for key in ("v", "q")}
    print(f"self-train {split}: {len(R['w']):,} rows, +{rounds} rounds", flush=True)
    return out


def predict_split(args) -> str:
    panel, split, models, names, month_prof = args
    g = Grid(panel, split, use_masked=False)
    prof = month_profiles(g) if month_prof else profiles(panel, g.order)
    d, t, l = np.nonzero(g.target)
    X, nm = build(g, d, t, l, prof, loo=False)
    assert [n for n in nm if n not in DROP] == names
    bv, bq = bases(X, nm)
    dm = xgb.DMatrix(drop_cols(X, nm))
    pv = np.clip((bv + models["v"].predict(dm)) * g.vf[l], 3, 140)
    pq = np.clip(bq + models["q"].predict(dm) * g.lanes[l], 60, None)
    V = np.full(g.target.shape, np.nan, np.float32); Q = np.full(g.target.shape, np.nan, np.float32)
    sc = g.order[l]
    V[d, t, sc] = pv; Q[d, t, sc] = pq
    np.savez_compressed(OUT / f"t1_{panel}_{split}.npz", V=V, Q=Q)
    return f"{panel} {split}: {len(d):,} cells"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="*", default=PANELS)
    ap.add_argument("--cap-train", type=int, default=1500)
    ap.add_argument("--cap-hold", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rounds", type=int, default=6000)
    ap.add_argument("--no-predict", action="store_true")
    ap.add_argument("--save-hold", action="store_true", help="save the Jun-Jan models (t1h_*.json) for local studies")
    ap.add_argument("--month-prof", action="store_true",
                    help="scored splits: weekday x slot profiles from the split's own masked layer (all days pooled)")
    ap.add_argument("--train-month-prof", action="store_true",
                    help="also train on month-own profiles (tested Oct 23: same as --month-prof alone, not used)")
    ap.add_argument("--predict-only", metavar="DIR", help="skip training; load t1_v.json / t1_q.json from DIR")
    ap.add_argument("--self-rounds", type=int, default=0,
                    help="per scored split, continue boosting on its own re-hidden observed cells (0 = off)")
    ap.add_argument("--self-frac", type=float, default=0.08, help="share of observed cells re-hidden for --self-rounds")
    ap.add_argument("--self-lr", type=float, default=0.05)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.predict_only:
        from pathlib import Path
        final = {k: xgb.Booster(model_file=str(Path(a.predict_only) / f"t1_{k}.json")) for k in ("v", "q")}
        g = Grid(a.panels[0], "train", d0=0, d1=1)
        names = [n for n in build(g, np.zeros(1, int), np.zeros(1, int), np.zeros(1, int), profiles(a.panels[0], g.order), loo=False)[1]
                 if n not in DROP]
        for s in ("validation", "private"):
            ms = self_train(final, a.panels, s, a.self_frac, a.self_rounds, a.self_lr, a.device, a.workers) if a.self_rounds else final
            for p in a.panels:
                print(predict_split((p, s, ms, names, a.month_prof)), flush=True)
        print(f"Task 1 predictions done {time.time() - t0:.0f}s", flush=True)
        return
    with ProcessPoolExecutor(a.workers) as ex:
        parts = list(ex.map(panel_rows, [(p, a.cap_train, a.cap_hold, a.train_month_prof) for p in a.panels]))
    names = parts[0]["names"]
    R = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "names"}
    print(f"rows {len(R['yv']):,}, features {R['X'].shape[1]}, {time.time() - t0:.0f}s", flush=True)
    bv, bq = bases(R["X"], names)
    tv = R["yv"] / R["vf"] - bv
    tq = (R["yq"] - bq) / R["lanes"]
    # Each panel counts equally in the metric, so each panel gets equal total weight.
    w = np.zeros(len(tv))
    for p in np.unique(R["panel"]):
        m = R["panel"] == p
        w[m] = 1.0 / m.sum()
    w *= len(w) / w.sum()
    h = R["hold"]
    tr = ~h
    models, best = {}, {}
    for key, y in (("v", tv), ("q", tq)):
        m = fit(R["X"][tr], y[tr], w[tr], R["X"][h], y[h], a.rounds, a.device)
        models[key] = m
        best[key] = m.best_iteration + 1
        print(f"{key}: best iteration {best[key]}, {time.time() - t0:.0f}s", flush=True)
    dh = xgb.DMatrix(R["X"][h])
    pv = (bv[h] + models["v"].predict(dh)) * R["vf"][h]
    pq = bq[h] + models["q"].predict(dh) * R["lanes"][h]
    s_model, per = score(pv, pq, R["yv"][h], R["yq"][h], R["lanes"][h], R["panel"][h], R["regime"][h])
    s_base, _ = score(bv[h] * R["vf"][h], bq[h], R["yv"][h], R["yq"][h], R["lanes"][h], R["panel"][h], R["regime"][h])
    rv = np.sqrt(np.mean((pv - R["yv"][h]) ** 2)); rq = np.sqrt(np.mean(((pq - R["yq"][h]) / R["lanes"][h]) ** 2))
    report = dict(S_state_holdout=s_model, S_state_base=s_base, rmse_v=float(rv), rmse_q_lane=float(rq),
                  per_panel=per, best_iter=best, features=names)
    print(json.dumps({k: v for k, v in report.items() if k != "features"}, indent=1), flush=True)
    (OUT / "t1_report.json").write_text(json.dumps(report, indent=1))
    if a.save_hold:
        for key, m in models.items():
            m[: m.best_iteration + 1].save_model(str(OUT / f"t1h_{key}.json"))
    if a.no_predict:
        return
    # Refit on every train day with the holdout's best iteration count (+10% for the extra month).
    final = {key: fit(R["X"], y, w, rounds=int(best[key] * 1.1), device=a.device) for key, y in (("v", tv), ("q", tq))}
    for key, m in final.items():
        m.save_model(str(OUT / f"t1_{key}.json"))
    del R, parts
    print(f"refit done {time.time() - t0:.0f}s", flush=True)
    for s in ("validation", "private"):
        ms = self_train(final, a.panels, s, a.self_frac, a.self_rounds, a.self_lr, a.device, a.workers) if a.self_rounds else final
        for p in a.panels:
            print(predict_split((p, s, ms, names, a.month_prof)), flush=True)
    print(f"Task 1 done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
