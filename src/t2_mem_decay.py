"""Recent-memory window shape after a regime switch.

Onset places are a per-split regime: train months agree with each other (monthly onset-place cosine
0.93-0.99), March breaks sharply at the split boundary (D12_I5_S 0.19), and every private onset window
lies on Apr 1-9, where a 7-day box memory still holds mostly March days. Variants of the memory columns
(recent_q, recent_n, recent_onset and their shifts; today_q_before unchanged):
  box7     last 7 days (the current recipe)
  box4     last 4 days
  exp2/3   last 21 days, weights halving every 2 / 3 days, scaled to sum 7
  split7   last 7 days of the window's own split, scaled to sum 7 (previous split only when none yet)
  split7h  same, but previous-split days kept at a quarter weight
  (On the first day of a split split7 falls back to the previous split's days. No March pseudo-window
  lies on Mar 1-2, so that fallback is untested here.)
The recent model is refit per variant (split7 / split7h reuse box7: train rows never cross a split) and
blended with the memory-free model as in t2_pipeline.py. Scored with the official aggregate and, as the
private-LB analog, on the early-month windows (day <= 9) of the March pseudo sets.
"""
from __future__ import annotations

import argparse
import pickle

import lightgbm as lgb
import numpy as np

from common import OUT, days
from t2_data import Recent, Static
from t2_model import HOLD, MEMORY, PARAMS, aggregate, decode, iou, predict_window, stack
from t2_onset_abl import LONG, onset_rows
from t2_onset_recent import recent_cols

VARIANTS = ("box7", "split7", "split7h", "box4", "exp2", "exp3")
HW = np.array([0.5, 1.0, 0.5])


def day_weights(var: str, c: int, start: int) -> tuple[np.ndarray, np.ndarray]:
    """Calendar-day indices before day c and their weights; start = first calendar day of c's split."""
    if var.startswith("box"):
        idx = np.arange(max(c - int(var[3:]), 0), c)
        return idx, np.ones(len(idx))
    if var.startswith("exp"):
        h = int(var[3:])
        idx = np.arange(max(c - 21, 0), c)
        full = 0.5 ** (np.arange(21) / h)
        return idx, 0.5 ** ((c - idx - 1) / h) * 7 / full.sum()
    idx = np.arange(max(c - 7, 0), c)
    own = idx >= start
    if not own.any():
        return idx, np.ones(len(idx))
    w = np.where(own, 1.0, 0.25 if var == "split7h" else 0.0)
    return idx, w * 7 / w.sum()


def refresh_var(rows_by_split, names, var: str):
    col = {n: i for i, n in enumerate(names)}
    for panel in sorted({r["panel"] for rs in rows_by_split.values() for r in rs}):
        rec = Recent(Static(panel))
        dq, do, don = (np.diff(a, axis=0) for a in (rec.cq, rec.co, rec.con))
        for split, rs in rows_by_split.items():
            di = {d: i for i, d in enumerate(days(panel, split))}
            for r in rs:
                if r["panel"] != panel:
                    continue
                c = rec.offset[split] + di[r["day"]]
                idx, w = day_weights(var, c, rec.offset[split])
                h = (r["slot"] + 6) // 12 % 24
                hrs = [(h - 1) % 24, h, (h + 1) % 24]
                q, n, on = (np.einsum("d,dlh,h->l", w, a[idx][:, :, hrs], HW) for a in (dq, do, don))
                base = rec.features(split, di[r["day"]], r["slot"])
                base.update(recent_q=q / np.maximum(n, 1), recent_n=n, recent_onset=on)
                for k, a in recent_cols(base).items():
                    r["X"][:, col[k]] = a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="onset", choices=("onset", "ongoing"))
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    a = ap.parse_args()
    if a.key == "onset":
        cond, hz, kw, thr, fb, rmodel, bmodel = "queue_onset", (6,), {}, 0.3, 3, "t2rec7_onset.txt", "t2abl_base_onset.txt"
        tr, ev, names = onset_rows()
    else:
        cond, hz, kw, thr, fb, rmodel, bmodel = ("queue_ongoing", (1, 2, 3, 4, 5, 6), dict(window_stride=3, n_h=3, far_keep=0.15),
                                                 0.45, 1, "t2og_rec7_ongoing.txt", "t2og_base_ongoing.txt")
        tr_all, ev_all, names = pickle.load(open(OUT / "t2_rowse.pkl", "rb"))
        tr, ev = tr_all["ongoing"], ev_all["ongoing"]
        del tr_all, ev_all
    feb = [r for r in ev if r["day"] >= HOLD]
    del ev
    ps = {s: pickle.load(open(OUT / f"t2_pseudo_{s}.pkl", "rb"))[0][a.key] for s in ("feb", "mar", "feb_off", "mar_off")}
    sets = {"feb": feb, "feb_ps": ps["feb"], "mar": ps["mar"], "feb_off": ps["feb_off"], "mar_off": ps["mar_off"]}
    groups = {"train": tr + feb + ps["feb"] + ps["feb_off"], "validation": ps["mar"] + ps["mar_off"]}
    kb = [i for i, n in enumerate(names) if not n.startswith(MEMORY)]
    kr = [i for i, n in enumerate(names) if not n.startswith(LONG)]
    nm = [names[i] for i in kr]
    io, ir = names.index("onset_prior"), names.index("recent_onset")
    base = lgb.Booster(model_file=str(OUT / bmodel))
    PB = {s: [predict_window(base, r["X"][:, kb], cond) for r in rs] for s, rs in sets.items()}

    def wbase(r):
        if a.key == "ongoing":
            return 0.3
        x, y = r["X"][:, ir], r["X"][:, io]
        return 0.3 if x.sum() == 0 or y.sum() == 0 else 0.5 * float(x @ y / np.linalg.norm(x) / np.linalg.norm(y))

    models = {}
    for var in a.variants:
        refresh_var(groups, names, var)
        fit_var = "box7" if var.startswith("split") else var
        if fit_var not in models:
            if fit_var == "box7" and (OUT / rmodel).exists():
                models[fit_var] = lgb.Booster(model_file=str(OUT / rmodel))
            else:
                sub = lambda rs: [dict(r, X=r["X"][:, kr]) for r in rs]
                X, y, w = stack(sub([r for r in tr if r["day"] < HOLD]), cond, hz, nm, seed=1, **kw)
                Xv, yv, wv = stack(sub([r for r in tr if r["day"] >= HOLD]), cond, hz, nm, seed=2, **kw)
                dt = lgb.Dataset(X, y, weight=w, feature_name=nm + ["h"])
                models[fit_var] = lgb.train(PARAMS, dt, 6000, valid_sets=[lgb.Dataset(Xv, yv, weight=wv, reference=dt)],
                                            callbacks=[lgb.early_stopping(150, verbose=False)])
                del X, y, w, Xv, yv, wv
                models[fit_var].save_model(str(OUT / f"t2mem_{fit_var}_{a.key}.txt"))
        m = models[fit_var]
        line, early = [], []
        for s, rs in sets.items():
            P = [(1 - wbase(r)) * predict_window(m, r["X"][:, kr], cond) + wbase(r) * pb for r, pb in zip(rs, PB[s])]
            sc = [iou(decode(p, cond, thr, fb), r["y"], r["elig"]) for p, r in zip(P, rs)]
            line.append(f"{s} {aggregate([(r['panel'], cond, x) for r, x in zip(rs, sc)])['overall']:.4f}")
            if s in ("mar", "mar_off"):
                e = [x for r, x in zip(rs, sc) if int(r["day"][8:10]) <= 9]
                early.append(f"{s} d<=9 {np.mean(e):.4f} (n={len(e)})")
        it = getattr(m, "best_iteration", 0)
        print(f"{a.key} {var:8s} it {it:5d} | " + " | ".join(line) + " || " + " | ".join(early), flush=True)


if __name__ == "__main__":
    main()
