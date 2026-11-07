"""Onset with extra any-hour recent memory: onsets and queued share per link over the last N days, all
hours, next to the per-hour recent memory. Scored with the official aggregation on Feb official-like
windows and Feb / March pseudo-windows, alone and blended with the memory-free base model."""
from __future__ import annotations

import pickle

import lightgbm as lgb
import numpy as np

from common import OUT, days
from t2_data import Recent, Static, _shift
from t2_model import HOLD, MEMORY, PARAMS, aggregate, decode, iou, predict_window, stack
from t2_onset_abl import LONG, onset_rows
from t2_onset_recent import refresh

COND = "queue_onset"
DAY = slice(60, 252)  # 05:00-21:00


def any_hour(rec: Recent, split: str, day: int, n_days: int) -> dict:
    c = rec.offset[split] + day
    a = max(c - n_days, 0)
    on = (rec.con[c] - rec.con[a]).sum(axis=1)
    q = (rec.cq[c] - rec.cq[a])[:, 5:21].sum(axis=1) / np.maximum((rec.co[c] - rec.co[a])[:, 5:21].sum(axis=1), 1)
    out = {}
    for k, v in (("any_onset", on), ("any_q", q)):
        out[k] = v
        for s in (-2, -1, 1, 2):
            out[f"{k}_{s:+d}"] = _shift(v, s)
    return out


def augment(rows_by_split, n_days):
    names = None
    for panel in sorted({r["panel"] for rs in rows_by_split.values() for r in rs}):
        rec = Recent(Static(panel))
        for split, rs in rows_by_split.items():
            di = {d: i for i, d in enumerate(days(panel, split))}
            for r in rs:
                if r["panel"] == panel:
                    f = any_hour(rec, split, di[r["day"]], n_days)
                    names = list(f)
                    r["X"] = np.column_stack([r["X"]] + [np.asarray(f[k], np.float32) for k in names])
    return names


def main():
    tr, ev, names = onset_rows()
    feb = [r for r in ev if r["day"] >= HOLD]
    ps = {s: pickle.load(open(OUT / f"t2_pseudo_{s}.pkl", "rb"))[0]["onset"] for s in ("feb", "mar")}
    groups = {"train": tr + feb + ps["feb"], "validation": ps["mar"]}
    refresh(groups, names, 7)
    base = lgb.Booster(model_file=str(OUT / "t2abl_base_onset.txt"))
    kb = [i for i, n in enumerate(names) if not n.startswith(MEMORY)]
    sets = {"feb": feb, "feb_ps": ps["feb"], "mar_ps": ps["mar"]}
    PB = {s: [predict_window(base, r["X"][:, kb], COND) for r in rs] for s, rs in sets.items()}
    extra = augment(groups, 7)
    allnames = names + extra
    variants = {"rec7": [i for i, n in enumerate(names) if not n.startswith(LONG)]}
    variants["rec7_any"] = variants["rec7"] + list(range(len(names), len(allnames)))
    for var, keep in variants.items():
        nm = [allnames[i] for i in keep]
        sub = lambda rs: [dict(r, X=r["X"][:, keep]) for r in rs]
        X, y, w = stack(sub([r for r in tr if r["day"] < HOLD]), COND, (6,), nm, seed=1)
        Xv, yv, wv = stack(sub([r for r in tr if r["day"] >= HOLD]), COND, (6,), nm, seed=2)
        dt = lgb.Dataset(X, y, weight=w, feature_name=nm + ["h"])
        m = lgb.train(PARAMS, dt, 4000, valid_sets=[lgb.Dataset(Xv, yv, weight=wv, reference=dt)],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        m.save_model(str(OUT / f"t2any_{var}_onset.txt"))
        for wb in (0.0, 0.3):
            line = []
            for s, rs in sets.items():
                P = [(1 - wb) * predict_window(m, r["X"][:, keep], COND) + wb * pb for r, pb in zip(rs, PB[s])]
                res = {thr: aggregate([(r["panel"], COND, iou(decode(p, COND, thr, 3), r["y"], r["elig"])) for p, r in zip(P, rs)])["overall"]
                       for thr in (0.2, 0.3, 0.45)}
                line.append(f"{s} " + " ".join(f"{res[t]:.4f}" for t in (0.2, 0.3, 0.45)))
            print(f"{var:9s} w_base {wb} it {m.best_iteration:4d} | thr .2/.3/.45 | " + " | ".join(line), flush=True)
        if var == "rec7_any":
            g = m.feature_importance("gain"); g = g / g.sum()
            print("  gain share any-hour:", round(float(sum(g[nm.index(k)] for k in extra)), 3),
                  "per-hour recent:", round(float(sum(g[i] for i, k in enumerate(nm) if k.startswith(MEMORY))), 3),
                  "pos/panel/slot:", round(float(sum(g[nm.index(k)] for k in ("pos", "panel", "slot"))), 3), flush=True)


if __name__ == "__main__":
    main()
