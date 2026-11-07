"""Onset (T+30) with the corridor CNN of t2_nn.py, trained on the step-6 cells only, compared with and
blended into the LightGBM rec7c onset blend (shift-adaptive weight) on Feb and Feb / March pseudo-windows."""
from __future__ import annotations

import argparse
import pickle

import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn

from common import OUT
from t2_model import HOLD, MEMORY, aggregate, decode, iou, predict_window
from t2_nn import DEV, Net, batches, predict, tensors, window_loss
from t2_onset_abl import LONG, onset_rows
from t2_onset_recent import refresh

COND = "queue_onset"


def onset_only(data):
    out = []
    for x, y, e in data:
        e = e.copy(); e[:5] = 0
        out.append((x, y, e))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    tr, ev, names = onset_rows()
    feb = [r for r in ev if r["day"] >= HOLD]
    ps = {s: pickle.load(open(OUT / f"t2_pseudo_{s}.pkl", "rb"))[0]["onset"] for s in ("feb", "mar", "feb_off", "mar_off")}
    refresh({"train": tr + feb + ps["feb"] + ps["feb_off"], "validation": ps["mar"] + ps["mar_off"]}, names, 7)
    keep = [i for i, n in enumerate(names) if not n.startswith(LONG)]
    kb = [i for i, n in enumerate(names) if not n.startswith(MEMORY)]
    fit_rows = [r for r in tr if r["day"] < HOLD]; val_rows = [r for r in tr if r["day"] >= HOLD]
    allx = np.concatenate([r["X"][:, keep] for r in fit_rows])
    mu = np.nanmean(allx, 0).astype(np.float32); sd = (np.nanstd(allx, 0) + 1e-6).astype(np.float32)
    del allx
    dtr, dva = onset_only(tensors(fit_rows, keep, mu, sd)), onset_only(tensors(val_rows, keep, mu, sd))
    print(f"train windows {len(dtr)}, val windows {len(dva)}", flush=True)
    model = Net(len(keep), a.width).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-3)
    steps = a.epochs * ((len(dtr) + 31) // 32)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps)
    best, best_state = 1e9, None
    for ep in range(a.epochs):
        model.train()
        for X, Y, E, _ in batches(dtr, 32, True, rng):
            loss = window_loss(model(X), Y, E)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            vl = np.mean([window_loss(model(X), Y, E).item() for X, Y, E, _ in batches(dva, 64, False)])
        if vl < best:
            best, best_state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep % 5 == 4:
            print(f"epoch {ep:2d} feb {vl:.5f} best {best:.5f}", flush=True)
    model.load_state_dict(best_state)
    torch.save(dict(state=best_state, mu=mu, sd=sd, keep=keep, width=a.width), OUT / f"t2nn_onset_s{a.seed}.pt")
    mr = lgb.Booster(model_file=str(OUT / "t2rec7_onset.txt")); mb = lgb.Booster(model_file=str(OUT / "t2abl_base_onset.txt"))
    io, ir = names.index("onset_prior"), names.index("recent_onset")

    def wgt(r):
        x, y = r["X"][:, ir], r["X"][:, io]
        return 0.3 if x.sum() == 0 or y.sum() == 0 else 0.5 * float(x @ y / np.linalg.norm(x) / np.linalg.norm(y))

    res = {}
    PN_all = {}
    for s, rs in {"feb": feb, **{("feb_ps" if k == "feb" else k): v for k, v in ps.items()}}.items():
        pn = predict(model, tensors(rs, keep, mu, sd))
        PN_all[s] = pn
        pl = [(1 - wgt(r)) * predict_window(mr, r["X"][:, keep], COND) + wgt(r) * predict_window(mb, r["X"][:, kb], COND) for r in rs]
        for name, P in (("lgb rec7c", pl), ("nn", pn), ("0.7 lgb + 0.3 nn", [0.7 * x + 0.3 * y for x, y in zip(pl, pn)]),
                        ("0.5 lgb + 0.5 nn", [0.5 * x + 0.5 * y for x, y in zip(pl, pn)])):
            for thr in (0.3, 0.4):
                res.setdefault(f"{name} @{thr}", {})[s] = aggregate(
                    [(r["panel"], COND, iou(decode(p, COND, thr, 3), r["y"], r["elig"])) for p, r in zip(P, rs)])["overall"]
    pickle.dump(PN_all, open(OUT / f"t2nn_onset_s{a.seed}_probs.pkl", "wb"))
    for name, d in res.items():
        print(f"{name:22s} | " + " | ".join(f"{s} {v:.4f}" for s, v in d.items()), flush=True)


if __name__ == "__main__":
    main()
