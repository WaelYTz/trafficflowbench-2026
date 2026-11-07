"""Ongoing queues with a 1D convolutional network along the corridor (GPU).

Each window is a sequence over links (travel order) whose channels are the per-link features of the
LightGBM models (memory-free + 7-day recent memory, standardised, with missing flags). Dilated residual
convolutions give ~60 links of context; the head predicts the six horizon steps for every link.
Trained on Jun-Jan ongoing windows, early-stopped on Feb, then scored alone and blended with the
LightGBM recent-memory blend (rec7b) on Feb official-like and Feb / March pseudo-windows.
"""
from __future__ import annotations

import argparse
import pickle

import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn

from common import OUT
from t2_model import HOLD, MEMORY, aggregate, decode, iou, predict_window
from t2_onset_abl import LONG
from t2_onset_recent import refresh

COND = "queue_ongoing"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Block(nn.Module):
    def __init__(self, c, d):
        super().__init__()
        self.net = nn.Sequential(nn.GroupNorm(8, c), nn.GELU(), nn.Conv1d(c, c, 3, padding=d, dilation=d),
                                 nn.GroupNorm(8, c), nn.GELU(), nn.Conv1d(c, c, 1))

    def forward(self, x):
        return x + self.net(x)


class Net(nn.Module):
    def __init__(self, f, c=128, dil=(1, 2, 4, 8, 16, 1, 2, 4)):
        super().__init__()
        self.inp = nn.Conv1d(2 * f, c, 1)
        self.blocks = nn.Sequential(*[Block(c, d) for d in dil])
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.GELU(), nn.Conv1d(c, 6, 1))

    def forward(self, x):
        return self.out(self.blocks(self.inp(x)))


def tensors(rows, keep, mu, sd):
    """Per window: (2F, L) input, (6, L) labels and eligibility."""
    out = []
    for r in rows:
        x = r["X"][:, keep].astype(np.float32)
        m = np.isfinite(x)
        z = np.where(m, np.clip((x - mu) / sd, -6, 6), 0).astype(np.float32)
        out.append((np.concatenate([z, m.astype(np.float32)], 1).T, r["y"].astype(np.float32), r["elig"].astype(np.float32)))
    return out


def batches(data, bs, shuffle, rng=None):
    idx = np.arange(len(data))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        part = [data[j] for j in idx[i:i + bs]]
        L = max(p[0].shape[1] for p in part)
        X = np.zeros((len(part), part[0][0].shape[0], L), np.float32)
        Y = np.zeros((len(part), 6, L), np.float32); E = np.zeros_like(Y)
        for k, (x, y, e) in enumerate(part):
            X[k, :, :x.shape[1]] = x; Y[k, :, :y.shape[1]] = y; E[k, :, :e.shape[1]] = e
        yield torch.from_numpy(X).to(DEV), torch.from_numpy(Y).to(DEV), torch.from_numpy(E).to(DEV), [p[0].shape[1] for p in part]


def window_loss(logit, y, e):
    l = nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none") * e
    return (l.sum((1, 2)) / e.sum((1, 2)).clamp(min=1)).mean()


def predict(model, data, bs=64):
    model.eval()
    out = []
    with torch.no_grad():
        for X, _, _, lens in batches(data, bs, False):
            p = torch.sigmoid(model(X)).cpu().numpy()
            out += [p[k, :, :n] for k, n in enumerate(lens)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    tr_all, ev_all, names = pickle.load(open(OUT / "t2_rowse.pkl", "rb"))
    tr, ev = tr_all["ongoing"], ev_all["ongoing"]
    del tr_all, ev_all
    feb = [r for r in ev if r["day"] >= HOLD]
    ps = {s: pickle.load(open(OUT / f"t2_pseudo_{s}.pkl", "rb"))[0]["ongoing"] for s in ("feb", "mar", "feb_off", "mar_off")}
    refresh({"train": tr + feb + ps["feb"] + ps["feb_off"], "validation": ps["mar"] + ps["mar_off"]}, names, 7)
    keep = [i for i, n in enumerate(names) if not n.startswith(LONG)]
    kb = [i for i, n in enumerate(names) if not n.startswith(MEMORY)]
    fit_rows = [r for r in tr if r["day"] < HOLD]
    val_rows = [r for r in tr if r["day"] >= HOLD]
    allx = np.concatenate([r["X"][:, keep] for r in fit_rows[::5]])
    mu = np.nanmean(allx, 0).astype(np.float32); sd = (np.nanstd(allx, 0) + 1e-6).astype(np.float32)
    del allx
    dtr, dva = tensors(fit_rows, keep, mu, sd), tensors(val_rows, keep, mu, sd)
    print(f"train windows {len(dtr)}, val windows {len(dva)}, features {len(keep)}", flush=True)
    model = Net(len(keep), a.width).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * ((len(dtr) + 63) // 64))
    best, best_state = 1e9, None
    for ep in range(a.epochs):
        model.train()
        tl = 0
        for X, Y, E, _ in batches(dtr, 64, True, rng):
            loss = window_loss(model(X), Y, E)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tl += loss.item()
        model.eval()
        with torch.no_grad():
            vl = np.mean([window_loss(model(X), Y, E).item() for X, Y, E, _ in batches(dva, 64, False)])
        if vl < best:
            best, best_state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep:2d} train {tl / ((len(dtr) + 63) // 64):.5f} feb {vl:.5f}", flush=True)
    model.load_state_dict(best_state)
    torch.save(dict(state=best_state, mu=mu, sd=sd, keep=keep, width=a.width), OUT / f"t2nn_ongoing_s{a.seed}.pt")
    sets = {"feb": feb, **{("feb_ps" if k == "feb" else k): v for k, v in ps.items()}}
    lgb_cache = OUT / "t2nn_lgb_probs.pkl"
    if lgb_cache.exists():
        PL = pickle.load(open(lgb_cache, "rb"))
    else:
        mr = lgb.Booster(model_file=str(OUT / "t2og_rec7_ongoing.txt")); mb = lgb.Booster(model_file=str(OUT / "t2og_base_ongoing.txt"))
        PL = {s: [0.7 * predict_window(mr, r["X"][:, keep], COND) + 0.3 * predict_window(mb, r["X"][:, kb], COND) for r in rs]
              for s, rs in sets.items()}
        pickle.dump(PL, open(lgb_cache, "wb"))
    PN = {s: predict(model, tensors(rs, keep, mu, sd)) for s, rs in sets.items()}
    pickle.dump(PN, open(OUT / f"t2nn_ongoing_s{a.seed}_probs.pkl", "wb"))
    res = {}
    for s, rs in sets.items():
        pn, pl = PN[s], PL[s]
        for name, P in (("lgb rec7b", pl), ("nn", pn), ("0.7 lgb + 0.3 nn", [0.7 * x + 0.3 * y for x, y in zip(pl, pn)]),
                        ("0.5 lgb + 0.5 nn", [0.5 * x + 0.5 * y for x, y in zip(pl, pn)])):
            res.setdefault(name, {})[s] = aggregate([(r["panel"], COND, iou(decode(p, COND, 0.45), r["y"], r["elig"])) for p, r in zip(P, rs)])["overall"]
    pickle.dump(res, open(OUT / f"t2nn_ongoing_s{a.seed}_res.pkl", "wb"))
    for name, d in res.items():
        print(f"{name:18s} | " + " | ".join(f"{s} {v:.4f}" for s, v in d.items()), flush=True)


if __name__ == "__main__":
    main()
