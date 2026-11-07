"""Corridor CNN for Task 2 (shared by the research scripts and t2_pipeline.py).

A window is a sequence over links in travel order. Its channels are the per-link features of the
LightGBM models, standardised, with a missing flag per feature. Dilated residual 1D convolutions give
~60 links of context, and the head outputs the six horizon steps for every link. Training minimises the
per-window mean BCE over eligible cells (onset: the T+30 step only). The best epoch by Feb loss is kept.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CFG = {"ongoing": dict(epochs=30, width=128, lr=2e-3, wd=1e-4, bs=64),
       "onset": dict(epochs=40, width=96, lr=2e-3, wd=1e-3, bs=32)}


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


def tensors(rows, keep, mu, sd, onset=False):
    out = []
    for r in rows:
        x = r["X"][:, keep].astype(np.float32)
        m = np.isfinite(x)
        z = np.where(m, np.clip((x - mu) / sd, -6, 6), 0).astype(np.float32)
        y = r["y"].astype(np.float32) if "y" in r else np.zeros((6, len(x)), np.float32)
        e = r["elig"].astype(np.float32) if "elig" in r else np.zeros((6, len(x)), np.float32)
        if onset:
            e = e.copy(); e[:5] = 0
        out.append((np.concatenate([z, m.astype(np.float32)], 1).T, y, e))
    return out


def batches(data, bs, shuffle, rng=None, aug=None):
    idx = np.arange(len(data))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        part = [data[j] for j in idx[i:i + bs]]
        if aug is not None:
            part = [(aug(x), y, e) for x, y, e in part]
        L = max(p[0].shape[1] for p in part)
        X = np.zeros((len(part), part[0][0].shape[0], L), np.float32)
        Y = np.zeros((len(part), 6, L), np.float32); E = np.zeros_like(Y)
        for k, (x, y, e) in enumerate(part):
            X[k, :, :x.shape[1]] = x; Y[k, :, :y.shape[1]] = y; E[k, :, :e.shape[1]] = e
        yield torch.from_numpy(X).to(DEV), torch.from_numpy(Y).to(DEV), torch.from_numpy(E).to(DEV), [p[0].shape[1] for p in part]


def window_loss(logit, y, e):
    l = nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none") * e
    return (l.sum((1, 2)) / e.sum((1, 2)).clamp(min=1)).mean()


def train(fit_rows, val_rows, keep, key, seed, log=print, aug=None):
    """Fit on fit_rows, keep the best epoch by val_rows loss. Returns a dict with the model and scaling.
    aug, if given, maps each training window's (channels, links) array to an augmented copy."""
    c = CFG[key]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    allx = np.concatenate([r["X"][:, keep] for r in fit_rows[::5 if key == "ongoing" else 1]])
    mu = np.nanmean(allx, 0).astype(np.float32); sd = (np.nanstd(allx, 0) + 1e-6).astype(np.float32)
    del allx
    onset = key == "onset"
    dtr, dva = tensors(fit_rows, keep, mu, sd, onset), tensors(val_rows, keep, mu, sd, onset)
    model = Net(len(keep), c["width"]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["wd"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=c["lr"], total_steps=c["epochs"] * ((len(dtr) + c["bs"] - 1) // c["bs"]))
    best, best_state, best_ep = 1e9, None, -1
    for ep in range(c["epochs"]):
        model.train()
        for X, Y, E, _ in batches(dtr, c["bs"], True, rng, aug):
            loss = window_loss(model(X), Y, E)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([window_loss(model(X), Y, E).item() for X, Y, E, _ in batches(dva, 64, False)]))
        if vl < best:
            best, best_ep, best_state = vl, ep, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    log(f"nn {key} seed {seed}: best epoch {best_ep}, val loss {best:.5f}")
    return dict(state={k: v.cpu() for k, v in best_state.items()}, mu=mu, sd=sd, keep=keep, width=c["width"], key=key)


def load(bundle):
    m = Net(len(bundle["keep"]), bundle["width"]).to(DEV)
    m.load_state_dict(bundle["state"])
    m.eval()
    return m


def predict(bundle, rows, model=None, bs=64):
    """(6, L) probabilities per row, in each row's own link order."""
    model = model or load(bundle)
    data = tensors(rows, bundle["keep"], bundle["mu"], bundle["sd"])
    out = []
    with torch.no_grad():
        for X, _, _, lens in batches(data, bs, False):
            p = torch.sigmoid(model(X)).cpu().numpy()
            out += [p[k, :, :n] for k, n in enumerate(lens)]
    return out
