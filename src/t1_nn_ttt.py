"""Transductive fine-tuning of the neural Task 1 imputer on each scored split's own masked layer.

Task 1 is offline imputation (same principle as the GBDT self-train, LB-confirmed): the scored split's masked layer is
the only input. Self-supervised labels are OBSERVED cells hidden with the target pattern of another day of the same
panel / split (per link kept with prob --frac), so the hidden cells look like real targets (runs, outages, all-link
gaps). No ground truth of a scored split is used anywhere.

  python t1_nn_ttt.py eval --tag d64f --width 64 --split validation   # S2 holdout: base vs fine-tuned per epoch
  python t1_nn_ttt.py run --tags d64f d64f2 --width 64 --out-sub pred_d64t
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np
import torch
import torch.nn as nn

import t1_nn as M
from common import PANELS, T

OUTD = M.OUTD


def pattern(pd_, d, rng, frac):
    """(T, L) travel-order hide pattern for day d: target mask of another day, each link kept with prob frac."""
    d2 = int(rng.integers(pd_.D - 1)); d2 += d2 >= d
    tg2 = np.asarray(pd_.TG[d2])[:, pd_.order].astype(bool)
    return tg2 & (rng.random(pd_.L) < frac)[None, :]


def labels(pd_, d, hide, excl=None):
    """Labels at hidden observed eligible non-target cells (v/vf, q/cap) and their mask."""
    o = pd_.order
    v = np.asarray(pd_.V[d])[:, o].astype(np.float32); q = np.asarray(pd_.Q[d])[:, o].astype(np.float32)
    e = np.asarray(pd_.E[d])[:, o].astype(bool); tg = np.asarray(pd_.TG[d])[:, o].astype(bool)
    m = hide & np.isfinite(v) & np.isfinite(q) & e & ~tg
    if excl is not None:
        m &= ~excl
    y = np.stack([np.nan_to_num(v / pd_.vf), np.nan_to_num(q / pd_.cap)]).astype(np.float32)
    return y, m


def finetune(net, data, dev, epochs, lr, frac, seed, s2=None, every=None):
    """Fine-tunes net in place on the split's own observed cells; s2[pi][d] cells stay hidden and unlabelled."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * sum(-(-pd_.D // M.batch_size(pd_.L)) for pd_ in data)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1, final_div_factor=20)
    scaler = torch.amp.GradScaler()
    sv = [torch.from_numpy(pd_.vf / 25).to(dev) for pd_ in data]
    sq = [torch.from_numpy(pd_.cap / pd_.lanes / 600).to(dev) for pd_ in data]
    net.train()
    for ep in range(epochs):
        t0, tl, n = time.time(), 0.0, 0
        for pi, ds in M.batches(data, [np.arange(pd_.D) for pd_ in data], rng):
            pd_ = data[pi]
            xs, ys, ms = [], [], []
            for d in ds:
                d = int(d)
                ex = None if s2 is None else s2[pi][d]
                hide = pattern(pd_, d, rng, frac)
                y, m = labels(pd_, d, hide, ex)
                if ex is not None:
                    hide = hide | ex
                xs.append(pd_.sample(d, hide)[0]); ys.append(y); ms.append(m.astype(np.float32))
            x = torch.from_numpy(np.stack(xs)).to(dev).to(memory_format=torch.channels_last)
            y = torch.from_numpy(np.stack(ys)).to(dev); m = torch.from_numpy(np.stack(ms)).to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                p = net(x)
            loss = M.loss_fn(p.float(), y, m, sv[pi].expand(len(x), -1), sq[pi].expand(len(x), -1))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tl += float(loss) * 2 / len(x); n += 1
        print(f"  ft ep {ep + 1}/{epochs} loss {tl / n:.5f} {time.time() - t0:.0f}s", flush=True)
        if every is not None:
            every(ep + 1)


def load_net(tag, width, dev):
    net = M.Net(width).to(dev).to(memory_format=torch.channels_last)
    net.load_state_dict(torch.load(OUTD / f"net_{tag}.pt", map_location=dev))
    return net


def s2_eval(net, data, dev, s2, lab):
    """S_state (official structure) on the S2 cells, with S2 hidden from the inputs."""
    P, R, pv, pq, yv, yq, ln = [], [], [], [], [], [], []
    for pi, pd_ in enumerate(data):
        ds = np.arange(pd_.D)
        V, Q = M.predict_days(net, pd_, ds, dev, {int(d): s2[pi][d] for d in ds})
        for d in ds:
            y, m = lab[pi][d]
            t, l = np.nonzero(m)
            pv.append(V[d, t, l]); pq.append(Q[d, t, l])
            yv.append(y[0, t, l] * pd_.vf[l]); yq.append(y[1, t, l] * pd_.cap[l]); ln.append(pd_.lanes[l])
            P.append(np.full(len(t), pd_.pno)); R.append(np.full(len(t), int(pd_.regime[d])))
    a = [np.concatenate(z) for z in (pv, pq, yv, yq, ln, P, R)]
    rv = np.sqrt(np.mean((a[0] - a[2]) ** 2)); rq = np.sqrt(np.mean(((a[1] - a[3]) / a[4]) ** 2))
    return M.s_state(*a), float(rv), float(rq), len(a[0])


def evaluate(a):
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(a.seed)
    data = [M.PanelData(p, a.split) for p in PANELS]
    rng = np.random.default_rng(999)
    s2, lab = [], []
    for pd_ in data:
        s2.append([pattern(pd_, d, rng, a.frac) for d in range(pd_.D)])
        lab.append([labels(pd_, d, s2[-1][d]) for d in range(pd_.D)])
    net = load_net(a.tag, a.width, dev)
    res = []
    r = s2_eval(net, data, dev, s2, lab)
    print(f"{a.split} {a.tag} base      S {r[0]:.5f} rmse_v {r[1]:.3f} rmse_q_lane {r[2]:.2f} ({r[3]:,} S2 cells)", flush=True)
    res.append((0, *r))

    def every(ep):
        r = s2_eval(net, data, dev, s2, lab)
        print(f"{a.split} {a.tag} ft ep{ep}    S {r[0]:.5f} rmse_v {r[1]:.3f} rmse_q_lane {r[2]:.2f}", flush=True)
        res.append((ep, *r))
    finetune(net, data, dev, a.epochs, a.lr, a.frac, a.seed, s2, every)
    (OUTD / f"ttt_eval_{a.tag}_{a.split}.json").write_text(json.dumps(dict(args=vars(a), res=res), indent=1))


def run(a):
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    od = OUTD / a.out_sub
    od.mkdir(exist_ok=True)
    for s in ("validation", "private"):
        data = [M.PanelData(p, s) for p in PANELS]
        acc = [(np.zeros((pd_.D, T, pd_.L), np.float32), np.zeros((pd_.D, T, pd_.L), np.float32)) for pd_ in data]
        for k, tag in enumerate(a.tags):
            torch.manual_seed(a.seed + k)
            net = load_net(tag, a.width, dev)
            print(f"{s} {tag}: fine-tune {a.epochs} ep", flush=True)
            finetune(net, data, dev, a.epochs, a.lr, a.frac, a.seed + k)
            for pi, pd_ in enumerate(data):
                v, q = M.predict_days(net, pd_, np.arange(pd_.D), dev)
                acc[pi][0][:] += v / len(a.tags); acc[pi][1][:] += q / len(a.tags)
            del net; torch.cuda.empty_cache()
        for pi, pd_ in enumerate(data):
            tg = np.asarray(pd_.TG).astype(bool)
            Vs = np.full(tg.shape, np.nan, np.float32); Qs = np.full(tg.shape, np.nan, np.float32)
            Vs[:, :, pd_.order] = acc[pi][0]; Qs[:, :, pd_.order] = acc[pi][1]
            Vs[~tg] = np.nan; Qs[~tg] = np.nan
            np.savez_compressed(od / f"t1_{pd_.panel}_{s}.npz", V=Vs, Q=Qs)
            print(f"{pd_.panel} {s}: written", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["eval", "run"])
    ap.add_argument("--tag", default="d64f")
    ap.add_argument("--tags", nargs="*", default=["d64f"])
    ap.add_argument("--split", default="validation")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out-sub", default="pred_ttt")
    a = ap.parse_args()
    evaluate(a) if a.mode == "eval" else run(a)


if __name__ == "__main__":
    main()
