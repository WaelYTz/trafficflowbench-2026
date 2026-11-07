"""Space-time ongoing net v2 for Task 2 (t2_stnet.py stays frozen: it is the submitted ST model).

v2 = t2_stnet.STNet plus
  * same-day context: the masked layer at slots T-36..T-13 (two hours before the 60-minute window, stamped < T)
    as a (4, 24, L) uint8 grid (speed / vf, flow / cap, density / kc, on-ramp inflow; 255 = unseen). Training rows
    read it from the MASKED train layer (vm / qm), as only the masked layer exists for those slots on the scored
    splits (official histories cover T-12..T-1 only). A small strided 2D encoder (24 -> 12 -> 6 slots) turns it
    into a per-link embedding that is added to the window embedding, so the net can see when a queue started
    and how it grew;
  * configurable width / dilated link blocks;
  * optional refit on every training window with a fixed epoch (no early stopping).
X (standardised float16), grids G and contexts C stay in CPU RAM and are moved per batch (GPU memory budget).
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from t2_stnet import (DEV, G_CH, H, Block, Block2d, grid_channels, make_grid, ramp_maps, stats, window_loss,  # noqa: F401
                      panel_iou)

CFG = dict(epochs=24, width=128, tw=32, cw=24, lr=2e-3, wd=1e-4, bs=64, drop=0.1, dil=(1, 2, 4, 8, 16, 1, 2, 4),
           ctx=True, stop=None)
C_SL = 24          # context slots T-36 .. T-13
C_RAW = 4          # stored context channels
C_CH = 7           # network context channels (ctx_channels)
C_HI = (1.6, 2.0, 6.0, 3000.0)


# ----------------------------------------------------------------------------------------------- context
def make_ctx(st, Vm, Qm, R, a_or):
    """Vm, Qm: (C_SL, L) raw masked-layer speed / flow for slots T-36..T-13 (NaN = unseen), R: (C_SL, n_ramps) raw
    ramp flows. Returns (4, C_SL, L) uint8 codes (255 = unseen)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        vn = Vm / st.vf[None]
        qn = Qm / st.cap[None]
        kn = (Qm / np.maximum(Vm, 1)) / st.kc[None]
    ro = np.nan_to_num(np.asarray(R, np.float64)) @ a_or
    out = np.empty((C_RAW, C_SL, st.L), np.uint8)
    for k, (a, hi) in enumerate(zip((vn, qn, kn, ro), C_HI)):
        fin = np.isfinite(a)
        out[k] = np.where(fin, np.round(np.clip(np.where(fin, a, 0), 0, hi) * (254 / hi)), 255).astype(np.uint8)
    return out


def ctx_from_layer(st, vs, qs, rq, d, t, a_or):
    """Context for day index d, origin slot t from masked-layer arrays vs / qs ((days, slots, links), file order)
    and ramp flows rq ((days, slots, ramps)). Reads slots t-36 .. t-13 only (all < T)."""
    o = st.order
    a, b = t - H - C_SL, t - H
    Vm = np.full((C_SL, st.L), np.nan); Qm = np.full((C_SL, st.L), np.nan)
    R = np.zeros((C_SL, rq.shape[2]), np.float32)
    lo = max(a, 0)
    if b > lo:
        Vm[lo - a:] = np.asarray(vs[d, lo:b], float)[:, o]
        Qm[lo - a:] = np.asarray(qs[d, lo:b], float)[:, o]
        R[lo - a:] = np.asarray(rq[d, lo:b], np.float32)
    return make_ctx(st, Vm, Qm, R, a_or)


def ctx_channels(C):
    """(B, 4, C_SL, L) uint8 -> (B, C_CH, C_SL, L) float32."""
    c = C.float()
    m = c < 255
    val = [c[:, k] * (hi / 254) for k, hi in enumerate(C_HI)]
    z = lambda a, mk, cc, s: torch.where(mk, (a - cc) / s, torch.zeros_like(a))
    mv, mq, mk = m[:, 0], m[:, 1], m[:, 2]
    ch = [z(val[0], mv, 0.8, 0.3), z(val[1], mq, 0.5, 0.3), z(val[2], mk, 1.0, 1.0), mv.float(), mq.float(),
          (mv & (val[0] <= 0.6)).float(), val[3] / 1000]
    return torch.stack(ch, 1)


# ----------------------------------------------------------------------------------------------- model
class STNet2(nn.Module):
    def __init__(self, f, c=128, tw=32, cw=24, nt=H + 1, tdil=(1, 2, 4, 1), dil=(1, 2, 4, 8, 16, 1, 2, 4), drop=0.1, ctx=True):
        super().__init__()
        self.ctx = ctx
        self.g_in = nn.Conv2d(G_CH, tw, 1)
        self.g_blocks = nn.Sequential(*[Block2d(tw, d, drop) for d in tdil])
        self.g_norm = nn.Sequential(nn.GroupNorm(8, tw), nn.GELU())
        self.g_out = nn.Conv1d(tw * nt, c, 1)
        if ctx:
            self.c_in = nn.Conv2d(C_CH, cw, 3, stride=(2, 1), padding=1)            # 24 -> 12 slots
            self.c_blocks = nn.Sequential(*[Block2d(cw, d, drop) for d in (1, 2, 4)])
            self.c_down = nn.Conv2d(cw, cw, 3, stride=(2, 1), padding=1)           # 12 -> 6 slots
            self.c_norm = nn.Sequential(nn.GroupNorm(8, cw), nn.GELU())
            self.c_out = nn.Conv1d(cw * (C_SL // 4), c, 1)
        self.x_in = nn.Conv1d(2 * f, c, 1)
        self.blocks = nn.Sequential(*[Block(c, d) for d in dil])
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.GELU(), nn.Conv1d(c, 6, 1))

    def forward(self, x, g, cx=None):
        g = self.g_norm(self.g_blocks(self.g_in(g)))
        h = self.g_out(g.flatten(1, 2)) + self.x_in(x)
        if self.ctx:
            q = self.c_norm(self.c_down(self.c_blocks(self.c_in(cx))))
            h = h + self.c_out(q.flatten(1, 2))
        return self.out(self.blocks(h))


def build(f, c):
    return STNet2(f, c["width"], c["tw"], c["cw"], dil=tuple(c["dil"]), drop=c["drop"], ctx=c["ctx"]).to(DEV)


# ----------------------------------------------------------------------------------------------- data
def pack_rows(rows):
    """Row dicts (X (L, F), G (5, 13, L), C (4, 24, L), optional y / elig) -> per-panel packs (order in pack["idx"])."""
    by = {}
    for i, r in enumerate(rows):
        by.setdefault(r["panel"], []).append(i)
    packs = []
    for p, ix in by.items():
        L = rows[ix[0]]["X"].shape[0]
        pk = dict(panel=p, idx=np.array(ix), X=np.stack([rows[i]["X"] for i in ix]), G=np.stack([rows[i]["G"] for i in ix]))
        if "C" in rows[ix[0]]:
            pk["C"] = np.stack([rows[i]["C"] for i in ix])
        if "y" in rows[ix[0]]:
            pk["y"] = np.stack([rows[i]["y"] for i in ix]).astype(bool); pk["e"] = np.stack([rows[i]["elig"] for i in ix]).astype(bool)
        else:
            pk["y"] = np.zeros((len(ix), 6, L), bool); pk["e"] = np.zeros((len(ix), 6, L), bool)
        packs.append(pk)
    return packs


def to_dev(pk, keep, mu, sd, chunk=512):
    """X standardised to float16 (N, F, L) in CPU RAM (NaN kept); y / e uint8 on the GPU; G / C stay where they are
    (CPU arrays or memmaps). Batches are gathered on the CPU and moved per step (GPU memory budget)."""
    N = len(pk["X"])
    Xh = np.empty((N, len(keep), pk["X"].shape[1]), np.float16)
    for a in range(0, N, chunk):
        x = (np.asarray(pk["X"][a:a + chunk][:, :, keep], np.float32) - mu) / sd
        Xh[a:a + chunk] = np.clip(x, -6, 6).transpose(0, 2, 1).astype(np.float16)
    return dict(panel=pk["panel"], X=Xh, G=pk["G"], C=pk.get("C"), y=torch.from_numpy(np.asarray(pk["y"], np.uint8)).to(DEV),
                e=torch.from_numpy(np.asarray(pk["e"], np.uint8)).to(DEV), n=N)


def trim():
    """Drop this process's working set (memory-mapped row pages) once X sits on the GPU (Windows)."""
    try:
        import ctypes
        k32, ps = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        ps.EmptyWorkingSet.argtypes = [ctypes.c_void_p]
        ps.EmptyWorkingSet(k32.GetCurrentProcess())
    except Exception:
        pass


def inputs(t, ix, ctx):
    """ix: sorted numpy indices into the pack."""
    it = torch.from_numpy(ix).to(DEV)
    sl = slice(ix[0], ix[-1] + 1) if ix[-1] - ix[0] + 1 == len(ix) else ix
    x = torch.from_numpy(t["X"][sl]).to(DEV).float()
    m = torch.isfinite(x)
    x = torch.cat([torch.where(m, x, torch.zeros_like(x)), m.float()], 1)
    g = grid_channels(torch.from_numpy(np.asarray(t["G"][sl], np.float16)).to(DEV))
    c = ctx_channels(torch.from_numpy(np.asarray(t["C"][sl], np.uint8)).to(DEV)) if ctx else None
    return it, x, g, c


def batch_plan(tens, bs, rng):
    plan = []
    for k, t in enumerate(tens):
        ix = rng.permutation(t["n"])
        plan += [(k, np.sort(ix[i:i + bs])) for i in range(0, len(ix), bs)]
    return [plan[i] for i in rng.permutation(len(plan))]


def evaluate(model, tens, ctx, bs=64):
    model.eval()
    losses, probs = [], []
    with torch.no_grad():
        for t in tens:
            pl = []
            for a in range(0, t["n"], bs):
                it, x, g, c = inputs(t, np.arange(a, min(a + bs, t["n"])), ctx)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
                    lo = model(x, g, c)
                lo = lo.float()
                losses.append(window_loss(lo, t["y"][it].float(), t["e"][it].float(), True).cpu().numpy())
                pl.append(torch.sigmoid(lo).cpu().numpy())
            probs.append(np.concatenate(pl))
    return np.concatenate(losses), probs


def train(fit, val, keep, seed=7, log=print, cfg=None):
    """fit / val: per-panel packs (X, G, C memmaps; y, e). cfg["stop"] = k: no early stopping, return the state after
    epoch k (0-based; val may then be part of fit, it is only reported). Returns a bundle."""
    c = dict(CFG, **(cfg or {}))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    mu, sd = stats(fit, keep)
    tf = [to_dev(pk, keep, mu, sd) for pk in fit]
    tv = [to_dev(pk, keep, mu, sd) for pk in val]
    trim()
    log(f"st2 data: fit {sum(t['n'] for t in tf)} val {sum(t['n'] for t in tv)} windows ({time.time() - t0:.0f}s) cfg {c}")
    model = build(len(keep), c)
    log(f"params {sum(p.numel() for p in model.parameters())}")
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["wd"])
    steps = c["epochs"] * len(batch_plan(tf, c["bs"], np.random.default_rng(0)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=c["lr"], total_steps=steps)
    best, best_state, best_ep = 1e9, None, -1
    for ep in range(c["epochs"]):
        model.train()
        tl = []
        for k, ix in batch_plan(tf, c["bs"], rng):
            t = tf[k]
            it, x, g, cx = inputs(t, ix, c["ctx"])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
                lo = model(x, g, cx)
            loss = window_loss(lo.float(), t["y"][it].float(), t["e"][it].float())
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tl.append(loss.item())
        vls, vps = evaluate(model, tv, c["ctx"])
        vl = float(vls.mean())
        log(f"  ep {ep:2d} train {np.mean(tl):.5f} val {vl:.5f} iou@.45 {panel_iou(vps, val):.4f} ({time.time() - t0:.0f}s, "
            f"gpu peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB)")
        if c["stop"] is not None:
            if ep == c["stop"]:
                best, best_ep, best_state = vl, ep, {k_: v.detach().clone() for k_, v in model.state_dict().items()}
                break
        elif vl < best:
            best, best_ep, best_state = vl, ep, {k_: v.detach().clone() for k_, v in model.state_dict().items()}
    log(f"st2 seed {seed}: best epoch {best_ep}, val loss {best:.5f} ({time.time() - t0:.0f}s)")
    del tf, tv
    torch.cuda.empty_cache()
    return dict(state={k_: v.cpu() for k_, v in best_state.items()}, mu=mu, sd=sd, keep=list(keep), cfg=c, key="ongoing",
                best_epoch=best_ep, val_loss=best, version=2)


def load(bundle):
    m = build(len(bundle["keep"]), bundle["cfg"])
    m.load_state_dict(bundle["state"])
    m.eval()
    return m


def predict(bundle, rows, model=None, bs=64):
    """(6, L) probabilities per row dict (X, G, C, panel), or a list of (N, 6, L) per pack when given packs."""
    model = model or load(bundle)
    packed = isinstance(rows[0], dict) and "idx" in rows[0]
    packs = rows if packed else pack_rows(rows)
    out = [None] * (0 if packed else len(rows))
    res = []
    for pk in packs:
        t = to_dev(pk, bundle["keep"], bundle["mu"], bundle["sd"])
        _, (p,) = evaluate(model, [t], bundle["cfg"]["ctx"], bs)
        res.append(p)
        if not packed:
            for j, i in enumerate(pk["idx"]):
                out[i] = p[j]
        del t
    return res if packed else out


def attach_ctx(panel: str, split: str, blocks: list) -> list:
    """Add C (masked layer of `split` at slots T-36..T-13, ramp flows at the same slots) to scored-window blocks."""
    import pandas as pd
    from common import REL, days, load as cload
    from t2_data import Static
    st = Static(panel)
    a_or, _ = ramp_maps(st)
    wi = pd.read_csv(REL / "task2" / panel / split / "window_index.csv").set_index("window_id")
    dl = {d: i for i, d in enumerate(days(panel, split))}
    vs = cload(panel, split, "v", mmap=True); qs = cload(panel, split, "q", mmap=True); rq = cload(panel, split, "rq", mmap=True)
    for b in blocks:
        origin = str(wi.loc[b["window_id"], "forecast_origin"])
        slot = int(origin[11:13]) * 12 + int(origin[14:16]) // 5
        b["C"] = ctx_from_layer(st, vs, qs, rq, dl[origin[:10]], slot, a_or)
    return blocks
