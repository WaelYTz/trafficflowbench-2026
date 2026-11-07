"""Space-time ongoing net for Task 2 (experiment; t2_nnlib.py stays the production CNN).

Inputs per window, links in travel order:
  X  (L, F)        the per-link engineered features of t2_data.window_features (standardised, + missing flags)
  G  (5, 13, L)    raw grid over the 12 history slots + the masked row at T: speed / vf, flow / capacity,
                   density / kc, on-ramp and off-ramp flow attached to each link (vph). Every cell is stamped <= T.
A space-time encoder (2D residual convs over time x links, dilated along the corridor) turns the grid into a
per-link embedding (time flattened), which is added to the projection of X; the dilated 1D residual blocks of
t2_nnlib then give ~60 links of context and the head outputs the six horizon steps. Loss: per-window mean BCE
over eligible cells, best epoch by Feb loss (as t2_nnlib).

Data are held as per-panel packs (every window of a panel has the same L), standardised to float16 on the GPU,
so batches never need padding.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import REL, network

DEV = "cuda" if torch.cuda.is_available() else "cpu"
H = 12
CFG = dict(epochs=24, width=128, tw=32, lr=2e-3, wd=1e-4, bs=64, drop=0.1)
G_CH = 8          # channels the grid becomes on the GPU (see grid_channels)


# ----------------------------------------------------------------------------------------------- grids
def ramp_maps(st):
    """(n_ramps, L) attachment matrices for on- and off-ramps, as t2_data.Recent builds them."""
    o = st.order
    net = network(st.panel)
    rank = np.empty(st.L, int); rank[o] = np.arange(st.L)
    li = {l: i for i, l in enumerate(net.link_id)}
    m = pd.read_csv(REL / "corridors" / st.panel / "network" / "ramp_attachment_map.csv", dtype=str)
    pos = np.array([rank[li[x]] if x in li else -1 for x in m.nearest_mainline_link_id])
    a_or = np.zeros((len(m), st.L), np.float32); a_fr = np.zeros((len(m), st.L), np.float32)
    for k, (kind, ps) in enumerate(zip(m.ramp_type, pos)):
        if ps >= 0:
            (a_or if kind == "OR" else a_fr)[k, ps] = 1
    return a_or, a_fr


def make_grid(st, V, Q, R13, a_or, a_fr):
    """V, Q: (13, L) raw speed / flow for slots T-12..T (row 12 = T, NaN where unseen); R13: (13, n_ramps)
    raw ramp flows for the same slots. Returns (5, 13, L) float16."""
    with np.errstate(invalid="ignore", divide="ignore"):
        vn = V / st.vf[None]
        qn = Q / st.cap[None]
        kn = (Q / np.maximum(V, 1)) / st.kc[None]
    r = np.nan_to_num(np.asarray(R13, np.float32))
    return np.stack([vn, qn, kn, r @ a_or, r @ a_fr]).astype(np.float16)


def grid_channels(G):
    """(B, 5, 13, L) float16 raw -> (B, G_CH, 13, L) float32 network input."""
    G = G.float()
    vn, qn, kn, ro, rf = G.unbind(1)
    mv, mq = torch.isfinite(vn), torch.isfinite(qn)
    z = lambda a, m, lo, hi, c, s: torch.where(m, (a.clamp(lo, hi) - c) / s, torch.zeros_like(a))
    ch = [z(vn, mv, 0, 1.6, 0.8, 0.3), z(qn, mq, 0, 2.0, 0.5, 0.3), z(kn, mq & mv, 0, 6.0, 1.0, 1.0),
          mv.float(), mq.float(), (mv & (vn <= 0.6)).float(), (ro / 1000).clamp(0, 3), (rf / 1000).clamp(0, 3)]
    return torch.stack(ch, 1)


# ----------------------------------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, c, d):
        super().__init__()
        self.net = nn.Sequential(nn.GroupNorm(8, c), nn.GELU(), nn.Conv1d(c, c, 3, padding=d, dilation=d),
                                 nn.GroupNorm(8, c), nn.GELU(), nn.Conv1d(c, c, 1))

    def forward(self, x):
        return x + self.net(x)


class Block2d(nn.Module):
    def __init__(self, c, d, drop=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.GroupNorm(8, c), nn.GELU(), nn.Conv2d(c, c, 3, padding=(1, d), dilation=(1, d)),
                                 nn.GroupNorm(8, c), nn.GELU(), nn.Dropout(drop), nn.Conv2d(c, c, 1))

    def forward(self, x):
        return x + self.net(x)


class STNet(nn.Module):
    def __init__(self, f, c=128, tw=32, nt=H + 1, tdil=(1, 2, 4, 1), dil=(1, 2, 4, 8, 16, 1, 2, 4), drop=0.1, use_x=True, use_g=True):
        super().__init__()
        self.use_x, self.use_g = use_x, use_g
        if use_g:
            self.g_in = nn.Conv2d(G_CH, tw, 1)
            self.g_blocks = nn.Sequential(*[Block2d(tw, d, drop) for d in tdil])
            self.g_norm = nn.Sequential(nn.GroupNorm(8, tw), nn.GELU())
            self.g_out = nn.Conv1d(tw * nt, c, 1)
        if use_x:
            self.x_in = nn.Conv1d(2 * f, c, 1)
        self.blocks = nn.Sequential(*[Block(c, d) for d in dil])
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.GELU(), nn.Conv1d(c, 6, 1))

    def forward(self, x, g):
        h = 0
        if self.use_g:
            g = self.g_norm(self.g_blocks(self.g_in(g)))      # (B, tw, 13, L)
            h = self.g_out(g.flatten(1, 2))                    # (B, c, L)
        if self.use_x:
            h = h + self.x_in(x)
        return self.out(self.blocks(h))


# ----------------------------------------------------------------------------------------------- data
def pack_rows(rows):
    """Group row dicts (X (L, F), G (5, 13, L), optional y / elig (6, L)) into per-panel packs, keeping the
    input order in pack["idx"]."""
    by = {}
    for i, r in enumerate(rows):
        by.setdefault(r["panel"], []).append(i)
    packs = []
    for p, ix in by.items():
        L = rows[ix[0]]["X"].shape[0]
        pk = dict(panel=p, idx=np.array(ix), X=np.stack([rows[i]["X"] for i in ix]), G=np.stack([rows[i]["G"] for i in ix]))
        if "y" in rows[ix[0]]:
            pk["y"] = np.stack([rows[i]["y"] for i in ix]).astype(bool); pk["e"] = np.stack([rows[i]["elig"] for i in ix]).astype(bool)
        else:
            pk["y"] = np.zeros((len(ix), 6, L), bool); pk["e"] = np.zeros((len(ix), 6, L), bool)
        packs.append(pk)
    return packs


def stats(packs, keep, every=5):
    """Mean / std per kept feature over every `every`-th window of the packs (NaN ignored)."""
    s = np.zeros(len(keep)); s2 = np.zeros(len(keep)); n = np.zeros(len(keep))
    for pk in packs:
        for i in range(0, len(pk["X"]), every):
            x = np.asarray(pk["X"][i][:, keep], np.float64)
            m = np.isfinite(x)
            s += np.where(m, x, 0).sum(0); s2 += np.where(m, x * x, 0).sum(0); n += m.sum(0)
    mu = s / np.maximum(n, 1)
    sd = np.sqrt(np.maximum(s2 / np.maximum(n, 1) - mu * mu, 0)) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def to_gpu(pk, keep, mu, sd, chunk=512):
    """Pack -> GPU tensors: X standardised (N, F, L) float16 with NaN kept, G (N, 5, 13, L) float16, y / e uint8."""
    N = len(pk["X"])
    Xg = torch.empty((N, len(keep), pk["X"].shape[1]), dtype=torch.float16, device=DEV)
    for a in range(0, N, chunk):
        x = (np.asarray(pk["X"][a:a + chunk][:, :, keep], np.float32) - mu) / sd
        Xg[a:a + chunk] = torch.from_numpy(np.clip(x, -6, 6).transpose(0, 2, 1).astype(np.float16)).to(DEV)
    Gg = torch.empty(pk["G"].shape, dtype=torch.float16, device=DEV)
    for a in range(0, N, chunk):
        Gg[a:a + chunk] = torch.from_numpy(np.asarray(pk["G"][a:a + chunk], np.float16)).to(DEV)
    w = torch.from_numpy(np.asarray(pk["w"], np.float32)).to(DEV) if "w" in pk else torch.ones(N, device=DEV)
    return dict(panel=pk["panel"], X=Xg, G=Gg, w=w, y=torch.from_numpy(np.asarray(pk["y"], np.uint8)).to(DEV),
                e=torch.from_numpy(np.asarray(pk["e"], np.uint8)).to(DEV), n=N)


def trim():
    """Drop this process's working set (memory-mapped row pages) once the data sit on the GPU (Windows)."""
    try:
        import ctypes
        ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())
    except Exception:
        pass


def inputs(t, ix):
    x = t["X"][ix].float()
    m = torch.isfinite(x)
    return torch.cat([torch.where(m, x, torch.zeros_like(x)), m.float()], 1), grid_channels(t["G"][ix])


def window_loss(logit, y, e, per_window=False):
    l = nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none") * e
    w = l.sum((1, 2)) / e.sum((1, 2)).clamp(min=1)
    return w if per_window else w.mean()


def panel_iou(probs, packs, thr=0.45):
    """Mean window IoU per panel, averaged over panels (probs: (N, 6, L) per pack)."""
    per = []
    for p, pk in zip(probs, packs):
        y, e = np.asarray(pk["y"], bool), np.asarray(pk["e"], bool)
        pr = (p > thr) & e; t = y & e
        u = (pr | t).sum((1, 2)); i = (pr & t).sum((1, 2))
        per.append(np.where(u == 0, 1.0, i / np.maximum(u, 1)).mean())
    return float(np.mean(per))


def batch_plan(tens, bs, rng):
    plan = []
    for k, t in enumerate(tens):
        ix = rng.permutation(t["n"])
        plan += [(k, ix[i:i + bs]) for i in range(0, len(ix), bs)]
    return [plan[i] for i in rng.permutation(len(plan))]


def evaluate(model, tens, bs=64):
    """Per-window val losses and probabilities (lists per pack)."""
    model.eval()
    losses, probs = [], []
    with torch.no_grad():
        for t in tens:
            pl = []
            for a in range(0, t["n"], bs):
                ix = torch.arange(a, min(a + bs, t["n"]), device=DEV)
                x, g = inputs(t, ix)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
                    lo = model(x, g)
                lo = lo.float()
                losses.append(window_loss(lo, t["y"][ix].float(), t["e"][ix].float(), True).cpu().numpy())
                pl.append(torch.sigmoid(lo).cpu().numpy())
            probs.append(np.concatenate(pl))
    return np.concatenate(losses), probs


def train(fit, val, keep, key="ongoing", seed=7, log=print, cfg=None, use_x=True, use_g=True):
    """Packs may carry "w" (N,) window weights: the training loss and the epoch-selection val loss are then
    weighted means over windows."""
    """fit / val: row dicts (with X, G, y, elig, panel) or per-panel packs (pack_rows / disk). Returns a bundle."""
    assert key == "ongoing"
    c = dict(CFG, **(cfg or {}))
    fit = fit if isinstance(fit[0], dict) and "idx" in fit[0] else pack_rows(fit)
    val = val if isinstance(val[0], dict) and "idx" in val[0] else pack_rows(val)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    mu, sd = stats(fit, keep)
    tf = [to_gpu(pk, keep, mu, sd) for pk in fit]
    tv = [to_gpu(pk, keep, mu, sd) for pk in val]
    trim()
    log(f"st data on {DEV}: fit {sum(t['n'] for t in tf)} val {sum(t['n'] for t in tv)} windows ({time.time() - t0:.0f}s)")
    model = STNet(len(keep), c["width"], c["tw"], drop=c["drop"], use_x=use_x, use_g=use_g).to(DEV)
    vw = torch.cat([t["w"] for t in tv]).cpu().numpy()
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["wd"])
    steps = c["epochs"] * len(batch_plan(tf, c["bs"], np.random.default_rng(0)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=c["lr"], total_steps=steps)
    best, best_state, best_ep = 1e9, None, -1
    for ep in range(c["epochs"]):
        model.train()
        tl = []
        for k, ix in batch_plan(tf, c["bs"], rng):
            t = tf[k]
            ix = torch.from_numpy(ix).to(DEV)
            x, g = inputs(t, ix)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
                lo = model(x, g)
            lw, w = window_loss(lo.float(), t["y"][ix].float(), t["e"][ix].float(), True), t["w"][ix]
            loss = (lw * w).sum() / w.sum()
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tl.append(loss.item())
        vls, vps = evaluate(model, tv)
        vl = float((vls * vw).sum() / vw.sum())
        log(f"  ep {ep:2d} train {np.mean(tl):.5f} val {vl:.5f} iou@.45 {panel_iou(vps, val):.4f} ({time.time() - t0:.0f}s)")
        if vl < best:
            best, best_ep, best_state = vl, ep, {k_: v.detach().clone() for k_, v in model.state_dict().items()}
    log(f"st {key} seed {seed}: best epoch {best_ep}, val loss {best:.5f} ({time.time() - t0:.0f}s)")
    del tf, tv
    torch.cuda.empty_cache()
    return dict(state={k_: v.cpu() for k_, v in best_state.items()}, mu=mu, sd=sd, keep=list(keep), cfg=c, use_x=use_x, use_g=use_g, key=key,
                best_epoch=best_ep, val_loss=best)


def load(bundle):
    c = bundle["cfg"]
    m = STNet(len(bundle["keep"]), c["width"], c["tw"], drop=c["drop"], use_x=bundle.get("use_x", True),
               use_g=bundle.get("use_g", True)).to(DEV)
    m.load_state_dict(bundle["state"])
    m.eval()
    return m


def predict(bundle, rows, model=None, bs=64):
    """(6, L) probabilities per row (rows: dicts with X, G, panel), in each row's own link order; or, given
    packs, a list of (N, 6, L) arrays per pack."""
    model = model or load(bundle)
    packed = isinstance(rows[0], dict) and "idx" in rows[0]
    packs = rows if packed else pack_rows(rows)
    out = [None] * (0 if packed else len(rows))
    res = []
    for pk in packs:
        t = to_gpu(pk, bundle["keep"], bundle["mu"], bundle["sd"])
        _, (p,) = evaluate(model, [t], bs)
        res.append(p)
        if not packed:
            for j, i in enumerate(pk["idx"]):
                out[i] = p[j]
        del t
    return res if packed else out


def attach_grids(panel: str, split: str, blocks: list, names: list) -> list:
    """Add G to t2_data.scored_windows(panel, split) blocks: the same 12 history slots from window_history.parquet
    and row T from the masked layer, plus ramp_states flows at T-12..T (everything stamped <= T)."""
    from common import days, load
    from t2_data import Static
    st = Static(panel); o = st.order
    a_or, a_fr = ramp_maps(st)
    hist = pd.read_parquet(REL / "task2" / panel / split / "window_history.parquet",
                           columns=["window_id", "timestamp", "link_id", "speed_kmh", "flow_vph"])
    wi = pd.read_csv(REL / "task2" / panel / split / "window_index.csv").set_index("window_id")
    li = {l: i for i, l in enumerate(network(panel).link_id)}
    rank = np.empty(st.L, int); rank[o] = np.arange(st.L)
    dl = {d: i for i, d in enumerate(days(panel, split))}
    vs = load(panel, split, "v", mmap=True); qs = load(panel, split, "q", mmap=True); rq = load(panel, split, "rq", mmap=True)
    for b in blocks:
        origin = str(wi.loc[b["window_id"], "forecast_origin"])
        slot = int(origin[11:13]) * 12 + int(origin[14:16]) // 5
        h = hist[hist.window_id == b["window_id"]]
        ts = h.timestamp.str
        s = (ts[11:13].astype(int) * 12 + ts[14:16].astype(int) // 5).to_numpy() - (slot - H)
        V = np.full((H + 1, st.L), np.nan); Q = np.full((H + 1, st.L), np.nan)
        lr = rank[h.link_id.map(li).to_numpy()]
        V[s, lr] = h.speed_kmh.to_numpy(); Q[s, lr] = h.flow_vph.to_numpy()
        d = dl[origin[:10]]
        V[H] = np.asarray(vs[d, slot], float)[o]; Q[H] = np.asarray(qs[d, slot], float)[o]
        iv = [names.index(f"v_s{k}") for k in range(H + 1)]
        assert np.allclose(V / st.vf[None], b["X"][:, iv].T, rtol=1e-4, atol=1e-5, equal_nan=True), b["window_id"]
        b["G"] = make_grid(st, V, Q, np.asarray(rq[d, slot - H:slot + 1], np.float32), a_or, a_fr)
    return blocks
