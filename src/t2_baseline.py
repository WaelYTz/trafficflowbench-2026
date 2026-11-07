"""Task 2 baseline. Uses only the train split and data stamped at or before each origin T.

ongoing windows: persistence of the queue state at T (the row at T if observed, else the latest
                 history observation), repeated over the six horizon steps.
onset windows:   the queue first appears at T+30 (window selector property, organizer-confirmed),
                 so only step 6 is predicted: the k links where onsets most often start at that time
                 of day in train ("place x time memory"); k is chosen on train by leave-one-month-out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import QUEUE_PANELS, REL, T, days, load, network


def queued_grid(panel: str, split: str):
    net = network(panel)
    v = load(panel, split, "v")
    e = load(panel, split, "elig")
    cut = net.v_cut.to_numpy()[None, None, :]
    return np.isfinite(v) & (v <= cut) & e, v, cut


def onset_events(panel: str):
    """(day, slot, queued links) of every train onset: a queue at t with none in the previous hour."""
    q, _, _ = queued_grid(panel, "train")
    active = q.any(axis=2)
    ev = []
    for d in range(q.shape[0]):
        for t in range(12, T):
            if active[d, t] and not active[d, t - 12:t].any():
                ev.append((d, t, np.flatnonzero(q[d, t])))
    return ev, q.shape[2]


def memory(ev, L, exclude_month=None, months=None):
    m = np.zeros((L, 24))
    for i, (d, t, links) in enumerate(ev):
        if exclude_month is not None and months[d] == exclude_month:
            continue
        m[links, t // 12] += 1
    return m


def score_links(mem: np.ndarray, slot: int) -> np.ndarray:
    h = slot // 12
    w = {-1: 0.5, 0: 1.0, 1: 0.5}
    return sum(wt * mem[:, (h + k) % 24] for k, wt in w.items())


def choose_k(panels) -> int:
    ious = {k: [] for k in range(1, 9)}
    for p in panels:
        ev, L = onset_events(p)
        months = [x[:7] for x in days(p, "train")]
        for mo in sorted(set(months)):
            mem = memory(ev, L, mo, months)
            for d, t, links in ev:
                if months[d] != mo:
                    continue
                s = score_links(mem, t)
                order = np.argsort(-s)
                truth = set(links.tolist())
                for k in ious:
                    pred = set(order[:k].tolist())
                    ious[k].append(len(pred & truth) / len(pred | truth))
    res = {k: float(np.mean(v)) for k, v in ious.items()}
    print("onset top-k IoU on train (leave-one-month-out):", {k: round(v, 3) for k, v in res.items()}, flush=True)
    return max(res, key=res.get)


def predict(k: int) -> dict:
    out = {}
    for p in QUEUE_PANELS:
        ev, L = onset_events(p)
        mem = memory(ev, L)
        cut = network(p).v_cut.to_numpy()
        for split in ("validation", "private"):
            wi = pd.read_csv(REL / "task2" / p / split / "window_index.csv")
            hist = pd.read_parquet(REL / "task2" / p / split / "window_history.parquet",
                                   columns=["window_id", "timestamp", "link_id", "speed_kmh"])
            li = {l: i for i, l in enumerate(network(p).link_id)}
            dl = {d: i for i, d in enumerate(days(p, split))}
            v_split = load(p, split, "v", mmap=True)
            for w in wi.itertuples(index=False):
                origin = str(w.forecast_origin)
                slot = int(origin[11:13]) * 12 + int(origin[14:16]) // 5
                pred = np.zeros((6, L), np.int8)
                if w.condition == "queue_onset":
                    s = score_links(mem, slot + 6)
                    pred[5, np.argsort(-s)[:k]] = 1
                else:
                    h = hist[hist.window_id == w.window_id].sort_values("timestamp")
                    last = np.full(L, np.nan)
                    for ts, g in h.groupby("timestamp"):
                        vals = g.speed_kmh.to_numpy()
                        idx = g.link_id.map(li).to_numpy()
                        ok = np.isfinite(vals)
                        last[idx[ok]] = vals[ok]
                    at_t = np.asarray(v_split[dl[origin[:10]], slot], float)  # the row at T is allowed
                    last = np.where(np.isfinite(at_t), at_t, last)
                    q = np.isfinite(last) & (last <= cut)
                    pred[:, q] = 1
                out[w.window_id] = pred
    return out


if __name__ == "__main__":
    k = choose_k(QUEUE_PANELS)
    print("k =", k)
