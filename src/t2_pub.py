"""Blend our Task 2 models with the public anubisyy queue models (CC0 assets in ext/anubis_assets).

Their onset model (LightGBM + CatBoost, 83 features) and ongoing model (LightGBM, 36 features) read
only the twelve history steps before T. They are scored on the Feb windows next to ours, alone and
blended, and blended queue files are written for the scored windows.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import OUT, QUEUE_PANELS, REL, ROOT, days, load, network
from t2_data import H, Static, scored_windows
from t2_model import HOLD, MEMORY, aggregate, decode, iou, predict_window

ASSETS = ROOT / "ext" / "anubis_assets"
os.environ.setdefault("TFB_DATA_ROOT", str(REL))
os.environ.setdefault("TFB_OUTPUT_ROOT", str(OUT / "anubis_tmp"))
os.environ.setdefault("TFB_THREADS", "8")
sys.path.insert(0, str(ASSETS))
import queue_features as QF  # noqa: E402
import queue_ongoing as QO  # noqa: E402
from catboost import CatBoostClassifier  # noqa: E402

CONDS = (("ongoing", "queue_ongoing"), ("onset", "queue_onset"))


class Public:
    def __init__(self):
        self.onset_lgb = lgb.Booster(model_file=str(ASSETS / "onset_lgb.txt"))
        self.onset_cat = CatBoostClassifier(); self.onset_cat.load_model(str(ASSETS / "onset.cbm"))
        self.ongoing = lgb.Booster(model_file=str(ASSETS / "ongoing_lgb.txt"))
        self.panels = {}

    def panel(self, name: str):
        if name not in self.panels:
            no = QUEUE_PANELS.index(name)
            st = Static(name)
            rank = np.empty(st.L, int); rank[st.order] = np.arange(st.L)
            sc = {l: i for i, l in enumerate(network(name).link_id)}
            onset_p, ongoing_p = QF.Panel(name, no), QO.Panel(name, no)
            # travel position of each of their (fd_parameters-ordered) links
            pos = rank[np.array([sc[l] for l in onset_p.links])]
            self.panels[name] = (onset_p, ongoing_p, pos)
        return self.panels[name]

    def probs(self, name: str, Vh, Qh, Eh, origin: pd.Timestamp, cond: str) -> np.ndarray:
        """Vh, Qh, Eh: (12, L) history T-60..T-5 in travel order. Returns (6, L) in travel order."""
        onset_p, ongoing_p, pos = self.panel(name)
        v, q, e = Vh[:, pos].astype(np.float32), Qh[:, pos].astype(np.float32), Eh[:, pos]
        L = len(pos)
        P = np.zeros((6, L))
        if cond == "queue_onset":
            X = onset_p.features(v, q, e.astype(np.float32), origin, cond)[-L:]
            p = 0.5 * self.onset_lgb.predict(X) + 0.5 * self.onset_cat.predict_proba(X)[:, 1]
            P[5, pos] = p
        else:
            X = ongoing_p.features(v, q, np.nan_to_num(e, nan=0).astype(bool), origin, cond)
            P[:, pos] = self.ongoing.predict(X).reshape(6, L)
        return P


def feb_windows(rows_tag: str):
    _, ev, names = pickle.load(open(OUT / f"t2_rows{rows_tag}.pkl", "rb"))
    return {k: [r for r in ev[k] if r["day"] >= HOLD] for k in ev}, names


def history_from_cache(panel: str, day: str, slot: int):
    st = Static(panel)
    d = days(panel, "train").index(day)
    o = st.order
    v = np.asarray(load(panel, "train", "v", mmap=True)[d, slot - H:slot], float)[:, o]
    q = np.asarray(load(panel, "train", "q", mmap=True)[d, slot - H:slot], float)[:, o]
    e = np.asarray(load(panel, "train", "elig", mmap=True)[d, slot - H:slot])[:, o]
    return v, q, e


def history_scored(panel: str, split: str):
    st = Static(panel)
    rank = np.empty(st.L, int); rank[st.order] = np.arange(st.L)
    li = {l: i for i, l in enumerate(network(panel).link_id)}
    wi = pd.read_csv(REL / "task2" / panel / split / "window_index.csv")
    hist = pd.read_parquet(REL / "task2" / panel / split / "window_history.parquet",
                           columns=["window_id", "timestamp", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"])
    out = {}
    for w in wi.itertuples(index=False):
        origin = str(w.forecast_origin)
        slot = int(origin[11:13]) * 12 + int(origin[14:16]) // 5
        h = hist[hist.window_id == w.window_id]
        ts = h.timestamp.str
        s = (ts[11:13].astype(int) * 12 + ts[14:16].astype(int) // 5).to_numpy() - (slot - H)
        V = np.full((H, st.L), np.nan); Q = np.full((H, st.L), np.nan); E = np.full((H, st.L), np.nan)
        lr = rank[h.link_id.map(li).to_numpy()]
        V[s, lr] = h.speed_kmh.to_numpy(); Q[s, lr] = h.flow_vph.to_numpy(); E[s, lr] = h.is_score_eligible.to_numpy(float)
        out[w.window_id] = (V, Q, E, pd.Timestamp(origin))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="f", help="our models t2<tag>_<key>.txt")
    ap.add_argument("--rows-tag", default="e")
    ap.add_argument("--final", action="store_true", help="use t2<tag>_<key>_final.txt for the scored windows")
    a = ap.parse_args()
    pub = Public()
    ours = {k: lgb.Booster(model_file=str(OUT / f"t2{a.tag}_{k}.txt")) for k, _ in CONDS}
    feb, names = feb_windows(a.rows_tag)
    keep = [i for i, n in enumerate(names) if not (n.startswith(MEMORY) and ours["onset"].num_feature() < len(names) + 1)]
    cache = OUT / f"t2pub_feb_{a.tag}.pkl"
    if cache.exists():
        PP = pickle.load(open(cache, "rb"))
    else:
        PP = {}
        for k, c in CONDS:
            PP[k] = []
            for r in feb[k]:
                v, q, e = history_from_cache(r["panel"], r["day"], r["slot"])
                origin = pd.Timestamp(f"{r['day']}T{r['slot'] // 12:02d}:{r['slot'] % 12 * 5:02d}:00Z")
                PP[k].append((predict_window(ours[k], r["X"][:, keep], c), pub.probs(r["panel"], v, q, e.astype(float), origin, c)))
        pickle.dump(PP, open(cache, "wb"))
    best = {}
    for k, c in CONDS:
        res = {}
        for w in (0.0, 0.3, 0.5, 0.7, 1.0):
            for thr in (0.3, 0.35, 0.4, 0.45, 0.5, 0.55):
                for fb in ((1, 3) if k == "onset" else (1,)):
                    s = [iou(decode(w * po + (1 - w) * pp, c, thr, fb), r["y"], r["elig"]) for (po, pp), r in zip(PP[k], feb[k])]
                    res[(w, thr, fb)] = (float(np.mean(s)), s)
        top = sorted(res, key=lambda x: res[x][0], reverse=True)
        print(k, "best (w_ours, thr, fb):", [(t, round(res[t][0], 4)) for t in top[:6]], flush=True)
        for w in (0.0, 0.5, 1.0):
            t = max((x for x in res if x[0] == w), key=lambda x: res[x][0])
            print(f"  w_ours={w}: {t} {res[t][0]:.4f}", flush=True)
        best[k] = top[0]
    # scored windows: blended queue files for a few settings
    finals = {k: lgb.Booster(model_file=str(OUT / f"t2{a.tag}_{k}{'_final' if a.final else ''}.txt")) for k, _ in CONDS}
    settings = {"pub": {"onset": (0.0, 0.55, 1), "ongoing": (0.0, 0.40, 1)},
                "mix": {"onset": (0.5, 0.45, 1), "ongoing": (0.5, 0.45, 1)},
                "best": best}
    queues = {s: {} for s in settings}
    for p in QUEUE_PANELS:
        for split in ("validation", "private"):
            blocks, nm = scored_windows(p, split)
            hs = history_scored(p, split)
            kp = [i for i, n in enumerate(nm) if not (n.startswith(MEMORY) and finals["onset"].num_feature() < len(nm) + 1)]
            for b in blocks:
                k = "onset" if b["condition"] == "queue_onset" else "ongoing"
                po = predict_window(finals[k], b["X"][:, kp], b["condition"])
                V, Q, E, origin = hs[b["window_id"]]
                pp = pub.probs(p, V, Q, E, origin, b["condition"])
                for sname, sset in settings.items():
                    w, thr, fb = sset[k]
                    pred = decode(w * po + (1 - w) * pp, b["condition"], thr, fb)
                    out = np.zeros_like(pred, dtype=np.int8); out[:, b["order"]] = pred
                    queues[sname][b["window_id"]] = out
    for sname, q in queues.items():
        pickle.dump(q, open(OUT / f"t2pub_{a.tag}_{sname}_queue.pkl", "wb"))
        on = [v[5].sum() for wid, v in q.items() if "validation" in wid and v[:5].sum() == 0 and v[5].sum() > 0]
        print(sname, settings[sname], "March onset links/window", round(float(np.mean(on)), 2), flush=True)


if __name__ == "__main__":
    main()
