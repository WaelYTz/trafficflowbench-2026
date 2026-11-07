"""Convert the parquet release into dense (day, slot, link) numpy arrays per panel and split.

Arrays written to cache/<panel>/<split>/:
  v, q, occ        float32  observed values (train: unmasked layer; scored splits: masked layer)
  vm, qm           float32  train only: the masked layer (Task 1 targets blanked)
  pct              uint8    pct_observed
  elig             bool     is_score_eligible
  target           bool     Task 1 target cells from the published template
  regime           int8     per day: 0,1,2 for R1,R2,R3
  rq, rpct, relig  ramp flow / pct_observed / eligible, (day, slot, ramp)
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import CACHE, PANELS, REL, SPLITS, T, link_index, network


def slot_of(ts: pd.Series) -> np.ndarray:
    s = ts.str
    return (s[11:13].astype(int) * 12 + s[14:16].astype(int) // 5).to_numpy()


def build(panel: str) -> str:
    net = network(panel)
    li = link_index(panel)
    L = len(net)
    pdir = REL / "corridors" / panel
    ramps = pd.read_csv(pdir / "network" / "ramp_attachment_map.csv", dtype=str)
    ri = {r: i for i, r in enumerate(ramps.ramp_link_id)}
    for split in SPLITS:
        out = CACHE / panel / split
        out.mkdir(parents=True, exist_ok=True)
        masked = sorted((pdir / split / "mainline_states_masked").glob("*/*.parquet"))
        day_regime = {}
        for f in masked:
            day_regime[f.name] = int(f.parent.name[-1]) - 1
        names = sorted(day_regime)
        days = [n.replace("synthetic_mainline_", "").replace(".parquet", "").replace("_", "-") for n in names]
        D = len(days)
        di = {d: i for i, d in enumerate(days)}
        arr = {k: np.full((D, T, L), np.nan, np.float32) for k in ("v", "q", "occ")}
        pct = np.zeros((D, T, L), np.uint8)
        elig = np.zeros((D, T, L), bool)
        if split == "train":
            vm = np.full((D, T, L), np.nan, np.float32)
            qm = np.full((D, T, L), np.nan, np.float32)
            src = {p.name: p for p in (pdir / split / "mainline_states").glob("*/*.parquet")}
        mpath = {f.name: f for f in masked}
        cols = ["date", "timestamp", "link_id", "speed_kmh", "flow_vph", "occupancy", "pct_observed", "is_score_eligible"]
        for name, day in zip(names, days):
            f = pd.read_parquet(src[name] if split == "train" else mpath[name], columns=cols)
            d = di[day]
            s = slot_of(f.timestamp)
            l = f.link_id.map(li).to_numpy()
            arr["v"][d, s, l] = f.speed_kmh.to_numpy(np.float32)
            arr["q"][d, s, l] = f.flow_vph.to_numpy(np.float32)
            arr["occ"][d, s, l] = f.occupancy.to_numpy(np.float32)
            pct[d, s, l] = f.pct_observed.to_numpy()
            elig[d, s, l] = f.is_score_eligible.to_numpy().astype(bool)
            if split == "train":
                m = pd.read_parquet(mpath[name], columns=["timestamp", "link_id", "speed_kmh", "flow_vph"])
                s2 = slot_of(m.timestamp)
                l2 = m.link_id.map(li).to_numpy()
                vm[d, s2, l2] = m.speed_kmh.to_numpy(np.float32)
                qm[d, s2, l2] = m.flow_vph.to_numpy(np.float32)
        for k, a in arr.items():
            np.save(out / f"{k}.npy", a)
        np.save(out / "pct.npy", pct)
        np.save(out / "elig.npy", elig)
        if split == "train":
            np.save(out / "vm.npy", vm)
            np.save(out / "qm.npy", qm)
        np.save(out / "regime.npy", np.array([day_regime[n] for n in names], np.int8))
        (out / "days.json").write_text(json.dumps(days))
        # Task 1 targets from the published template.
        tpl = pd.read_csv(REL / "task1" / panel / split / "sample_submission_state.csv",
                          usecols=["timestamp", "link_id"], dtype=str)
        target = np.zeros((D, T, L), bool)
        target[tpl.timestamp.str[:10].map(di).to_numpy(), slot_of(tpl.timestamp), tpl.link_id.map(li).to_numpy()] = True
        np.save(out / "target.npy", target)
        # Ramps.
        R = len(ramps)
        rq = np.full((D, T, R), np.nan, np.float32)
        rpct = np.zeros((D, T, R), np.uint8)
        relig = np.zeros((D, T, R), bool)
        for f in sorted((pdir / split / "ramp_states").glob("*/*.parquet")):
            r = pd.read_parquet(f, columns=["date", "timestamp", "ramp_link_id", "flow_vph", "pct_observed", "is_score_eligible"])
            r = r[r.date.isin(di)]
            dd = r.date.map(di).to_numpy()
            s = slot_of(r.timestamp)
            k = r.ramp_link_id.map(ri).to_numpy()
            rq[dd, s, k] = r.flow_vph.to_numpy(np.float32)
            rpct[dd, s, k] = r.pct_observed.to_numpy()
            relig[dd, s, k] = r.is_score_eligible.to_numpy().astype(bool)
        np.save(out / "rq.npy", rq)
        np.save(out / "rpct.npy", rpct)
        np.save(out / "relig.npy", relig)
        print(f"{panel} {split}: {D} days, {L} links, {R} ramps, {int(target.sum()):,} targets", flush=True)
    return panel


if __name__ == "__main__":
    panels = sys.argv[1:] or PANELS
    with ProcessPoolExecutor(max_workers=4) as ex:
        for p in ex.map(build, panels):
            print("done", p, flush=True)
