"""Local Task 3 score on a train month, using the official scorer's own state reconstruction.

The organizer boundary fluxes are projected so that the released truth conserves exactly, so on a
fully labelled split the conservation right-hand side equals the true accumulation change, and
  E_LWR = sum |dN_sub - dN_true| / sum |dN_true|
over the scorer's transitions (same mode, observability and ramp-validity filters). S_FD is the
official formula unchanged.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.csv as pacsv

from common import REL, ROOT, days, network

EXT = Path(os.environ.get("TFB_EXT", ROOT / "ext" / "trafficflowbench-public"))
sys.path.insert(0, str(EXT / "src"))
import task3.score_task3 as S3  # noqa: E402

MODES = json.loads((EXT / "config" / "task3_lwr_modes.json").read_text())
REGIMES = ("R1", "R2", "R3")
_files = S3.files
_month = {"m": "2031-02"}
S3.files = lambda panel_dir, split: [p for p in _files(panel_dir, split)
                                     if _month["m"] in p.name.replace("_", "-")]


@lru_cache(maxsize=None)
def template(panel: str, month: str) -> pd.DataFrame:
    t = pacsv.read_csv(REL / "task1" / panel / "train" / "sample_submission_state.csv",
                       convert_options=pacsv.ConvertOptions(
                           include_columns=["panel", "timestamp", "station_id", "link_id", "mask_regime"],
                           column_types={c: "string" for c in ("panel", "timestamp", "station_id", "link_id", "mask_regime")}))
    t = t.filter(pc.starts_with(t["timestamp"], month)).to_pandas()
    return t


def _published_targets(panel_dir, panel, split, regime):
    t = template(panel, _month["m"])
    t = t[t.mask_regime == regime]
    return set(zip(pd.to_datetime(t.timestamp, utc=True), t.station_id, t.link_id))


S3.published_targets = _published_targets


@lru_cache(maxsize=None)
def panel_static(panel: str):
    pdir = REL / "corridors" / panel
    params = S3.network_parameters(pdir).set_index("link_id")
    order = S3.link_order(pdir, "train")
    ramps = S3.load_ramp_flows(pdir, "train")
    ramps = ramps[ramps.timestamp.dt.strftime("%Y-%m") == _month["m"]]
    return params, order, ramps


def state_frame(panel: str, V: np.ndarray, Q: np.ndarray, day_idx: np.ndarray) -> pd.DataFrame:
    """Template rows of the month with values read from (day, slot, link) arrays in scorer order.
    day_idx maps the month's days to the first axis of V/Q."""
    t = template(panel, _month["m"]).copy()
    di = {d: i for i, d in enumerate(days(panel, "train"))}
    li = {l: i for i, l in enumerate(network(panel).link_id)}
    d = t.timestamp.str[:10].map(di).to_numpy()
    pos = np.searchsorted(day_idx, d)
    s = t.timestamp.str[11:13].astype(int).to_numpy() * 12 + t.timestamp.str[14:16].astype(int).to_numpy() // 5
    l = t.link_id.map(li).to_numpy()
    t["speed_kmh"] = V[pos, s, l]
    t["flow_vph"] = Q[pos, s, l]
    return t


def _lwr_rows(x: pd.DataFrame, panel: str) -> pd.DataFrame:
    """score_panel's filters and forward differences, returning the transition rows."""
    p = S3.network_parameters(REL / "corridors" / panel).set_index("link_id")
    x = x.join(p, on="link_id", rsuffix="_net")
    num = ["speed_kmh", "flow_vph", "density_vpkm", "inflow_vph", "outflow_vph", "on_ramp_flow_vph", "off_ramp_flow_vph",
           "on_ramp_valid", "off_ramp_valid", "accumulation_N"]
    for c in num:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=num + ["length_km", "free_speed_kmh", "capacity_vph", "critical_density", "k_jam"])
    mode = str(MODES["panels"][panel]["mode"])
    topo = pd.read_csv(REL / "corridors" / panel / "network" / "lwr_mainline_topology.csv", dtype=str)
    ramp_links = set(topo.loc[topo.on_ramp_link_ids.fillna("").str.len() > 0, "link_id"]) | \
        set(topo.loc[topo.off_ramp_link_ids.fillna("").str.len() > 0, "link_id"])
    if mode == "C" and ramp_links:
        x = x[~x.link_id.isin(ramp_links)].copy()
    topo_ids = set(topo.link_id.astype(str)); sub_ids = set(x.link_id.astype(str))
    obs = {}
    for row in topo.itertuples(index=False):
        inc = {v.strip() for v in str(row.incoming_link_ids).split(";") if v.strip()}
        out = {v.strip() for v in str(row.outgoing_link_ids).split(";") if v.strip()}
        obs[str(row.link_id)] = (inc & topo_ids).issubset(sub_ids) and (out & topo_ids).issubset(sub_ids)
    x = x[x.link_id.astype(str).map(obs).fillna(False).astype(bool)].copy()
    x = x.sort_values(["link_id", "timestamp"])
    x["dN"] = x.groupby("link_id").accumulation_N.shift(-1) - x.accumulation_N
    x["dt"] = (x.groupby("link_id").timestamp.shift(-1) - x.timestamp).dt.total_seconds() / 3600.0
    x = x[(x.dt > 0) & (x.dt <= 5.1 / 60.0)].copy()
    if mode in ("A", "B"):
        x = x[(x.on_ramp_valid >= 0.5) & (x.off_ramp_valid >= 0.5)].copy()
    return x


def evaluate(panel: str, sub: pd.DataFrame, true: pd.DataFrame, regimes=REGIMES) -> list[dict]:
    params, order, ramps = panel_static(panel)
    out = []
    for r in regimes:
        ds, _ = S3.derive_physics(panel, REL, "train", sub, r, order, params, ramps, None)
        dt_, _ = S3.derive_physics(panel, REL, "train", true, r, order, params, ramps, None)
        if ds.empty:
            continue
        s_fd = S3.score_panel(panel, ds, REL, MODES, r)["S_FD"]
        xs, xt = _lwr_rows(ds, panel), _lwr_rows(dt_, panel)
        m = xs[["timestamp", "link_id", "dN", "accumulation_N"]].merge(
            xt[["timestamp", "link_id", "dN", "accumulation_N"]], on=["timestamp", "link_id"], suffixes=("", "_t"))
        e = float(np.abs(m.dN - m.dN_t).sum() / (np.abs(m.dN_t).sum() + 1e-9))
        s_lwr = max(0.0, 1.0 - min(1.0, e))
        out.append(dict(panel=panel, regime=r, S_FD=s_fd, S_LWR=s_lwr, E_LWR=e, n=len(m), S_physics=(s_fd + 2 * s_lwr) / 3))
    return out
