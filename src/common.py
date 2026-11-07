"""Shared paths, panel metadata and cache access for the TrafficFlowBench pipeline."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
# Overridable so the same code runs locally and inside a Kaggle notebook.
REL = Path(os.environ.get("TFB_REL", ROOT / "kaggle_public"))
CACHE = Path(os.environ.get("TFB_CACHE", ROOT / "cache"))
OUT = Path(os.environ.get("TFB_OUT", ROOT / "out"))
SPLITS = ("train", "validation", "private")
REGIMES = ("R1", "R2", "R3")
T = 288  # five-minute slots per day

PANELS = [p["corridor_id"] for p in json.loads((REL / "config" / "corridors.json").read_text())["panels"]]
FAMILY = {p["corridor_id"]: p["family_id"] for p in json.loads((REL / "config" / "corridors.json").read_text())["panels"]}
QUEUE_PANELS = [p for p in PANELS if p not in ("D12_I405_N", "D12_I405_S")]


@lru_cache(maxsize=None)
def network(panel: str) -> pd.DataFrame:
    """One row per mainline link in scorer order (milepost, link_id), with FD and topology columns."""
    nd = REL / "corridors" / panel / "network"
    fd = pd.read_csv(nd / "fd_parameters.csv", dtype={"link_id": str})
    topo = pd.read_csv(nd / "lwr_mainline_topology.csv", dtype=str)
    first = sorted((REL / "corridors" / panel / "train" / "mainline_states").glob("*/*.parquet"))[0]
    mp = pd.read_parquet(first, columns=["link_id", "station_id", "milepost"]).drop_duplicates("link_id")
    net = fd.merge(mp, on="link_id", how="left")
    net = net.merge(topo[["link_id", "incoming_link_ids", "outgoing_link_ids", "on_ramp_link_ids",
                          "off_ramp_link_ids", "has_sensor"]], on="link_id", how="left")
    net = net.sort_values(["milepost", "link_id"]).reset_index(drop=True)
    net["v_cut"] = 0.6 * net.free_speed_kmh
    idx = {l: i for i, l in enumerate(net.link_id)}
    # Travel direction along the milepost axis, from the topology's outgoing links.
    votes = []
    for i, row in net.iterrows():
        for o in str(row.outgoing_link_ids).split(";"):
            if o in idx:
                votes.append(np.sign(idx[o] - i))
    net.attrs["direction"] = 1 if np.sum(votes) >= 0 else -1
    return net


def link_index(panel: str) -> dict[str, int]:
    return {l: i for i, l in enumerate(network(panel).link_id)}


def cache_path(panel: str, split: str, name: str) -> Path:
    return CACHE / panel / split / f"{name}.npy"


def load(panel: str, split: str, name: str, mmap: bool = False) -> np.ndarray:
    return np.load(cache_path(panel, split, name), mmap_mode="r" if mmap else None)


def days(panel: str, split: str) -> list[str]:
    f = CACHE / panel / split / "days.json"
    if f.exists():
        return json.loads(f.read_text())
    # Same order build_cache uses: sorted masked-file names.
    names = sorted(p.name for p in (REL / "corridors" / panel / split / "mainline_states_masked").glob("*/*.parquet"))
    return [n.replace("synthetic_mainline_", "").replace(".parquet", "").replace("_", "-") for n in names]


def slot_ts(day: str, slot: int) -> str:
    return f"{day}T{slot // 12:02d}:{(slot % 12) * 5:02d}:00Z"
