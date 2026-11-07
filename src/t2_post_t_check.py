"""Post-T deletion test (the organizers' rerun rule): for each scored Task 2 window, blank every released
masked-layer cell stamped after its origin T (the rest of that day, all later days, and every later split on
the train -> March -> April calendar), rebuild the window's features from scratch and assert they are
bit-identical to the features computed from the full release. Features are the only Task 2 input besides
window_history (T-60 .. T-5, untouched), so identical features mean identical queue predictions.

  python t2_post_t_check.py [N_PER_PANEL_SPLIT]     (default: every window)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import common
import t2_data
from common import QUEUE_PANELS, days
from t2_data import CAL_SPLITS, scored_windows

t2_data.RECENT_DAYS = 7
_load = common.load


def truncated_load(cut_split: str, cut_day: int, cut_slot: int):
    later = CAL_SPLITS[CAL_SPLITS.index(cut_split) + 1:]

    def load(panel, split, key, mmap=False):
        a = _load(panel, split, key, mmap=mmap)
        if key not in ("v", "q", "vm", "qm", "rq") or (split != cut_split and split not in later):
            return a
        a = np.array(a, dtype=float, copy=True)
        if split in later:
            a[:] = np.nan
        else:
            a[cut_day, cut_slot + 1:] = np.nan
            a[cut_day + 1:] = np.nan
        return a
    return load


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    bad = checked = 0
    for p in QUEUE_PANELS:
        for split in ("validation", "private"):
            ref, names = scored_windows(p, split)
            wi = pd.read_csv(common.REL / "task2" / p / split / "window_index.csv")
            dl = {d: i for i, d in enumerate(days(p, split))}
            for k, w in enumerate(wi.itertuples(index=False)):
                if n is not None and k >= n:
                    break
                o = str(w.forecast_origin)
                d, slot = dl[o[:10]], int(o[11:13]) * 12 + int(o[14:16]) // 5
                t2_data.load = truncated_load(split, d, slot)
                try:
                    blocks, _ = scored_windows(p, split)
                finally:
                    t2_data.load = _load
                new = next(b for b in blocks if b["window_id"] == w.window_id)["X"]
                old = next(b for b in ref if b["window_id"] == w.window_id)["X"]
                same = np.array_equal(np.nan_to_num(old, nan=-9e9), np.nan_to_num(new, nan=-9e9))
                checked += 1
                bad += not same
                if not same:
                    diff = np.flatnonzero((np.nan_to_num(old, nan=-9e9) != np.nan_to_num(new, nan=-9e9)).any(0))
                    print(f"DIFF {w.window_id}: columns {[names[i] for i in diff]}", flush=True)
            print(f"{p} {split}: checked {checked} windows so far, {bad} differ", flush=True)
    print(f"DONE: {checked} windows checked, {bad} differ", flush=True)


if __name__ == "__main__":
    main()
