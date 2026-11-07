"""Queue-only probe files: only Task 2 rows filled, so the public score is 0.3 * S_queue.

With --keep onset the ongoing windows are left empty (IoU 0 there), so the score becomes
0.3 * 0.5 * (onset IoU, family-averaged) and reveals the onset half on its own.
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd

from common import OUT, QUEUE_PANELS, REL
from subkey import write_submission


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--keep", choices=("all", "onset", "ongoing"), default="all")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    q = pickle.load(open(a.queue, "rb"))
    cond = {}
    for p in QUEUE_PANELS:
        for s in ("validation", "private"):
            wi = pd.read_csv(REL / "task2" / p / s / "window_index.csv")
            cond.update(dict(zip(wi.window_id, wi.condition)))
    if a.keep != "all":
        q = {w: (v if cond[w] == f"queue_{a.keep}" else np.zeros_like(v)) for w, v in q.items()}
    write_submission(OUT / f"{a.name}.zip", queue=q)


if __name__ == "__main__":
    main()
