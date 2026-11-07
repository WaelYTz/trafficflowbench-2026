"""End-to-end pipeline: cache -> Task 1 models -> Task 2 models (t2_pipeline.py) -> Task 4 L2 -> submission zip.

Runs locally or in a Kaggle notebook (paths come from TFB_REL / TFB_CACHE / TFB_OUT). The defaults are the
settings of the final submissions (Task 1: 4000 sampled targets per day, up to 20000 rounds).
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time

import numpy as np

from common import CACHE, OUT, PANELS
from subkey import write_submission


def sh(args):
    print(">>", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--cap-train", type=int, default=4000)
    ap.add_argument("--rounds", type=int, default=20000)
    ap.add_argument("--pub-sub", default=None, help="optional public submission.csv whose state rows are blended in 30%%")
    ap.add_argument("--train-prof", action="store_true",
                    help="Task 1: use train-split profiles for the scored months (default: the month's own profiles)")
    a = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    if not (CACHE / PANELS[-1] / "private" / "relig.npy").exists():
        sh(["build_cache.py"])
    if not (CACHE / "subkey.npz").exists():
        sh(["subkey.py"])
    print(f"cache ready {time.time() - t0:.0f}s", flush=True)

    sh(["t1_train.py", "--device", a.device, "--workers", str(a.workers), "--cap-train", str(a.cap_train),
        "--rounds", str(a.rounds)] + ([] if a.train_prof else ["--month-prof"]))

    sh(["t2_pipeline.py"])
    queue = pickle.load(open(OUT / "t2_queue_final.pkl", "rb"))
    print(f"Task 2 done {time.time() - t0:.0f}s", flush=True)

    from task4 import problem, solve_l2
    odme = {(p, s): solve_l2(problem(p, s)) for p in PANELS for s in ("validation", "private")}
    pickle.dump(odme, open(OUT / "t4_l2.pkl", "wb"))
    print(f"Task 4 done {time.time() - t0:.0f}s", flush=True)

    state = {}
    for p in PANELS:
        for s in ("validation", "private"):
            z = np.load(OUT / f"t1_{p}_{s}.npz")
            state[(p, s)] = (z["V"], z["Q"])
    blend = None
    if a.pub_sub:
        import pyarrow.csv as pacsv
        import subkey
        pub = pacsv.read_csv(a.pub_sub, convert_options=pacsv.ConvertOptions(
            include_columns=["submission_id", "speed_kmh", "flow_vph"])).to_pandas()
        assert (pub.submission_id.to_numpy() == subkey.key()["submission_id"]).all(), "row order differs"
        blend = (pub.speed_kmh.to_numpy(float), pub.flow_vph.to_numpy(float), 0.7)
    write_submission(OUT / "submission.zip", state=state, queue=queue, odme=odme, state_blend=blend)
    print(json.dumps({"total_seconds": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    main()
