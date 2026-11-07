"""Assemble an upload file from Task 1 prediction files and pickled Task 2 / Task 4 answers.

Used inside a light Kaggle notebook that reads the Task 1 outputs of the training notebook, so the
large prediction files never have to be downloaded.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

import subkey
from common import CACHE, OUT, PANELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1-dir", required=True, help="folder holding t1_<panel>_<split>.npz")
    ap.add_argument("--queue", required=True, help="pickle: window_id -> (6, L) 0/1 in scorer order")
    ap.add_argument("--odme", required=True, help="pickle: (panel, split) or ('l2', panel, split) -> path flows")
    ap.add_argument("--name", default="submission")
    ap.add_argument("--pub-sub", default=None, help="another full submission.csv whose state rows are blended in")
    ap.add_argument("--w-ours", type=float, default=0.7)
    ap.add_argument("--extra-sub", nargs="*", default=[], help="more full submission.csv files whose state rows are blended in")
    ap.add_argument("--w-extra", nargs="*", type=float, default=[], help="their shares of the final state (taken from --pub-sub's)")
    ap.add_argument("--shock-w", type=float, default=0.0,
                    help="free-flow shock correction weight (t1_shock_ps.npz next to this script); 0 = off")
    ap.add_argument("--w-ours-flow", type=float, default=None,
                    help="our share for flow only (default --w-ours); the public sources keep their relative shares")
    ap.add_argument("--nn-w", type=float, nargs=2, default=None, metavar=("W_V", "W_Q"),
                    help="neural imputer share of our state (t1nn_post.npz next to this script)")
    ap.add_argument("--nn-gap", type=float, default=0.0, help="neural imputer share of the final state on all-link gap cells")
    ap.add_argument("--nn-file", default="t1nn_post.npz", help="neural imputer predictions next to this script (t1nn_post2.npz = v2)")
    ap.add_argument("--anchor", type=float, nargs="+", default=None, metavar="W ALPHA MAX_RUN [W_OFF]",
                    help="Task 3 anchor-offset density correction (t3_anchor.npz next to this script); W_OFF = weight "
                         "outside the free-flow gate")
    ap.add_argument("--noise", type=float, nargs=2, default=None, metavar=("RHO", "MAX_RUN"),
                    help="Task 3 anchor-noise offsets (t3_noise.npz next to this script)")
    a = ap.parse_args()
    if not (CACHE / "subkey.npz").exists():
        subkey.build()
    # "none" leaves a task out (it scores 0), so a file with one task filled is a probe of that task alone
    state = None
    if a.t1_dir != "none":
        t1 = Path(a.t1_dir)
        state = {}
        for p in PANELS:
            for s in ("validation", "private"):
                z = np.load(t1 / f"t1_{p}_{s}.npz")
                state[(p, s)] = (z["V"], z["Q"])
    queue = None if a.queue == "none" else pickle.load(open(a.queue, "rb"))
    odme = None
    if a.odme != "none":
        od = pickle.load(open(a.odme, "rb"))
        odme = {(p, s): od[("l2", p, s)] if ("l2", p, s) in od else od[(p, s)] for p in PANELS for s in ("validation", "private")}
    OUT.mkdir(parents=True, exist_ok=True)
    blend = None
    if a.pub_sub and state is not None:
        import pyarrow.csv as pacsv
        pub = pacsv.read_csv(a.pub_sub, convert_options=pacsv.ConvertOptions(
            include_columns=["submission_id", "speed_kmh", "flow_vph"])).to_pandas()
        k = subkey.key()
        assert len(pub) == len(k["task"]) and (pub.submission_id.to_numpy() == k["submission_id"]).all(), "row order differs"
        bs, bf = pub.speed_kmh.to_numpy(float), pub.flow_vph.to_numpy(float)
        # shares of the final state: ours w_ours, each extra w_extra[i], the --pub-sub file the rest
        rest = 1 - a.w_ours
        bs, bf = bs * (rest - sum(a.w_extra)) / rest, bf * (rest - sum(a.w_extra)) / rest
        for f, w in zip(a.extra_sub, a.w_extra):
            p2 = pacsv.read_csv(f, convert_options=pacsv.ConvertOptions(
                include_columns=["submission_id", "speed_kmh", "flow_vph"])).to_pandas()
            assert (p2.submission_id.to_numpy() == k["submission_id"]).all(), f"row order differs ({f})"
            bs = bs + w / rest * p2.speed_kmh.to_numpy(float)
            bf = bf + w / rest * p2.flow_vph.to_numpy(float)
        blend = (bs, bf, a.w_ours if a.w_ours_flow is None else (a.w_ours, a.w_ours_flow))
    post = None
    if a.shock_w and state is not None:
        z = np.load(Path(__file__).resolve().parent / "t1_shock_ps.npz")
        post = (z["ps"], z["gate"], a.shock_w)
    nn = None
    if (a.nn_w or a.nn_gap) and state is not None:
        z = np.load(Path(__file__).resolve().parent / a.nn_file)
        nn = (z["vn"].astype(np.float64), z["qn"].astype(np.float64), tuple(a.nn_w or (0.0, 0.0)), z["gap"], a.nn_gap)
    nzz = None
    if a.noise and state is not None:
        z = np.load(Path(__file__).resolve().parent / "t3_noise.npz")
        nzz = ({x: z[x] for x in z.files}, a.noise[0], a.noise[1])
    anc = None
    if a.anchor and state is not None:
        z = np.load(Path(__file__).resolve().parent / "t3_anchor.npz")
        anc = ({x: z[x] for x in z.files}, *a.anchor[:4])
    subkey.write_submission(OUT / f"{a.name}.zip", state=state, queue=queue, odme=odme, state_blend=blend, state_post=post,
                            state_nn=nn, state_anchor=anc, state_noise=nzz)


if __name__ == "__main__":
    main()
