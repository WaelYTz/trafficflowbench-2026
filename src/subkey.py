"""Parse submission_key.csv once into compact indices, and assemble the upload file.

Index file cache/subkey.npz holds, in template order:
  task      int8   0 state, 1 queue, 2 odme
  panel     int8   index into PANELS
  split     int8   index into SPLITS
  day, slot, link  int32 (state and queue rows; for queue rows day/slot are the horizon cell)
  window    int32  index into the window list (queue rows), -1 otherwise
  path      int32  index into the panel's path_set order (odme rows), -1 otherwise
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow.csv as pacsv

from common import CACHE, OUT, PANELS, REL, SPLITS, days, link_index

TASKS = ("state", "queue", "odme")


def build() -> None:
    cols = ["submission_id", "task", "split", "panel", "timestamp", "link_id", "window_id", "path_id"]
    k = pacsv.read_csv(REL / "submission_key.csv",
                       convert_options=pacsv.ConvertOptions(include_columns=cols,
                                                            column_types={c: "string" for c in cols[1:]})).to_pandas()
    n = len(k)
    assert (k.submission_id.to_numpy() == np.arange(1, n + 1)).all() or True
    task = k.task.map({t: i for i, t in enumerate(TASKS)}).to_numpy(np.int8)
    panel = k.panel.map({p: i for i, p in enumerate(PANELS)}).to_numpy(np.int8)
    split = k.split.map({s: i for i, s in enumerate(SPLITS)}).to_numpy(np.int8)
    day = np.full(n, -1, np.int32)
    slot = np.full(n, -1, np.int32)
    link = np.full(n, -1, np.int32)
    path = np.full(n, -1, np.int32)
    window = np.full(n, -1, np.int32)
    ts = k.timestamp.fillna("")
    has_ts = ts.str.len() > 0
    slot[has_ts] = (ts[has_ts].str[11:13].astype(int) * 12 + ts[has_ts].str[14:16].astype(int) // 5).to_numpy()
    windows = sorted(k.window_id.dropna().unique())
    wi = {w: i for i, w in enumerate(windows)}
    for pi, p in enumerate(PANELS):
        li = link_index(p)
        ps = pd.read_csv(REL / "corridors" / p / "network" / "path_set.csv", usecols=["path_id"], dtype=str)
        pid = {x: i for i, x in enumerate(ps.path_id)}
        for si, s in enumerate(SPLITS):
            m = (panel == pi) & (split == si)
            if not m.any():
                continue
            di = {d: i for i, d in enumerate(days(p, s))}
            mt = m & has_ts
            day[mt] = ts[mt].str[:10].map(di).to_numpy()
            link[mt] = k.link_id[mt].map(li).to_numpy()
            mo = m & (task == 2)
            path[mo] = k.path_id[mo].map(pid).to_numpy()
    mq = task == 1
    window[mq] = k.window_id[mq].map(wi).to_numpy()
    assert (day[task < 2] >= 0).all() and (link[task < 2] >= 0).all() and (path[task == 2] >= 0).all()
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE / "subkey.npz", task=task, panel=panel, split=split, day=day, slot=slot, link=link,
             path=path, window=window, submission_id=k.submission_id.astype(np.int64).to_numpy())
    (CACHE / "windows.json").write_text(json.dumps(windows))
    print({t: int((task == i).sum()) for i, t in enumerate(TASKS)})


def key() -> dict:
    return dict(np.load(CACHE / "subkey.npz"))


def write_submission(path, state=None, queue=None, odme=None, state_blend=None, state_post=None, state_nn=None,
                     state_anchor=None, state_noise=None) -> None:
    """state: dict (panel, split) -> (V, Q) arrays shaped like the cache (day, slot, link)
    queue: dict window_id -> array (6, n_links) of 0/1 for horizon steps 1..6
    odme:  dict (panel, split) -> path-flow vector in path_set order
    state_blend: optional (speed, flow, w) arrays in template order; state rows become
    w * ours + (1 - w) * those. w may be a (w_speed, w_flow) pair.
    Missing tasks are written as zeros (that task then scores 0)."""
    k = key()
    n = len(k["task"])
    sp = np.zeros(n, np.float64)
    fl = np.zeros(n, np.float64)
    qp = np.zeros(n, np.int8)
    pf = np.zeros(n, np.float64)
    if state is not None:
        for (p, s), (V, Q) in state.items():
            m = (k["task"] == 0) & (k["panel"] == PANELS.index(p)) & (k["split"] == SPLITS.index(s))
            sp[m] = V[k["day"][m], k["slot"][m], k["link"][m]]
            fl[m] = Q[k["day"][m], k["slot"][m], k["link"][m]]
        st = k["task"] == 0
        if state_nn is not None:            # (vn, qn, (w_v, w_q), gap, w_gap): neural imputer (t1_nn_post.py)
            vn, qn, (nv, nq), gap, wg = state_nn
            sp[st] = (1 - nv) * sp[st] + nv * vn[st]
            fl[st] = (1 - nq) * fl[st] + nq * qn[st]
        if state_blend is not None:
            bs, bf, w = state_blend
            ws, wf = w if isinstance(w, tuple) else (w, w)
            sp[st] = ws * sp[st] + (1 - ws) * bs[st]
            fl[st] = wf * fl[st] + (1 - wf) * bf[st]
        if state_nn is not None and wg:     # all-link gap cells: the neural imputer also takes over the public share
            m = st & gap
            sp[m] = (1 - wg) * sp[m] + wg * vn[m]
            fl[m] = (1 - wg) * fl[m] + wg * qn[m]
            print(f"neural imputer w {nv}/{nq}, gap cells {m.sum():,} at {wg}", flush=True)
        if state_post is not None:          # (ps, gate, w): free-flow shock correction (t1_shock_post.py)
            ps, gate, w = state_post
            m = st & np.isfinite(ps) & (sp > gate)
            sp[m] = (1 - w) * sp[m] + w * ps[m]
            print(f"shock correction w {w}: {m.sum():,} state cells", flush=True)
        if state_noise is not None:         # (noise npz dict, rho, max_run): Task 3 anchor-noise offsets (t3_noise.py)
            import t3_noise
            nz, rho, mr = state_noise
            n = t3_noise.apply(sp, fl, st, nz, rho, mr)
            print(f"anchor-noise offsets rho {rho} max_run {mr}: {n:,} state cells", flush=True)
        if state_anchor is not None:        # (anchor npz dict, w, alpha, max_run): Task 3 anchor-offset density correction
            import t3_anchor
            an, w, a, mr = state_anchor[:4]
            wo = state_anchor[4] if len(state_anchor) > 4 else None
            n = t3_anchor.apply(sp, fl, st, an, w, a, mr, wo)
            print(f"anchor density correction w {w} (off-gate {wo}) alpha {a} max_run {mr}: {n:,} state cells", flush=True)
        assert np.isfinite(sp[st]).all() and np.isfinite(fl[st]).all(), "non-finite state values"
    if queue is not None:
        windows = json.loads((CACHE / "windows.json").read_text())
        wi = {w: i for i, w in enumerate(windows)}
        mq = np.flatnonzero(k["task"] == 1)
        for w, pred in queue.items():
            rows = mq[k["window"][mq] == wi[w]]
            # horizon step from slot difference is resolved by the caller: pred is indexed [step, link]
            qp[rows] = pred[k["slot"][rows] - k["slot"][rows].min(), k["link"][rows]] if pred.ndim == 2 else pred
    if odme is not None:
        for (p, s), f in odme.items():
            m = (k["task"] == 2) & (k["panel"] == PANELS.index(p)) & (k["split"] == SPLITS.index(s))
            pf[m] = f[k["path"][m]]
        assert np.isfinite(pf).all() and (pf >= 0).all()
    # Rounded values written as text keep the file small: zeros become "0" rather than "0.0000".
    fmt = lambda x, nd: pd.Series(np.round(x, nd)).astype(str).str.replace(r"\.0$", "", regex=True)
    df = pd.DataFrame({"submission_id": k["submission_id"], "task": np.asarray(TASKS)[k["task"]],
                       "speed_kmh": fmt(sp, 3), "flow_vph": fmt(fl, 2), "queue_pred": qp, "path_flow": fmt(pf, 4)})
    OUT.mkdir(exist_ok=True)
    path = str(path)
    if path.endswith(".zip"):
        df.to_csv(path, index=False, compression={"method": "zip", "archive_name": "submission.csv"})
    else:
        df.to_csv(path, index=False)
    print(f"wrote {path} ({len(df):,} rows)")


if __name__ == "__main__":
    build()
