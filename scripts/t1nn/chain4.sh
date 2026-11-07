#!/usr/bin/env bash
# Nov 4: transductive fine-tuning (t1_nn_ttt.py) of the d64 family per scored split -> pred_d64t,
# pred10 = normal (pred4 + 2 d64t)/3, gap (2 d64t + g48b)/3 -> kaggle_pkg/src_ds/t1nn_post4.npz. Log: scripts/t1nn/chain4.log
cd "$(dirname "$0")/../../src" || exit 1
export T1NN_BASE=1 T1NN_CHAR=1 T1NN_V2=1 PYTHONUNBUFFERED=1
O=../out_t1nn
st() { echo "$(date '+%H:%M:%S') $*" >> $O/chain4.log; }
until grep -q "CHAIN DONE" $O/chain3.log 2>/dev/null; do sleep 30; done
st "chain4 start"
python -W ignore t1_nn_ttt.py run --tags d64h d64f d64f2 d64f3 --width 64 --epochs 6 --lr 3e-4 --frac 0.5 --out-sub pred_d64t > $O/ttt_run.log 2>&1; st "pred_d64t done ($?)"
python -W ignore - > $O/pred10.log 2>&1 <<'PY'
import numpy as np
from common import PANELS, ROOT, load
O = ROOT / "out_t1nn"; od = O / "pred10"; od.mkdir(exist_ok=True)
for p in PANELS:
    for s in ("validation", "private"):
        a, d, g = (np.load(O / x / f"t1_{p}_{s}.npz") for x in ("pred4", "pred_d64t", "pred_g48b"))
        allgap = ~np.isfinite(load(p, s, "v")).any(axis=2)
        out = {k: np.where(allgap[:, :, None], (2 * d[k] + g[k]) / 3, (a[k] + 2 * d[k]) / 3).astype(np.float32) for k in ("V", "Q")}
        np.savez_compressed(od / f"t1_{p}_{s}.npz", **out); print(p, s, "ok", flush=True)
PY
st "pred10 done ($?)"
python -W ignore -c "import t1_nn_post2 as m; from common import ROOT; m.main(ROOT/'out_t1nn'/'pred10', ROOT/'kaggle_pkg'/'src_ds'/'t1nn_post4.npz')" > $O/post4.log 2>&1
st "t1nn_post4.npz done ($?)"
st "CHAIN DONE"
