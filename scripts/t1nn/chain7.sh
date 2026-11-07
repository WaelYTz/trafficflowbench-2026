#!/usr/bin/env bash
# Nov 5: if the wider net (w96h) beats d64h (0.95000) by >= 0.002 on the Feb holdout, train 2 Jun-Feb w96 nets,
# predict pred_w96b (w96h w96f w96f2), pred11 = normal (pred4 + 2 d64b + 2 w96b)/5, gap = pred9's (2 d64b + g48b)/3,
# -> kaggle_pkg/src_ds/t1nn_post5.npz (nf=5). Log: scripts/t1nn/chain7.log
cd "$(dirname "$0")/../../src" || exit 1
export T1NN_BASE=1 T1NN_CHAR=1 T1NN_V2=1 T1NN_MD=0 PYTHONUNBUFFERED=1
O=../out_t1nn
st() { echo "$(date '+%H:%M:%S') $*" >> $O/chain7.log; }
until grep -q "CHAIN DONE" $O/chain6.log 2>/dev/null; do sleep 60; done
S=$(grep "NN ep14" $O/train_w96h.log | grep -o "S_state [0-9.]*" | awk '{print $2}')
st "w96h Feb S_state $S (pass if >= 0.9505)"
python -c "import sys; sys.exit(0 if float('${S:-0}') >= 0.9505 else 1)" || { st "NOT PASSED - stop"; st "CHAIN DONE"; exit 0; }
python -W ignore t1_nn.py train --tag w96f --all-days --width 96 --epochs 14 --lr 1.2e-3 --seed 13 > $O/train_w96f.log 2>&1; st "w96f done ($?)"
python -W ignore t1_nn.py train --tag w96f2 --all-days --width 96 --epochs 14 --lr 1.2e-3 --seed 14 > $O/train_w96f2.log 2>&1; st "w96f2 done ($?)"
python -W ignore t1_nn.py predict --tags w96h w96f w96f2 --width 96 --out-sub pred_w96b > $O/predict_w96b.log 2>&1; st "pred_w96b done ($?)"
python -W ignore - > $O/pred11.log 2>&1 <<'PY'
import numpy as np
from common import PANELS, ROOT, load
O = ROOT / "out_t1nn"; od = O / "pred11"; od.mkdir(exist_ok=True)
for p in PANELS:
    for s in ("validation", "private"):
        a, d, w, g = (np.load(O / x / f"t1_{p}_{s}.npz") for x in ("pred4", "pred_d64b", "pred_w96b", "pred_g48b"))
        allgap = ~np.isfinite(load(p, s, "v")).any(axis=2)
        out = {k: np.where(allgap[:, :, None], (2 * d[k] + g[k]) / 3, (a[k] + 2 * d[k] + 2 * w[k]) / 5).astype(np.float32) for k in ("V", "Q")}
        np.savez_compressed(od / f"t1_{p}_{s}.npz", **out); print(p, s, "ok", flush=True)
PY
st "pred11 done ($?)"
python -W ignore -c "import t1_nn_post2 as m; from common import ROOT; m.main(ROOT/'out_t1nn'/'pred11', ROOT/'kaggle_pkg'/'src_ds'/'t1nn_post5.npz')" > $O/post5.log 2>&1
st "t1nn_post5.npz done ($?)"
st "CHAIN DONE"
