#!/usr/bin/env bash
# Nov 4: more Jun-Feb seeds for the v2 neural imputer (variance reduction), same family mix as pred8
# (normal: old 1 + d64 2; gap: d64 2 + g48 1) -> out_t1nn/pred9 -> kaggle_pkg/src_ds/t1nn_post3.npz. Log: scripts/t1nn/chain3.log.
cd "$(dirname "$0")/../../src" || exit 1
export T1NN_BASE=1 T1NN_CHAR=1 T1NN_V2=1 PYTHONUNBUFFERED=1
O=../out_t1nn
st() { echo "$(date '+%H:%M:%S') $*" >> $O/chain3.log; }
st "chain3 start"
(
  python -W ignore t1_nn.py train --tag d64f2 --all-days --width 64 --epochs 10 --lr 1.5e-3 --seed 13 > $O/train_d64f2.log 2>&1; st "d64f2 done ($?)"
  python -W ignore t1_nn.py train --tag d64f3 --all-days --width 64 --epochs 10 --lr 1.5e-3 --seed 14 > $O/train_d64f3.log 2>&1; st "d64f3 done ($?)"
  python -W ignore t1_nn.py predict --tags d64h d64f d64f2 d64f3 --width 64 --out-sub pred_d64b > $O/predict_d64b.log 2>&1; st "pred_d64b done ($?)"
) &
(
  sleep 60
  python -W ignore t1_nn.py train --tag g48f2 --all-days --width 48 --epochs 8 --lr 1.5e-3 --seed 23 --gap-p 1.0 --gap-w 2 --gap-real > $O/train_g48f2.log 2>&1; st "g48f2 done ($?)"
  python -W ignore t1_nn.py predict --tags g48h g48f g48f2 --width 48 --out-sub pred_g48b > $O/predict_g48b.log 2>&1; st "pred_g48b done ($?)"
) &
wait
python -W ignore - > $O/pred9.log 2>&1 <<'PY'
import numpy as np
from common import PANELS, ROOT, load
O = ROOT / "out_t1nn"; od = O / "pred9"; od.mkdir(exist_ok=True)
for p in PANELS:
    for s in ("validation", "private"):
        a, d, g = (np.load(O / x / f"t1_{p}_{s}.npz") for x in ("pred4", "pred_d64b", "pred_g48b"))
        allgap = ~np.isfinite(load(p, s, "v")).any(axis=2)
        out = {k: np.where(allgap[:, :, None], (2 * d[k] + g[k]) / 3, (a[k] + 2 * d[k]) / 3).astype(np.float32) for k in ("V", "Q")}
        np.savez_compressed(od / f"t1_{p}_{s}.npz", **out); print(p, s, "ok", flush=True)
PY
st "pred9 done ($?)"
python -W ignore -c "import t1_nn_post2 as m; from common import ROOT; m.main(ROOT/'out_t1nn'/'pred9', ROOT/'kaggle_pkg'/'src_ds'/'t1nn_post3.npz')" > $O/post3.log 2>&1
st "t1nn_post3.npz done ($?)"
st "CHAIN DONE"
