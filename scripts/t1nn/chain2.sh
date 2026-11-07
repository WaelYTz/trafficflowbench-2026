#!/usr/bin/env bash
# Night chain (Nov 3 ~21:05): stronger neural Task 1 imputer. Two GPU streams in parallel (~1.4 + 1.1 GB GPU, ~2 GB RAM each),
# then Feb-holdout selection -> out_t1nn/pred8 -> kaggle_pkg/src_ds/t1nn_post2.npz. Step log: scripts/t1nn/chain2.log.
cd "$(dirname "$0")/../../src" || exit 1
export T1NN_BASE=1 T1NN_CHAR=1 T1NN_V2=1 PYTHONUNBUFFERED=1
O=../out_t1nn
st() { echo "$(date '+%H:%M:%S') $*" >> $O/chain2.log; }
st "chain2 start"
# Stream A: width 64, 10 epochs, V2 channels (ramp demand ratio + congestion frequency): Jun-Jan (Feb eval) then Jun-Feb
(
  python -W ignore t1_nn.py train --tag d64h --width 64 --epochs 10 --eval-every 5 --lr 1.5e-3 --seed 11 > $O/train_d64h.log 2>&1
  st "d64h done ($?): $(grep 'NN ep10' $O/train_d64h.log)"
  python -W ignore t1_nn.py train --tag d64f --all-days --width 64 --epochs 10 --lr 1.5e-3 --seed 12 > $O/train_d64f.log 2>&1
  st "d64f done ($?)"
  python -W ignore t1_nn.py predict --tags d64h d64f --width 64 --out-sub pred_d64 > $O/predict_d64.log 2>&1
  st "pred_d64 done ($?)"
) &
# Stream B: gap specialist, width 48, 8 epochs, synthetic all-link gaps on every day (starts 55-250 like the real ones), gap loss x2
(
  sleep 60
  python -W ignore t1_nn.py train --tag g48h --width 48 --epochs 8 --eval-every 4 --lr 1.5e-3 --seed 21 --gap-p 1.0 --gap-w 2 --gap-real > $O/train_g48h.log 2>&1
  st "g48h done ($?): $(grep 'NN ep8' $O/train_g48h.log)"
  python -W ignore t1_nn.py train --tag g48f --all-days --width 48 --epochs 8 --lr 1.5e-3 --seed 22 --gap-p 1.0 --gap-w 2 --gap-real > $O/train_g48f.log 2>&1
  st "g48f done ($?)"
  python -W ignore t1_nn.py predict --tags g48h g48f --width 48 --out-sub pred_g48 > $O/predict_g48.log 2>&1
  st "pred_g48 done ($?)"
) &
wait
st "training streams finished"
python -W ignore t1_nn_select.py > $O/select.log 2>&1
st "select + pred8 done ($?)"
python -W ignore t1_nn_post2.py pred8 > $O/post2.log 2>&1
st "t1nn_post2.npz done ($?)"
st "CHAIN DONE"
