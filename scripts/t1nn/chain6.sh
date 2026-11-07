#!/usr/bin/env bash
# Nov 5 night: wider + longer neural T1 (width 96, 14 epochs), Jun-Jan fit, Feb holdout vs d64h (S 0.95000).
cd "$(dirname "$0")/../../src" || exit 1
export T1NN_BASE=1 T1NN_CHAR=1 T1NN_V2=1 T1NN_MD=0 PYTHONUNBUFFERED=1
O=../out_t1nn
st() { echo "$(date '+%H:%M:%S') $*" >> $O/chain6.log; }
st "chain6 start"
python -W ignore t1_nn.py train --tag w96h --width 96 --epochs 14 --lr 1.2e-3 --seed 11 --eval-every 7 > $O/train_w96h.log 2>&1; st "w96h done ($?)"
st "CHAIN DONE"
