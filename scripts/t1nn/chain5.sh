#!/usr/bin/env bash
# Nov 5: neural T1 with neighbouring-day channels (T1NN_MD=1), Jun-Jan fit, Feb holdout vs d64h (S 0.95000 at ep10).
cd "$(dirname "$0")/../../src" || exit 1
export T1NN_BASE=1 T1NN_CHAR=1 T1NN_V2=1 T1NN_MD=1 PYTHONUNBUFFERED=1
O=../out_t1nn
st() { echo "$(date '+%H:%M:%S') $*" >> $O/chain5.log; }
st "chain5 start"
python -W ignore t1_nn.py train --tag m64h --width 64 --epochs 10 --lr 1.5e-3 --seed 11 --eval-every 5 > $O/train_m64h.log 2>&1; st "m64h done ($?)"
st "CHAIN DONE"
