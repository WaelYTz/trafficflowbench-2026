#!/usr/bin/env bash
# Offline chain for the neural Task 1 imputer (launched ~15:45 Nov 3). One GPU job at a time (~1.1 GB GPU, ~2 GB RAM).
cd "$(dirname "$0")/../../src" || exit 1
export T1NN_BASE=1 T1NN_CHAR=1 PYTHONUNBUFFERED=1
O=../out_t1nn
st() { echo "$(date '+%H:%M:%S') $*" >> $O/chain.log; }
st "chain start"
# 0. Feb blend reports for the finished holdout net c48 (CPU)
python -W ignore t1_nn_blend.py c48 > $O/blend_c48.log 2>&1
python -W ignore t1_nn_blend.py --gap c48 > $O/blend_c48_gap.log 2>&1
st "blend c48 done"
# 1. wait for the all-days net c48f (already running since 15:37, log train_c48f.log)
# watchdog: if the log has not moved for 15 min (process killed), retrain c48f inside the chain
until grep -qE "^done|Traceback" $O/train_c48f.log; do
  if [ -n "$(find $O/train_c48f.log -mmin +15)" ]; then
    st "c48f log stale -> relaunching c48f"
    python -W ignore t1_nn.py train --tag c48f --all-days --width 48 --epochs 8 --lr 1.5e-3 --seed 2 > $O/train_c48f.log 2>&1
  fi
  sleep 30
done
st "c48f finished: $(tail -1 $O/train_c48f.log)"
# 2. validation/private predictions, 2-net ensemble (Jun-Jan c48 + Jun-Feb c48f) -> out_t1nn/pred2/
python -W ignore t1_nn.py predict --tags c48 c48f --width 48 --out-sub pred2 > $O/predict_pred2.log 2>&1
st "pred2 done ($?)"
# 3. second Jun-Jan seed: Feb ensemble gain + blend weight for a 2-seed NN
python -W ignore t1_nn.py train --tag c48s3 --width 48 --epochs 8 --eval-every 4 --lr 1.5e-3 --seed 3 > $O/train_c48s3.log 2>&1
st "c48s3 done ($?)"
python -W ignore t1_nn_blend.py c48 c48s3 > $O/blend_c48_c48s3.log 2>&1
python -W ignore t1_nn_blend.py --gap c48 c48s3 > $O/blend_c48_c48s3_gap.log 2>&1
st "blend 2-seed done"
# 4. second all-days seed, then the 4-net ensemble -> out_t1nn/pred4/
python -W ignore t1_nn.py train --tag c48f4 --all-days --width 48 --epochs 8 --lr 1.5e-3 --seed 4 > $O/train_c48f4.log 2>&1
st "c48f4 done ($?)"
python -W ignore t1_nn.py predict --tags c48 c48s3 c48f c48f4 --width 48 --out-sub pred4 > $O/predict_pred4.log 2>&1
st "pred4 done ($?)"
st "CHAIN DONE"
