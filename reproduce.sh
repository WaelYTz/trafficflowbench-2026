#!/usr/bin/env bash
# Local stages for the final counting submission (see REPRODUCE.md). NOT executed by the author.
# Run from Git Bash at the project root. Needs a GPU for stages C and D3. Kaggle stages are manual (comments).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
SRC="$ROOT/src"; DS="$ROOT/kaggle_pkg/src_ds"
NEW="$ROOT/out_new"                      # TFB_OUT for the Task 2 probability pickles (must start empty)
export PYTHONUNBUFFERED=1

# Guard: this script overwrites shipped artifacts (src_ds pickles, out_t1nn nets, out_st nets).
if [ "${REPRO_OVERWRITE:-0}" != "1" ] && { [ -e "$ROOT/out_st/st_ongoing_s7.pt" ] || [ -e "$ROOT/out_t1nn/net_c48.pt" ]; }; then
  echo "Shipped artifacts exist; run in a fresh copy or set REPRO_OVERWRITE=1 (back up kaggle_pkg/src_ds, out_st, out_t1nn first)."; exit 1
fi
mkdir -p "$NEW" "$ROOT/out_t1nn" "$ROOT/out_st"
[ -d "$DS" ] && [ ! -d "$ROOT/kaggle_pkg/src_ds_shipped" ] && cp -r "$DS" "$ROOT/kaggle_pkg/src_ds_shipped"

# ---------------------------------------------------------------- A. cache (~6 min)
( cd "$SRC" && python -W ignore build_cache.py && python -W ignore subkey.py )

# ---------------------------------------------------------------- B. Task 1 trees: MANUAL on Kaggle (GPU)
#   cd kaggle_pkg/t1big && kaggle kernels push -p .      # kernel waelyt/tfb-t1big, runs
#   TFB_T1_SHOCK=1 t1_train.py --device cuda --workers 3 --cap-train 8000 --rounds 20000 --month-prof --self-rounds 600
#   The kernel output (t1_<panel>_<split>.npz) stays on Kaggle and is read by tfb-assemble.

# ---------------------------------------------------------------- C. neural imputer -> t1nn_post5.npz (~7 h GPU)
export T1NN_BASE=1 T1NN_CHAR=1
O="$ROOT/out_t1nn"
nn() { ( cd "$SRC" && python -W ignore t1_nn.py "$@" ); }
# C1. c48 family (V2 off)
T1NN_V2=0 nn train --tag c48   --width 48 --epochs 8 --eval-every 2 --lr 1.5e-3 --seed 1 > $O/train_c48.log 2>&1
T1NN_V2=0 nn train --tag c48f  --all-days --width 48 --epochs 8 --lr 1.5e-3 --seed 2 > $O/train_c48f.log 2>&1
T1NN_V2=0 nn train --tag c48s3 --width 48 --epochs 8 --eval-every 4 --lr 1.5e-3 --seed 3 > $O/train_c48s3.log 2>&1
T1NN_V2=0 nn train --tag c48f4 --all-days --width 48 --epochs 8 --lr 1.5e-3 --seed 4 > $O/train_c48f4.log 2>&1
T1NN_V2=0 nn predict --tags c48 c48s3 c48f c48f4 --width 48 --out-sub pred4 > $O/predict_pred4.log 2>&1
# C2. V2 families
export T1NN_V2=1 T1NN_MD=0
nn train --tag d64h  --width 64 --epochs 10 --eval-every 5 --lr 1.5e-3 --seed 11 > $O/train_d64h.log 2>&1
nn train --tag d64f  --all-days --width 64 --epochs 10 --lr 1.5e-3 --seed 12 > $O/train_d64f.log 2>&1
nn train --tag d64f2 --all-days --width 64 --epochs 10 --lr 1.5e-3 --seed 13 > $O/train_d64f2.log 2>&1
nn train --tag d64f3 --all-days --width 64 --epochs 10 --lr 1.5e-3 --seed 14 > $O/train_d64f3.log 2>&1
nn predict --tags d64h d64f d64f2 d64f3 --width 64 --out-sub pred_d64b > $O/predict_d64b.log 2>&1
nn train --tag g48h  --width 48 --epochs 8 --eval-every 4 --lr 1.5e-3 --seed 21 --gap-p 1.0 --gap-w 2 --gap-real > $O/train_g48h.log 2>&1
nn train --tag g48f  --all-days --width 48 --epochs 8 --lr 1.5e-3 --seed 22 --gap-p 1.0 --gap-w 2 --gap-real > $O/train_g48f.log 2>&1
nn train --tag g48f2 --all-days --width 48 --epochs 8 --lr 1.5e-3 --seed 23 --gap-p 1.0 --gap-w 2 --gap-real > $O/train_g48f2.log 2>&1
nn predict --tags g48h g48f g48f2 --width 48 --out-sub pred_g48b > $O/predict_g48b.log 2>&1
nn train --tag w96h  --width 96 --epochs 14 --lr 1.2e-3 --seed 11 --eval-every 7 > $O/train_w96h.log 2>&1
grep "NN ep14" $O/train_w96h.log   # expect S_state ~0.9508 (chain7 required >= 0.9505)
nn train --tag w96f  --all-days --width 96 --epochs 14 --lr 1.2e-3 --seed 13 > $O/train_w96f.log 2>&1
nn train --tag w96f2 --all-days --width 96 --epochs 14 --lr 1.2e-3 --seed 14 > $O/train_w96f2.log 2>&1
nn predict --tags w96h w96f w96f2 --width 96 --out-sub pred_w96b > $O/predict_w96b.log 2>&1
# C3. combine -> pred11 -> t1nn_post5.npz (literal copy of the chain7.sh heredoc)
( cd "$SRC" && python -W ignore - <<'PY'
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
python -W ignore -c "import t1_nn_post2 as m; from common import ROOT; m.main(ROOT/'out_t1nn'/'pred11', ROOT/'kaggle_pkg'/'src_ds'/'t1nn_post5.npz')" )
unset T1NN_V2 T1NN_MD T1NN_BASE T1NN_CHAR

# ---------------------------------------------------------------- C4. free-flow shock -> t1_shock_ps.npz
( cd "$SRC" && python -W ignore t1_shock_post.py )

# ---------------------------------------------------------------- D. Task 2 queue -> queue_msmU_stF.pkl
# D1. pipeline probabilities, used only for per-window (order, condition); TFB_OUT = empty dir (~47 min)
( cd "$SRC" && TFB_OUT="$NEW" python -W ignore t2_pipeline.py --nn-w 0.5 0.8 --out t2rec7gc8_queue_final.pkl )
( cd "$SRC" && TFB_OUT="$NEW" python -W ignore t2_pipeline.py --out t2rec7g2_queue_final.pkl )
# D2. public onset probabilities + queue_pub_c8.pkl (its onset windows = public onset, thr 0.55 top-1)
( cd "$SRC" && TFB_OUT="$NEW" python -W ignore t2_final_queues.py --pub --probs-dir "$NEW" --out-dir "$ROOT/repro/from_scratch" --report-only )
# D3. space-time ongoing nets (GPU; rows 4.2 GB; ~14 min/seed)
TFB_YOUNG_STEP=1 TFB_RAMPS=1 python -W ignore scripts/st_build_rows.py
python -W ignore scripts/st_train.py --seeds 7 11 23 > out/st_train.log 2>&1
python -W ignore scripts/st_train.py --seeds 31 47 > out/st_train_more.log 2>&1
# D4. 5-seed ST probabilities on the 160 scored windows (--yr only supplies order/condition)
TFB_RAMPS=1 TFB_FILL_BLANKS=1 python -W ignore scripts/st_scored.py --yr "out_new/t2rec7gc8_queue_final.pkl" \
    --w 1.0 --thr 0.5 --out t2st_queue.pkl --probs-out st_scored_probs_F.pkl
# D5. queue_pub_stF.pkl = public onset + ST ongoing (thr 0.5). Reconstruction; verified equal to the shipped file from the shipped inputs.
( cd "$SRC" && TFB_OUT="$NEW" python -W ignore - <<'PY'
import pickle
import numpy as np
from common import ROOT
from t2_model import decode
DS = ROOT / "kaggle_pkg" / "src_ds"
pubc8 = pickle.load(open(ROOT / "repro" / "from_scratch" / "queue_pub_c8.pkl", "rb"))   # onset part used
probs = pickle.load(open(ROOT / "out_new" / "t2rec7gc8_queue_final_probs.pkl", "rb"))   # (P, order, cond)
st = pickle.load(open(ROOT / "out_st" / "st_scored_probs_F.pkl", "rb"))
q = {}
for wid, (P, order, cond) in probs.items():
    if cond != "queue_ongoing":
        q[wid] = pubc8[wid]
    else:
        out = np.zeros_like(pubc8[wid]); out[:, order] = decode(st[wid], cond, 0.5); q[wid] = out
pickle.dump(q, open(DS / "queue_pub_stF.pkl", "wb"))
print("queue_pub_stF.pkl:", len(q), "windows")
PY
)
# D6. month-pure memory onset (needs out_new/t2pub_scored_probs.pkl via TFB_OUT)
TFB_OUT="$NEW" python -W ignore scripts/onset_msm_scored.py 1.0 1 0.5 0 msmU     # -> src_ds/queue_msmU_stF.pkl

# ---------------------------------------------------------------- E. Task 4 L2 (lambda 0.05) -> t4_l2.pkl
( cd "$SRC" && python -W ignore make_t4_probes.py && python -W ignore - <<'PY'
import pickle
from common import ROOT, OUT
c = pickle.load(open(OUT / "t4_candidates.pkl", "rb"))
pickle.dump({k: v for k, v in c.items() if k[0] == "l2"}, open(ROOT / "kaggle_pkg" / "src_ds" / "t4_l2.pkl", "wb"))
PY
)

# ---------------------------------------------------------------- F. Kaggle assembly: MANUAL
#   1. kaggle_pkg/src_ds/asm_jobs.txt must be exactly:
#        msmU_stF 0.9 0.07 0.03 q=0.8 s=0.8 n=0.85,0.5 g=1.0 nf=5
#      and src/*.py copied into src_ds (they are identical to src/).
#   2. cd kaggle_pkg/src_ds && kaggle datasets version -p . -m "final msmU_stF"
#   3. (t1big output must exist) cd ../assemble && kaggle kernels push -p .       # waelyt/tfb-assemble
#   4. kaggle kernels output waelyt/tfb-assemble -p out  ->  out/sub_msmU_stF_f0_y7_r3_q80_s80_n85-50_g100_nf5.zip
#      kernel_sources: waelyt/tfb-t1big, anubisyy/catboost-lgbm-nnls-pipeline-top-50-0-86517,
#      yeashusemwal/trafficflowbench-trees-u-net-cnn-lb-0-87, robinamirbahar/trafficflowbench-physics-guided-state-odme

# ---------------------------------------------------------------- Post-T / causality checks (local)
TFB_OUT="$NEW" python -W ignore scripts/msm_post_t_check.py > "$NEW/msm_post_t_check.log" 2>&1
TFB_RAMPS=1 TFB_FILL_BLANKS=1 python -W ignore scripts/st2_post_t_check.py > "$NEW/st2_post_t_check_all.log" 2>&1
( cd "$SRC" && TFB_YOUNG_STEP=1 TFB_RAMPS=1 TFB_FILL_BLANKS=1 python -W ignore t2_post_t_check.py > "$NEW/post_t_check_all_flags.log" 2>&1 )
echo "local stages done"
