# Reproducing the final counting submission (private 0.88281)

Target file: `out/sub_msmU_stF_f0_y7_r3_q80_s80_n85-50_g100_nf5.zip`, built by Kaggle script kernel
`waelyt/tfb-assemble` **version 51** (`kaggle_pkg/assemble/run.py` + `kernel-metadata.json`) with the single job line
`msmU_stF 0.9 0.07 0.03 q=0.8 s=0.8 n=0.85,0.5 g=1.0 nf=5` in `kaggle_pkg/src_ds/asm_jobs.txt`. `run.py` turns it into
`python assemble.py --t1-dir <tfb-t1big output> --queue queue_msmU_stF.pkl --odme t4_l2.pkl --name <name> --pub-sub <anubisyy submission.csv> --w-ours 0.9 --extra-sub <yeash> <robin> --w-extra 0.07 0.03 --w-ours-flow 0.8 --shock-w 0.8 --nn-w 0.85 0.5 --nn-gap 1.0 --nn-file t1nn_post5.npz`
(anubisyy is passed but has share 0 = 1 - 0.9 - 0.07 - 0.03). `reproduce.sh` runs the local stages in order.

## Environment
Local: Windows 11, Python 3.11.9, numpy 2.4.6, pandas 3.0.6, scipy 1.17.1, pyarrow 24.0.0, scikit-learn 1.8.0, lightgbm 4.7.0,
xgboost 3.2.0, catboost 1.2.10, torch 2.11.0+cu128, RTX 4060 Laptop GPU (driver 621.14), 12 CPU threads.
Kaggle: `tfb-t1big` on a GPU session (XGBoost GPU hist); `tfb-assemble` on CPU (image Python 3.13, per `report/reproduce_plan.md`).
Paths (`src/common.py`): `TFB_REL` (release folder, default `<root>/kaggle_public`), `TFB_CACHE` (default `<root>/cache`), `TFB_OUT`
(default `<root>/out`). Some scripts hard-code `<root>/out_t1nn`, `out_st`, `kaggle_pkg/src_ds`, `ext/anubis_assets`
(`src/t1_nn*.py`, `src/t1_shock_post.py`, `src/t2_pub.py`, `scripts/*.py`), so use the project-root layout. Stages C, D3-D6, E
OVERWRITE files in `kaggle_pkg/src_ds`, `out_t1nn`, `out_st`; run on a fresh copy or let `reproduce.sh` guard (it refuses unless `REPRO_OVERWRITE=1`).

## Data layout
`kaggle_public/` = competition release. `ext/anubis_assets/` = CC0 checkpoints of the public anubisyy solution (onset_lgb.txt, onset.cbm,
ongoing_lgb.txt, queue_features.py). `kaggle_pkg/src_ds/` = dataset `waelyt/tfb-src` (copies of `src/*.py` + pickles/npz; assemble, common,
subkey, task4, t1_train, t1_features, t1_shock_post, t3_anchor verified byte-identical to `src/`).

## Stages (S = seed/GPU dependent, D = deterministic)

### A. Cache (D, ~6 min)
`python src/build_cache.py ; python src/subkey.py` -> `cache/<panel>/<split>/*.npy`, `cache/subkey.npz`.

### B. Task 1 tree model `t1_<panel>_<split>.npz` (Kaggle GPU, manual; S)
Kernel `waelyt/tfb-t1big` (`kaggle_pkg/t1big/run.py`; GPU, internet off, dataset `waelyt/tfb-src`, competition data). It runs
`TFB_T1_SHOCK=1 build_cache.py`, then `t1_train.py --device cuda --workers 3 --cap-train 8000 --rounds 20000 --month-prof --self-rounds 600`
(free-flow shock features, 8000 sampled targets/day, Feb early stopping, refit on all train days, month-own profiles, 600 rounds per-split
transductive fine-tune, self-frac 0.08, self-lr 0.05). Feb holdout S_state 0.9502. Output: 20 files `t1_<panel>_<split>.npz` (V, Q); `tfb-assemble`
locates them via `t1_D12_I5_N_validation.npz`. Push: `cd kaggle_pkg/t1big; kaggle kernels push -p .`. Its output is not stored locally.

### C. Neural imputer `kaggle_pkg/src_ds/t1nn_post5.npz` (S; local GPU, ~7 h sequential)
`src/t1_nn.py`; checkpoints `out_t1nn/net_<tag>.pt`; predictions `out_t1nn/<dir>/t1_<panel>_<split>.npz`. Always `T1NN_BASE=1 T1NN_CHAR=1`.
Literal records: `scripts/t1nn/chain.sh`..`chain7.sh`, `out_t1nn/log_<tag>.json` (args), `chain*.log` (timestamps).
1. c48 family, V2 unset (width 48, 8 ep, lr 1.5e-3): `c48` seed 1, `c48s3` seed 3 (Jun-Jan); `c48f` seed 2, `c48f4` seed 4 (`--all-days`). Then
   `predict --tags c48 c48s3 c48f c48f4 --width 48 --out-sub pred4` (chain.sh).
2. d64 family, `T1NN_V2=1` (width 64, 10 ep, lr 1.5e-3): `d64h` s11 (Jun-Jan), `d64f` s12, `d64f2` s13, `d64f3` s14 (`--all-days`);
   `predict --tags d64h d64f d64f2 d64f3 --width 64 --out-sub pred_d64b` (chain2 stream A + chain3).
3. g48 gap family, `T1NN_V2=1` (width 48, 8 ep, `--gap-p 1.0 --gap-w 2 --gap-real`): `g48h` s21 (Jun-Jan), `g48f` s22, `g48f2` s23 (`--all-days`);
   `predict --tags g48h g48f g48f2 --width 48 --out-sub pred_g48b` (chain2 stream B + chain3).
4. w96 family, `T1NN_V2=1 T1NN_MD=0` (width 96, 14 ep, lr 1.2e-3): `w96h` s11 (Jun-Jan), `w96f` s13, `w96f2` s14 (`--all-days`);
   `predict --tags w96h w96f w96f2 --width 96 --out-sub pred_w96b` (chain6 + chain7; chain7 proceeds only if w96h Feb S_state >= 0.9505; got 0.95078).
5. Combine (heredoc in chain7.sh), V and Q: where the whole link row is missing (`allgap = ~isfinite(load(p,s,"v")).any(axis=2)`) use
   (2*pred_d64b + pred_g48b)/3, else (pred4 + 2*pred_d64b + 2*pred_w96b)/5 -> `out_t1nn/pred11/`.
6. `t1_nn_post2.main(out_t1nn/pred11, kaggle_pkg/src_ds/t1nn_post5.npz)` (submission-key order).
Times on the 4060 (logs): c48 ~35 min, c48f ~37 min, c48s3 ~33 min, c48f4 ~37 min, d64 ~25-40 min, g48 ~25-40 min (two streams were run in parallel),
w96h/f/f2 ~63 min each. Not on the path: chain4 (transductive fine-tune, post4), chain5 (MD=1), pred2/8/9/10, t1nn_post{,2,3,4,6}.

### C2. Free-flow shock `kaggle_pkg/src_ds/t1_shock_ps.npz` (D, minutes)
`cd src; python t1_shock_post.py` (no flags; invocation per its docstring). Reads only the scored split's own masked layer (`t1_features.shock_grid`);
writes `ps` and `gate = 0.92 vf`; used as `--shock-w 0.8`.

### D. Task 2 queue `kaggle_pkg/src_ds/queue_msmU_stF.pkl`
- **D1 (S, ~47 min)** `cd src; TFB_OUT=<empty dir> python t2_pipeline.py --nn-w 0.5 0.8 --out t2rec7gc8_queue_final.pkl` and `... python t2_pipeline.py --out t2rec7g2_queue_final.pkl`
  (`repro/README.md`). Needed ONLY for per-window `(order, condition)` (travel -> scorer link permutation, onset/ongoing label). `order`/`cond` were checked
  identical to the original `out_yr/t2yrF_queue_final_probs.pkl` that the shipped ST file used.
- **D2 (D, minutes)** `python t2_final_queues.py --pub --probs-dir <that dir> --out-dir ../repro/from_scratch --report-only` -> `<dir>/t2pub_scored_probs.pkl`
  (anubisyy onset LGBM+CatBoost from `ext/anubis_assets`, reads only `window_history` T-60..T-5) and `queue_pub_c8.pkl`, whose ONSET windows (decode thr 0.55, top-1) are the
  public onset. Recompute reproduced the shipped probabilities 160/160 bit-identically (`repro/t2_final_queues_from_out.log`).
- **D3 (S, 577-843 s/seed on the 4060; rows 4.2 GB)** space-time ongoing net `src/t2_stnet.py`:
  `TFB_YOUNG_STEP=1 TFB_RAMPS=1 python scripts/st_build_rows.py`; `python scripts/st_train.py --seeds 7 11 23` and `--seeds 31 47` (tag st, Jun-Jan, 24 epochs,
  best epoch by Feb loss) -> `out_st/st_ongoing_s{7,11,23,31,47}.pt`.
- **D4 (D given D3)** `TFB_RAMPS=1 TFB_FILL_BLANKS=1 python scripts/st_scored.py --yr <dir>/t2rec7gc8_queue_final.pkl --w 1.0 --thr 0.5 --out t2st_queue.pkl --probs-out st_scored_probs_F.pkl`
  (original used `--yr out_yr/t2yrF_queue_final.pkl`) -> `out_st/st_scored_probs_F.pkl` (5-seed mean).
- **D5 (D)** `queue_pub_stF.pkl`: onset windows = `queue_pub_c8.pkl`; ongoing windows = `decode(st_scored_probs_F, "queue_ongoing", 0.5)` with `out[:, order] = pred`.
  The inline script that wrote the original (Nov 1 12:52) was not saved; `reproduce.sh` holds a reconstruction (same logic as the heredoc in `scripts/st10_chain.sh`),
  verified against the shipped file: onset 80/80 equal to `queue_pub_c8.pkl`, ongoing 80/80 equal to `decode(out_st/st_scored_probs_F.pkl, 0.5)`.
- **D6 (D, ~1 min)** `python scripts/onset_msm_scored.py 1.0 1 0.5 0 msmU` (RHO 1.0, MINEV 1, W 0.5, SLOT0 0; set `TFB_OUT` to the D1 dir) reads `t2pub_scored_probs.pkl` +
  `queue_pub_stF.pkl`, writes `queue_msmU_stF.pkl` (8 public / 10 private onset windows change). Memory = onset footprints (links queued within 30 min of a corridor onset start)
  on the SAME split's earlier days, hour band +-1 h, blended 0.5 into public P[T+30], decode 0.55 top-1.

### E. Task 4 `kaggle_pkg/src_ds/t4_l2.pkl` (D, minutes)
`cd src; python make_t4_probes.py` -> `out/t4_candidates.pkl` (L2 lambda 0.05 via `task4.solve_l2`, plus KL variants). `t4_l2.pkl` = its 20 `("l2", panel, split)` entries
(verified array-equal to the shipped file; the extraction line was not saved): `pickle.dump({k: v for k, v in c.items() if k[0] == "l2"}, open(".../t4_l2.pkl", "wb"))`.

### F. Assembly (Kaggle CPU, manual; D given inputs)
1. Third-party Task 1 state sources (`kernel_sources`; share of the final state): `anubisyy/catboost-lgbm-nnls-pipeline-top-50-0-86517` 0.00,
   `yeashusemwal/trafficflowbench-trees-u-net-cnn-lb-0-87` 0.07, `robinamirbahar/trafficflowbench-physics-guided-state-odme` 0.03; ours 0.90 (speed: 0.85 of ours is the neural
   imputer, flow 0.5; our flow share 0.8; shock 0.8; all-link gap cells 100% neural imputer). Their `submission.csv` outputs are used unmodified (picked by path substring).
   Kernel sources: `waelyt/tfb-t1big` + the three above. Dataset `waelyt/tfb-src` must contain `queue_msmU_stF.pkl`, `t4_l2.pkl`, `t1nn_post5.npz`, `t1_shock_ps.npz`, `src/*.py`.
2. `asm_jobs.txt` = exactly `msmU_stF 0.9 0.07 0.03 q=0.8 s=0.8 n=0.85,0.5 g=1.0 nf=5`.
3. `cd kaggle_pkg/src_ds; kaggle datasets version -p . -m msg` ; `cd ../assemble; kaggle kernels push -p .` ; `kaggle kernels output waelyt/tfb-assemble -p out`.

## Causality / post-T verification (Task 2 rerun deletes all rows after each window origin T)
- Public onset model: reads only the organizer `window_history.parquet` (T-60..T-5 per window).
- Month-memory onset (msmU): `python scripts/msm_post_t_check.py` -> `logs/msm_post_t_check.log`: 80/80 onset windows identical after blanking every masked cell after T
  (rest of day, later days, later splits; `src/t2_post_t_check.truncated_load`), 80/80 reproduce `queue_msmU_stF.pkl`, negative control (cut a day early) changes 8.
- Space-time ongoing inputs: `TFB_RAMPS=1 TFB_FILL_BLANKS=1 python scripts/st2_post_t_check.py` -> `logs/st2_post_t_check_all.log`: 80/80 windows bit-identical for X (all columns and the
  non-LONG ones the ST nets use), grid G and context C; negative control (cut at T-20) changes them. ~11 min.
- Feature path of the yr/c8 models (memory, ramps, bridged blanks): `cd src; TFB_YOUNG_STEP=1 TFB_RAMPS=1 TFB_FILL_BLANKS=1 python t2_post_t_check.py` -> `logs/post_t_check_all_flags.log`: 160 windows, 0 differ.
- Task 1 uses the released masked layers of the scored split by design (offline imputation); Task 4 uses the released counts; neither depends on Task 2 rows.

## Reproducibility notes
- C and D3 retraining, and B (GPU XGBoost), are not bit-exact; the 1% tolerance is expected to absorb it (extra NN seeds moved the public score by ~0.00003; a few
  changed queue windows move it ~0.0004 each).
- Resolved: `tfb-t1big` is version 1 (cap 8000, `TFB_T1_SHOCK=1`, Feb holdout S_state 0.9502), the only run, wired into `tfb-assemble` kernel_sources since assemble v33 (Nov 1). Anubis CC0 assets = Kaggle dataset `anubisyy/trafficflowbench-086517-public-assets` (local copy `ext/anubis_assets`, checksums verified; loaded by `src/t2_pub.py`).
- Shell commands for c48/c48s3/c48f, the queue_pub_stF writer and the t4_l2 extraction were not saved; equivalents verified against shipped files. Exact `st_train.py` seed-group
  invocation (7 11 23 / 31 47) is inferred from `out/st_train.log` and `out/st_train_more.log`.
