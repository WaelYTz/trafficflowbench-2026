"""Task 1 retrain with the free-flow shock features (TFB_T1_SHOCK=1: month-own link level, per-timestamp corridor
shock and the censored free-flow prediction, t1_features.shock_grid), same recipe as the final Task 1 but twice the training cells (cap 8000,
20000 rounds with Feb early stopping, refit on all train days, month-own profiles, 600 rounds of per-split
transductive fine-tuning). Writes t1_<panel>_<split>.npz (for tfb-assemble), t1_v/q.json and t1_report.json
(Feb holdout S_state; the shock-free v2 run scored 0.9487)."""
import glob, os, shutil, subprocess, sys
rel = os.path.dirname(glob.glob("/kaggle/input/**/submission_key.csv", recursive=True)[0])
src = os.path.dirname(glob.glob("/kaggle/input/**/assemble.py", recursive=True)[0])
print("data:", rel, "| code:", src, flush=True)
shutil.copytree(src, "/kaggle/working/src", dirs_exist_ok=True)
out = "/kaggle/working/out"
os.makedirs(out, exist_ok=True)
os.environ.update(TFB_REL=rel, TFB_CACHE="/tmp/cache", TFB_OUT=out, TFB_T1_SHOCK="1")
w = "/kaggle/working/src"
subprocess.run(["nvidia-smi"])
run = lambda *a: subprocess.run([sys.executable, "-W", "ignore", *a], cwd=w, check=True)
run("build_cache.py")
run("t1_train.py", "--device", "cuda", "--workers", "3", "--cap-train", "8000", "--rounds", "20000",
    "--month-prof", "--self-rounds", "600")
shutil.rmtree(w, ignore_errors=True)
