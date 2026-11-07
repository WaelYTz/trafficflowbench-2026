"""Builds upload files from tfb-t1month's Task 1 outputs (v2 models, month-own profiles, per-split fine-tuning)
plus the Task 2 / Task 4 pickles in tfb-src. asm_jobs.txt lists "<queue> <w_ours>" pairs: w_ours < 1 blends
the public anubisyy Task 1 state at 1 - w_ours. Output: sub_<queue>_f<public %>.zip.
"probe <name>" lines write an onset-only probe from probeq_<name>.pkl."""
import glob, os, shutil, subprocess, sys
rel = os.path.dirname(glob.glob("/kaggle/input/**/submission_key.csv", recursive=True)[0])
src = os.path.dirname(glob.glob("/kaggle/input/**/assemble.py", recursive=True)[0])
t1 = sorted(glob.glob("/kaggle/input/**/t1_D12_I5_N_validation.npz", recursive=True))
assert len(t1) == 1, t1
t1 = os.path.dirname(t1[0])
pub = [p for p in glob.glob("/kaggle/input/**/submission.csv", recursive=True) if "catboost" in p or "anubis" in p]
# extra public state sources, in asm_jobs token order after <w_ours>: yeash (U-Net), robin (physics-guided)
EXTRA = [[p for p in glob.glob("/kaggle/input/**/submission.csv", recursive=True) if key in p] for key in ("u-net", "physics-guided")]
print("data:", rel, "| code:", src, "| task1:", t1, "| public state:", pub, EXTRA, flush=True)
shutil.copytree(src, "/kaggle/working/src", dirs_exist_ok=True)
os.environ.update(TFB_REL=rel, TFB_CACHE="/tmp/cache", TFB_OUT="/kaggle/working/out")
w_ = "/kaggle/working/src"
for line in open(os.path.join(w_, "asm_jobs.txt")):
    if not line.strip():
        continue
    if line.startswith("probe"):
        name = line.split()[1]
        subprocess.run([sys.executable, "probe.py", f"probeq_{name}.pkl", f"probe_q_{name}_onset"], cwd=w_, check=True)
        continue
    if line.startswith("qprobe"):          # queue-only probe: "qprobe <queue>"
        q = line.split()[1]
        subprocess.run([sys.executable, "assemble.py", "--t1-dir", "none", "--queue", f"queue_{q}.pkl", "--odme", "none",
                        "--name", f"probe_q_{q}"], cwd=w_, check=True)
        continue
    probe = line.startswith("stateprobe")  # state-only probe: "stateprobe <w_ours> [<w_yeash> <w_robin>]"
    if probe:
        line = "none " + line.split(None, 1)[1]
    q, w = line.split()[0], float(line.split()[1])
    wf = [float(x[2:]) for x in line.split()[2:] if x.startswith("q=")]   # "q=<w>": our share for flow only
    ws = [float(x[2:]) for x in line.split()[2:] if x.startswith("s=")]   # "s=<w>": free-flow shock correction
    wn = [x[2:].split(",") for x in line.split()[2:] if x.startswith("n=")]  # "n=<w_v>,<w_q>": neural imputer share of ours
    wg = [float(x[2:]) for x in line.split()[2:] if x.startswith("g=")]      # "g=<w>": neural share on all-link gap cells
    nf = [x[3:] for x in line.split()[2:] if x.startswith("nf=")]               # "nf=2": t1nn_post2.npz (imputer v2)
    an = [x[2:].split(",") for x in line.split()[2:] if x.startswith("a=")]  # "a=<w>,<alpha>,<max_run>": Task 3 anchor correction
    rn = [x[2:].split(",") for x in line.split()[2:] if x.startswith("r=")]  # "r=<rho>,<max_run>": Task 3 anchor-noise offsets
    wx = [float(x) for x in line.split()[2:] if not x[:2] in ("q=", "s=", "n=", "g=", "a=", "r=") and not x.startswith("nf=")]
    name = f"sub_{q}_f{round(100 * (1 - w - sum(wx)))}" + "".join(f"_{tag}{round(100 * x)}" for tag, x in zip("yr", wx) if x)
    name += f"_q{round(100 * wf[0])}" if wf else ""
    name += f"_s{round(100 * ws[0])}" if ws else ""
    name += f"_n{round(100 * float(wn[0][0]))}-{round(100 * float(wn[0][1]))}" if wn else ""
    name += f"_g{round(100 * wg[0])}" if wg else ""
    name += f"_nf{nf[0]}" if nf else ""
    name += f"_a{round(100 * float(an[0][0]))}-{round(100 * float(an[0][1]))}-{an[0][2]}" if an else ""
    name += f"-g{round(100 * float(an[0][3]))}" if an and len(an[0]) > 3 else ""
    name += f"_r{round(100 * float(rn[0][0]))}-{rn[0][1]}" if rn else ""
    if probe:
        name = "probe_state" + name[len("sub_none"):]
    cmd = [sys.executable, "assemble.py", "--t1-dir", t1, "--queue", "none" if probe else f"queue_{q}.pkl",
           "--odme", "none" if probe else "t4_l2.pkl", "--name", name]
    if w < 1:
        cmd += ["--pub-sub", pub[0], "--w-ours", str(w)]
    if any(wx):
        cmd += ["--extra-sub"] + [EXTRA[i][0] for i, x in enumerate(wx) if x] + ["--w-extra"] + [str(x) for x in wx if x]
    if wf:
        cmd += ["--w-ours-flow", str(wf[0])]
    if ws:
        cmd += ["--shock-w", str(ws[0])]
    if wn:
        cmd += ["--nn-w", wn[0][0], wn[0][1]]
    if wg:
        cmd += ["--nn-gap", str(wg[0])]
    if nf:
        cmd += ["--nn-file", f"t1nn_post{nf[0]}.npz"]
    if an:
        cmd += ["--anchor"] + an[0]
    if rn:
        cmd += ["--noise"] + rn[0]
    subprocess.run(cmd, cwd=w_, check=True)
shutil.rmtree(w_, ignore_errors=True)
