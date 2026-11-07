"""Figure 1 of the report: private vs public score of every full submission (report/submissions_public_private.csv).

  python report/make_figure.py
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
r = [x for x in csv.DictReader(open(HERE / "submissions_public_private.csv")) if x["privateScore"] and float(x["publicScore"]) > 0.85]
is_mem = lambda d: any(k in d.lower() for k in ("rec7g", "gate", "memory onset", "msmu", "month-pure", "pick a")) and "memory-free" not in d.lower()
pu = np.array([float(x["publicScore"]) for x in r]); pr = np.array([float(x["privateScore"]) for x in r])
m = np.array([is_mem(x["description"]) for x in r])
fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=150)
ax.scatter(pu[~m], pr[~m], s=22, c="#4a72b0", label="public (memory-free) onset model", alpha=.8)
ax.scatter(pu[m], pr[m], s=34, c="#d1495b", marker="D", label="recent-memory onset", alpha=.9)
lo, hi = 0.853, 0.889
ax.plot([lo, hi], [lo, hi], "--", c="#999", lw=1, label="private = public")
f = [x for x in r if "v51" in x["description"]][0]; a = [x for x in r if "v47" in x["description"]][0]
for x, t, dy in ((f, "final pick B (v51)\n0.88076 / 0.88281", 18), (a, "pick A (v47)\n0.88653 / 0.87625", -38)):
    ax.annotate(t, (float(x["publicScore"]), float(x["privateScore"])), xytext=(-95, dy), textcoords="offset points",
                fontsize=8, arrowprops=dict(arrowstyle="->", lw=.8))
ax.set_xlabel("public LB (March)"); ax.set_ylabel("private LB (April)"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_title("All full submissions: memory onset gains on April, loses on March", fontsize=10)
ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(HERE / "fig_pub_priv.png")
print(f"memory onset: mean private-public {(pr - pu)[m].mean():+.4f}; others {(pr - pu)[~m].mean():+.4f}")
