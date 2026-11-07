"""Onset-only (or any partial) queue probe: writes a submission whose only non-zero task is the given queue
pickle, so its LB score is 0.15 x the IoU of those windows."""
import pickle
import sys

import subkey
from common import CACHE, OUT

if not (CACHE / "subkey.npz").exists():
    subkey.build()
subkey.write_submission(OUT / f"{sys.argv[2]}.zip", queue=pickle.load(open(sys.argv[1], "rb")))
