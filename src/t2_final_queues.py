"""Task 2 queue files of the two final picks, rebuilt from the probability pickles.

  Pick B  queue_pub_c8.pkl  onset   = public anubisyy model (t2_pub.Public), thr 0.55, top-1 fallback
                            ongoing = c8: 0.2 LightGBM blend + 0.8 CNN x3 (t2_pipeline.py --nn-w 0.5 0.8), thr 0.45
  Pick A  queue_g_c8.pkl    onset   = rec7g memory onset (t2_pipeline.py defaults), thr 0.3, top-3 fallback
                            ongoing = the same c8

Each file maps window_id -> (6, L) int8 0/1 in scorer link order (fd_parameters / milepost order). The
probabilities are in travel order and carry the travel -> scorer permutation `order`: out[:, order] = pred.

Inputs (in --probs-dir, default OUT = $TFB_OUT):
  t2rec7gc8_queue_final_probs.pkl  wid -> (P, order, cond)  t2_pipeline.py --nn-w 0.5 0.8 --out t2rec7gc8_queue_final.pkl
  t2rec7g2_queue_final_probs.pkl   wid -> (P, order, cond)  t2_pipeline.py --out t2rec7g2_queue_final.pkl
  t2pub_scored_probs.pkl           wid -> P (travel order)  --pub below (same code as t2_mixpub.py, which also
                                                            writes queue files into kaggle_pkg/src_ds)
The onset part of the two pipeline pickles is the same (onset CNN weight 0.5 in both runs); rec7g onset is read
from t2rec7g2, as the original queue_g_g.pkl was.

Provenance of the submitted files (Oct 24 22:46): t2_mixpub.py wrote queue_pub_g.pkl and queue_g_g.pkl, which were
then updated with the ongoing windows of t2rec7gc8_queue_final.pkl. Because their onset and ongoing arrays come
from two different pickle loads, the files hold two int8 dtype objects; build() repeats that (a pickle round trip
per part) so the outputs are byte-identical to kaggle_pkg/src_ds, not just equal in content.

  python t2_final_queues.py                          rebuild from OUT, write to repro/, assert == kaggle_pkg/src_ds
  python t2_final_queues.py --pub                    also recompute the public onset probabilities first
  python t2_final_queues.py --report-only --ref-probs DIR   compare (no assert) and diff the probabilities vs DIR
"""
from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np

from common import OUT, QUEUE_PANELS, ROOT
from t2_model import decode

C8, REC7G, PUB = "t2rec7gc8_queue_final_probs.pkl", "t2rec7g2_queue_final_probs.pkl", "t2pub_scored_probs.pkl"
ONSET = {"pub": (0.55, 1), "g": (0.3, 3)}          # decode threshold, top-k fallback at T+30
ONGOING_THR = 0.45
PICKS = {"queue_pub_c8.pkl": "pub", "queue_g_c8.pkl": "g"}


def pub_probs(cond: dict) -> dict:
    """Public onset / ongoing model on the scored windows' T-60..T-5 history (t2_mixpub.py, lines 27-32)."""
    from t2_pub import Public, history_scored
    pub, pp = Public(), {}
    for p in QUEUE_PANELS:
        for split in ("validation", "private"):
            for wid, (V, Q, E, origin) in history_scored(p, split).items():
                pp[wid] = pub.probs(p, V, Q, E, origin, cond[wid])
    return pp


def scorer(pred: np.ndarray, order: np.ndarray) -> np.ndarray:
    out = np.zeros_like(pred, dtype=np.int8)
    out[:, order] = pred
    return out


def build(c8: dict, rec: dict, pub: dict, onset: str) -> dict:
    thr, fb = ONSET[onset]
    on, og = {}, {}
    for wid, (P, order, cond) in c8.items():
        if cond == "queue_onset":
            on[wid] = scorer(decode(pub[wid] if onset == "pub" else rec[wid][0], cond, thr, fb), order)
        else:
            og[wid] = scorer(decode(P, cond, ONGOING_THR), order)
    on, og = (pickle.loads(pickle.dumps(d)) for d in (on, og))    # byte layout of the originals (see docstring)
    return {wid: on[wid] if wid in on else og[wid] for wid in c8}


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a.astype(bool) | b.astype(bool)).sum()
    return 1.0 if u == 0 else float((a.astype(bool) & b.astype(bool)).sum() / u)


def compare(new: dict, ref: dict, cond: dict) -> tuple[bool, str]:
    lines, ok = [], list(new) == list(ref)
    for c in ("queue_onset", "queue_ongoing"):
        ws = [w for w in new if cond[w] == c]
        same = [np.array_equal(new[w], ref[w]) and new[w].dtype == ref[w].dtype for w in ws]
        cells = sum(int((new[w] != ref[w]).sum()) for w in ws)
        ious = [iou(new[w], ref[w]) for w in ws]
        ok &= all(same)
        ch =[w for w, s in zip(ws, same) if not s]
        lines.append(f"    {c:13s} identical {sum(same)}/{len(ws)} windows, cells differing {cells}, "
                     f"mean IoU new vs ref {np.mean(ious):.4f} (min {min(ious):.3f}); changed: "
                     f"{sum('validation' in w for w in ch)} public, {sum('private' in w for w in ch)} private")
    return ok, "\n".join(lines)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def prob_diff(name: str, new: dict, ref: dict, cond: dict) -> str:
    """Per condition: windows with identical probabilities, max |dP|, mean |dP|."""
    get = (lambda v: v[0]) if isinstance(next(iter(new.values())), tuple) else (lambda v: v)
    out =[f"  {name} vs reference ({len(new)} windows; key order equal: {list(new) == list(ref)})"]
    for c in ("queue_onset", "queue_ongoing"):
        ws = [w for w in new if cond[w] == c]
        if not ws:
            continue
        d = [np.abs(np.asarray(get(new[w]), float) - np.asarray(get(ref[w]), float)) for w in ws]
        same = sum(np.array_equal(get(new[w]), get(ref[w])) for w in ws)
        out.append(f"    {c:13s} identical {same}/{len(ws)}, max |dP| {max(x.max() for x in d):.2e}, "
                   f"mean |dP| {np.mean(np.concatenate([x.ravel() for x in d])):.2e}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--probs-dir", type=Path, default=OUT, help="folder with the three probability pickles (default OUT)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "repro", help="where the queue files go (default repro/)")
    ap.add_argument("--ref", type=Path, default=ROOT / "kaggle_pkg" / "src_ds", help="submitted queue files to compare with")
    ap.add_argument("--ref-probs", type=Path, help="folder with reference probability pickles to diff against")
    ap.add_argument("--pub", action="store_true", help="recompute t2pub_scored_probs.pkl with t2_pub.Public; written to "
                    "--probs-dir if absent there, else only compared with the existing file")
    ap.add_argument("--report-only", action="store_true", help="report differences instead of asserting equality")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    assert a.out_dir.resolve() not in (a.ref.resolve(), OUT.resolve(), (ROOT / "out").resolve()), "refusing to write there"

    c8 = pickle.load(open(a.probs_dir / C8, "rb"))
    rec = pickle.load(open(a.probs_dir / REC7G, "rb"))
    cond = {w: v[2] for w, v in c8.items()}
    assert list(rec) == list(c8) and all(rec[w][2] == cond[w] and np.array_equal(rec[w][1], c8[w][1]) for w in c8)
    on = [w for w in c8 if cond[w] == "queue_onset"]
    print(f"{len(c8)} windows ({len(on)} onset, {len(c8) - len(on)} ongoing) from {a.probs_dir}", flush=True)
    print(f"onset probabilities identical in {C8} and {REC7G}: "
          f"{sum(np.array_equal(c8[w][0], rec[w][0]) for w in on)}/{len(on)}", flush=True)

    pub_f = a.probs_dir / PUB
    if a.pub:
        pp = pub_probs(cond)
        if pub_f.exists():
            print("recomputed public probabilities vs existing file:\n"
                  + prob_diff(PUB, pp, pickle.load(open(pub_f, "rb")), cond), flush=True)
        else:
            pickle.dump(pp, open(pub_f, "wb"))
            print(f"wrote {pub_f}", flush=True)
    pub = pickle.load(open(pub_f, "rb"))

    if a.ref_probs:
        print(f"probabilities vs {a.ref_probs}:", flush=True)
        for f in (C8, REC7G, PUB):
            print(prob_diff(f, pickle.load(open(a.probs_dir / f, "rb")), pickle.load(open(a.ref_probs / f, "rb")), cond),
                  flush=True)

    all_ok = True
    for fname, onset in PICKS.items():
        q = build(c8, rec, pub, onset)
        dst = a.out_dir / fname
        pickle.dump(q, open(dst, "wb"))
        ref_f = a.ref / fname
        ok, rep = compare(q, pickle.load(open(ref_f, "rb")), cond)
        same_bytes = md5(dst) == md5(ref_f)
        print(f"{fname}: onset {onset} {ONSET[onset]}, ongoing c8 thr {ONGOING_THR} -> {dst}\n"
              f"  md5 {md5(dst)} (ref {md5(ref_f)}; byte-identical: {same_bytes}); equal to {ref_f}: {ok}\n{rep}", flush=True)
        all_ok &= ok
    if not a.report_only:
        assert all_ok, "rebuilt queue files differ from the reference"
        print("OK: both queue files equal the reference", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
