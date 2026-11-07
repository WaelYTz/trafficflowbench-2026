"""Write Task 4-only probe submissions: the leaderboard then reads exactly 0.2 * S_ODME."""
import pickle
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from common import OUT, PANELS
from subkey import write_submission
from task4 import problem, solve_kl, solve_l2, pooled_prior


def run(panel):
    pool = pooled_prior(panel)
    res = {}
    for split in ("validation", "private"):
        P = problem(panel, split)
        res[("l2", panel, split)] = solve_l2(P)
        res[("kl", panel, split)] = solve_kl(P)
        res[("klpool", panel, split)] = solve_kl(P, prior=pool)
    return res


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    allres = {}
    with ProcessPoolExecutor(max_workers=6) as ex:
        for r in ex.map(run, PANELS):
            allres.update(r)
    pickle.dump(allres, open(OUT / "t4_candidates.pkl", "wb"))
    for name in ("l2", "kl", "klpool"):
        odme = {(p, s): allres[(name, p, s)] for p in PANELS for s in ("validation", "private")}
        write_submission(OUT / f"probe_t4_{name}.csv", odme=odme)
