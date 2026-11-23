"""Exactness gate for the fast rollout net (c/fastnet.h): the same games of c2 vs c2 with rl_decide (reference) and with
the fast agent everywhere (bv_set_fast(2)). Gate: >= 98% identical games (winners, scores, turns) and the fast score
inside the reference's 95% interval. Out: results/fast_equivalence.json.

    python experiments/fast_equivalence.py [--kingdoms 1000]
"""
import argparse, json, math, multiprocessing as mp, os, random, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def job(g):
    from bv import core
    k = g["k"]
    out = []
    for f in (0, 2):
        core.lib.bv_set_fast(f)
        r = core.play([core.seat_for("c2", k), core.seat_for("c2", k)], k, g["seed"], g["seed"] ^ 0x5EED)
        out.append((tuple(r["winners"]), tuple(r["scores"]), tuple(r["turns"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kingdoms", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    from bv.core import random_kingdom
    rng = random.Random("fast equivalence")
    gs = [{"k": random_kingdom(rng), "seed": rng.getrandbits(32)} for _ in range(a.kingdoms)]
    t = time.time()
    with mp.Pool(a.workers) as p:
        res = p.map(job, gs, chunksize=10)
    same = sum(r[0] == r[1] for r in res) / len(res)
    s0 = sum((0.5 if len(r[0][0]) == 2 else float(0 in r[0][0])) for r in res) / len(res)
    s1 = sum((0.5 if len(r[1][0]) == 2 else float(0 in r[1][0])) for r in res) / len(res)
    h = 1.96 * math.sqrt(s0 * (1 - s0) / len(res))
    ok = same >= 0.98 and abs(s1 - s0) <= h
    out = {"games": len(res), "identical": round(same, 4), "seat0_score_ref": round(s0, 4), "seat0_score_fast": round(s1, 4),
           "ref_ci95": [round(s0 - h, 4), round(s0 + h, 4)], "passed": ok, "secs": round(time.time() - t)}
    json.dump(out, open(os.path.join(ROOT, "results", "fast_equivalence.json"), "w"), indent=1)
    print(out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
