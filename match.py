"""A paired match: model A (c2 with a bv mode) against an opponent, both seats on each kingdom, per-game results saved.

    python match.py --name gate_k16_m04 --bv 1 --k0 16 --k1 64 --margin 0.04 --opp c2 --kingdoms 200 [--student x.npy]

Kingdoms come from a seeded stream per --stream (gate / dev / reserved), so different runs play the same games.
Out: results/match_<name>.json (summary) + results/match_<name>.jsonl (games).
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
_A: dict = {}


def init(a):
    from bv import core
    _A.update(a)
    st = np.load(a["student"]).astype(np.float32) if a["student"] else None
    core.configure(a["k0"], a["k1"], a["se"], a["topn"], a["margin"], st, a["z"])


def job(g):
    from bv import core
    k, s = g["kingdom"], g["seat"]
    me = core.seat_for(_A["base"], k, _A["bv"])
    op = core.seat_for(_A["opp"], k)
    t = time.time()
    before = core.stats()
    r = core.play([me, op] if s == 0 else [op, me], k, g["seed"], g["seed"] ^ 0x5EED)
    after = core.stats()
    return {**g, "points": core.points(r, s), "secs": round(time.time() - t, 2), "capped": r["capped"],
            "error": r["error"], **{x: after[x] - before[x] for x in ("rollouts", "overrides", "buys")}}


def games(stream, n, skip=0):
    rng = random.Random(f"bv-v1 {stream} kingdoms")
    from bv.core import random_kingdom
    out = []
    for i in range(skip + n):
        k, seed = random_kingdom(rng), rng.getrandbits(32)
        if i >= skip:
            out += [{"i": i, "kingdom": k, "seed": seed, "seat": 0}, {"i": i, "kingdom": k, "seed": seed, "seat": 1}]
    return out


def summary(res):
    by = {}
    for r in res:
        by.setdefault(r["i"], []).append(r["points"])
    blocks = np.array([np.mean(v) for v in by.values()])
    rng = np.random.default_rng(0)
    boot = [blocks[rng.integers(0, len(blocks), len(blocks))].mean() for _ in range(2000)] if len(blocks) else [0]
    m = float(np.mean([r["points"] for r in res])) if res else 0.0
    b = sum(r["buys"] for r in res)
    return {"games": len(res), "score": round(m, 4), "ci95": [round(float(np.percentile(boot, 2.5)), 4),
            round(float(np.percentile(boot, 97.5)), 4)], "override_rate": round(sum(r["overrides"] for r in res) / max(b, 1), 4),
            "secs_per_game": round(float(np.mean([r["secs"] for r in res])), 2) if res else 0,
            "rollouts": sum(r["rollouts"] for r in res), "capped": sum(r["capped"] for r in res),
            "errors": sum(r["error"] for r in res)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", default="c2")
    ap.add_argument("--bv", type=int, default=1)
    ap.add_argument("--opp", default="c2")
    ap.add_argument("--k0", type=int, default=16)
    ap.add_argument("--k1", type=int, default=64)
    ap.add_argument("--se", type=float, default=1.5)
    ap.add_argument("--topn", type=int, default=5)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--student", default="")
    ap.add_argument("--z", type=float, default=0.0)
    ap.add_argument("--stream", default="gate")
    ap.add_argument("--kingdoms", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    a = vars(ap.parse_args())
    gs = games(a["stream"], a["kingdoms"])
    out = os.path.join(ROOT, "results", f"match_{a['name']}")
    res = []
    t = time.time()
    with mp.Pool(a["workers"], init, (a,)) as pool, open(out + ".jsonl", "w") as f:
        for i, r in enumerate(pool.imap_unordered(job, gs)):
            res.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
            if (i + 1) % 50 == 0 or i + 1 == len(gs):
                s = summary(res)
                json.dump({"name": a["name"], "args": a, "t": time.strftime("%Y-%m-%d %H:%M"), "done": i + 1 == len(gs),
                           "wall_secs": round(time.time() - t), **s}, open(out + ".json", "w"), indent=1)
                print(f"{a['name']}: {i + 1}/{len(gs)} score {100 * s['score']:.1f}% ci {s['ci95']} "
                      f"override {s['override_rate']:.2f}", flush=True)


if __name__ == "__main__":
    main()
