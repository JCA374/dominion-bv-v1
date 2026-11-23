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
    core.lib.bv_set_act(a["act"])


def job(g):
    from bv import core
    k, s = g["kingdom"], g["seat"]
    me = core.seat_for(_A["base"], k, _A["bv"])
    op = core.seat_for(_A["opp"], k)
    t = time.time()
    before = core.stats()
    r = core.play([me, op] if s == 0 else [op, me], k, g["seed"], g["seed"] ^ 0x5EED)
    after = core.stats()
    import ctypes
    acts = (ctypes.c_long * 5)()
    core.lib.bv_act_stats(acts)
    act = list(acts)
    prev = _A.setdefault("_act", [0] * 5)
    dact = [x - y for x, y in zip(act, prev)]
    _A["_act"] = act
    return {**g, "act_graded": dact[0], "act_overrides": dact[1], "act_play_not_stop": dact[2],
            "act_stop_not_play": dact[3], "act_other_action": dact[4],
            "lat": [round(float(x), 5) for x in core.latencies()],
            "act_log": [[round(float(v), 4) for v in row] for row in core.action_log()], "points": core.points(r, s), "secs": round(time.time() - t, 2), "capped": r["capped"],
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


STREAMS = {"gate": 200, "dev": 300, "dev2": 300, "reserved": 2000, "confirm": 2000, "suite": 300}


def eval_kingdoms():
    """Canonical keys of every kingdom any evaluation stream can deal (collect.py never trains on them)."""
    from bv.core import random_kingdom
    out = set()
    for st, n in STREAMS.items():
        rng = random.Random(f"bv-v1 {st} kingdoms")
        for _ in range(n):
            out.add(",".join(sorted(random_kingdom(rng))))
            rng.getrandbits(32)
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
    lat = np.array([x for r in res for x in r.get("lat", [])])
    alog = [row for r in res for row in r.get("act_log", [])]
    from bv.core import NAME
    nm = lambda a: "end actions" if int(a) == 33 else NAME[int(a)]  # noqa: E731
    changes = {}
    for row in alog:
        kk = f"{nm(row[1])} -> {nm(row[2])}"
        changes[kk] = changes.get(kk, 0) + 1
    return {"latency_s": ({"decisions": int(len(lat)), "p50": round(float(np.percentile(lat, 50)), 4),
                           "p95": round(float(np.percentile(lat, 95)), 4)} if len(lat) else None),
            "action_overrides": ({"n": len(alog), "mean_gain": round(float(np.mean([r[3] for r in alog])), 4),
                                  "top_changes": sorted(changes.items(), key=lambda kv: -kv[1])[:12]} if alog else None),"games": len(res), "score": round(m, 4), "ci95": [round(float(np.percentile(boot, 2.5)), 4),
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
    ap.add_argument("--act", type=int, default=0, help="1: also grade actions, 2: actions only")
    ap.add_argument("--resume", action="store_true", help="keep the games already in the .jsonl")
    ap.add_argument("--stream", default="gate")
    ap.add_argument("--kingdoms", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    a = vars(ap.parse_args())
    if a["stream"] == "suite" and not os.environ.get("BV_SO") and os.path.exists(os.path.join(ROOT, "bv_suite.so")):
        os.environ["BV_SO"] = os.path.join(ROOT, "bv_suite.so")   # all suite arms on one frozen build (the old sampler)
    gs = games(a["stream"], a["kingdoms"])
    out = os.path.join(ROOT, "results", f"match_{a['name']}")
    from bv import core as _c
    manifest = {k: v for k, v in a.items() if k not in ("workers", "resume")}
    manifest["models"] = {m: _c.sha(m) for m in ("rl3_c2_2654.npy", "rl3_c1_2300.npy", "rl1_r2_2200.npy")}
    manifest["student_sha"] = (__import__("hashlib").sha256(open(a["student"], "rb").read()).hexdigest()[:16]
                               if a["student"] else None)
    res = []
    if a["resume"] and os.path.exists(out + ".json"):
        old = json.load(open(out + ".json")).get("manifest")
        if old is not None and old != manifest:          # never mix games of different settings
            raise SystemExit(f"resume refused: manifest differs from {out}.json")
    if a["resume"] and os.path.exists(out + ".jsonl"):
        res = [json.loads(line) for line in open(out + ".jsonl")]
        have = {(r["i"], r["seat"]) for r in res}
        gs = [g for g in gs if (g["i"], g["seat"]) not in have]
    t = time.time()
    n0 = len(res)
    with mp.Pool(a["workers"], init, (a,)) as pool, open(out + ".jsonl", "a" if n0 else "w") as f:
        for i, r in enumerate(pool.imap_unordered(job, gs), start=n0):
            res.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
            if (i + 1) % 50 == 0 or i + 1 == n0 + len(gs):
                s = summary(res)
                json.dump({"name": a["name"], "args": a, "manifest": manifest, "t": time.strftime("%Y-%m-%d %H:%M"), "done": i + 1 == n0 + len(gs),
                           "wall_secs": round(time.time() - t), **s}, open(out + ".json", "w"), indent=1)
                print(f"{a['name']}: {i + 1}/{n0 + len(gs)} score {100 * s['score']:.1f}% ci {s['ci95']} "
                      f"override {s['override_rate']:.2f}", flush=True)


if __name__ == "__main__":
    main()
