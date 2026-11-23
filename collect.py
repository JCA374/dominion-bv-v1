"""Collect teacher labels: c2 plays (or the student on c2, DAgger) against an opponent drawn from a fixed mixture; the
teacher grades every buy of the c2 seat by paired rollouts. Kingdoms: uniform random plus attack / Gardens / trasher /
engine slices. Split by canonical kingdom hash: 0 dev, 1 test, else train.

    python collect.py --run p1 --states 20000 [--student runs/p1/student.npy --dagger]
Out: data/<run>/shard_<time>_<worker>.npz (X, Q, SE, INC, CHOSE, K, TURN, OPP, SPLIT, KH)
"""
from __future__ import annotations

import argparse
import json
import hashlib
import multiprocessing as mp
import os
import random
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OPPS = ["c2", "c1", "rl1", "ga2"]
SLICES = {"random": [], "attack": ["Witch", "Militia", "Bandit", "Bureaucrat"], "gardens": ["Gardens"],
          "trasher": ["Chapel", "Remodel", "Moneylender", "Sentry"],
          "engine": ["Village", "Festival", "Throne Room"]}
SLICE_W = [0.5, 0.15, 0.1, 0.15, 0.1]


def split_of(kingdom):
    h = int(hashlib.sha1(",".join(sorted(kingdom)).encode()).hexdigest()[:8], 16)
    return h, (0 if h % 10 == 0 else 1 if h % 10 == 1 else 2)


def meta(a):
    """Settings and model hashes stored with every shard (labels are only comparable under the same ones)."""
    from bv import core
    return {"k0": a["k0"], "k1": a["k1"], "se": a["se"], "topn": a["topn"], "margin": a["margin"],
            "dagger": a["dagger"], "student": a["student"] and core.hashlib.sha256(open(a["student"], "rb").read()).hexdigest()[:16],
            "models": {m: core.sha(m) for m in ("rl3_c2_2654.npy", "rl3_c1_2300.npy", "rl1_r2_2200.npy")},
            "opps": OPPS, "slices": list(SLICES), "round": a.get("round", 0), "run": a["run"],
            "sampler": "known-top"}


def save_shard(out, name, cat, info):
    """Atomic: write a hidden temp file, then rename (a crash never leaves half a shard)."""
    tmp = os.path.join(out, f".tmp_{name}")
    np.savez_compressed(tmp, META=np.array(json.dumps(info)), **cat)
    os.replace(tmp + ".npz", os.path.join(out, name + ".npz"))


def worker(a, wid, quota, q):
    from bv import core
    from match import eval_kingdoms
    st = np.load(a["student"]).astype(np.float32) if a["student"] else None
    info = meta(a)
    if a["student"]:                                  # keep the exact student: DAgger labels replay with it
        f = core.student_file(a["run"], info["student"])
        if not os.path.exists(f):
            os.makedirs(os.path.dirname(f), exist_ok=True)
            tmp = f"{f}.{wid}.tmp.npy"
            np.save(tmp, np.load(a["student"]))
            os.replace(tmp, f)
    core.configure(a["k0"], a["k1"], a["se"], a["topn"], a["margin"], st)
    banned = eval_kingdoms()
    rng = random.Random(f"bv collect {a['run']} round {a.get('round', 0)} worker {wid}")   # deterministic: replayable
    out = os.path.join(ROOT, "data", a["run"])
    os.makedirs(out, exist_ok=True)
    buf, n, part = [], 0, 0
    while n < quota:
        sl = rng.choices(list(SLICES), SLICE_W)[0]
        must = [rng.choice(SLICES[sl])] if SLICES[sl] else []
        k = core.random_kingdom(rng, must)
        if ",".join(sorted(k)) in banned:
            continue
        kh, sp = split_of(k)
        opp = rng.choice(OPPS)
        s = rng.randrange(2)
        me = core.seat_for("c2", k, 4 if a["dagger"] else 2)
        op = core.seat_for(opp, k)
        seed = rng.getrandbits(32)
        r = core.play([me, op] if s == 0 else [op, me], k, seed, seed ^ 0x5EED)
        rec = r["rec"]
        m = len(rec["K"])
        if m:
            buf.append({**{x: rec[x] for x in ("X", "Q", "SE", "INC", "CHOSE", "K", "TURN", "CAND", "POL", "DIDX", "CAPS")},
                        "SEED": np.full(m, seed, np.int64), "SEAT0": np.full(m, s, np.int8),
                        "KING": np.full(m, ",".join(k)),
                        "OPP": np.full(m, OPPS.index(opp), np.int8), "SPLIT": np.full(m, sp, np.int8),
                        "KH": np.full(m, kh, np.int64), "SLICE": np.full(m, list(SLICES).index(sl), np.int8),
                        "RESULT": np.full(m, core.points(r, s), np.float32)})
            n += m
        q.put(("game", m, core.stats()))
        if buf and (sum(len(b["K"]) for b in buf) >= 500 or n >= quota):
            cat = {x: np.concatenate([b[x] for b in buf]) for x in buf[0]}
            save_shard(out, f"shard_r{a.get('round', 0):03d}_w{wid}_{part:03d}", cat, info)
            part += 1
            buf = []
    st_ = core.stats()
    st_["rejected"] = int(core.lib.bv_rejected())
    q.put(("done", wid, st_))


def collect(a):
    """Runs workers until a['states'] labels exist; returns rollout stats."""
    ctx = mp.get_context("spawn")             # clean workers: not a copy of the trainer's data and torch
    q = ctx.Queue()
    if a["states"] <= 0:
        return {"rollouts": 0, "capped": 0, "errors": 0, "graded": 0, "overrides": 0, "buys": 0, "states": 0,
                "games": 0, "secs": 0, "rejected": 0}
    per = -(-a["states"] // a["workers"])
    ps = [ctx.Process(target=worker, args=(a, w, per, q)) for w in range(a["workers"])]
    for p in ps:
        p.start()
    done, states, games, t = 0, 0, 0, time.time()
    last = {}
    import queue as _queue
    while done < len(ps):
        try:
            kind, x, st = q.get(timeout=120)
        except _queue.Empty:
            dead = [p for p in ps if p.exitcode not in (None, 0)]
            if dead:                                    # a worker crashed: stop clearly, never hang
                for p in ps:
                    p.terminate()
                raise RuntimeError(f"collect: worker exit codes {[p.exitcode for p in ps]}")
            continue
        if kind == "done":
            done += 1
            last[x] = st
        else:
            states += x
            games += 1
            if games % 50 == 0:
                print(f"collect {a['run']}: {states} states, {games} games, {states / (time.time() - t):.1f} states/s",
                      flush=True)
    for p in ps:
        p.join()
    tot = {k: sum(s[k] for s in last.values()) for k in next(iter(last.values()))}
    tot.update(states=states, games=games, secs=round(time.time() - t))
    return tot


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--states", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--k0", type=int, default=16)
    ap.add_argument("--k1", type=int, default=64)
    ap.add_argument("--se", type=float, default=1.5)
    ap.add_argument("--topn", type=int, default=5)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--student", default="")
    ap.add_argument("--dagger", action="store_true")
    print(collect(vars(ap.parse_args())))
