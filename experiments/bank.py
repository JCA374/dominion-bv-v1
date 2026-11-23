"""The frozen validation bank: a fixed set of dev-split label positions regraded with many independent draws and all
legal buys as candidates. Student, controls and the incumbent are scored against it (not against the noisy draws they
trained on). Also measures what the top-5 shortlist misses.

    python experiments/bank.py --n 2000 --k 128 [--workers 7]
Out: data/bank/bank.npz (X, Q, SE, CAND, POL, INC, TURN, coins, buys) + results/bank.json.
"""
import argparse, glob, json, multiprocessing as mp, os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def job(item):
    from bv import core
    key, didx, seed, k = item
    g = core.regrade(key, didx, k=k, seed=seed, all_legal=True)
    return None if g is None else {x: g[x] for x in ("X", "Q", "SE", "CAND", "POL", "INC", "TURN", "K", "CAPS")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    from bv import core
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "p1", "*.npz"))):
        with np.load(f) as z:
            if "KING" not in z.files:
                continue                                   # old shards have no replay key
            D = {x: z[x] for x in ("KING", "SEED", "SEAT0", "OPP", "DIDX", "SPLIT", "Q", "INC", "CAND")}
            meta = json.loads(str(z["META"]))
        if meta.get("dagger") and not os.path.exists(core.student_file(meta.get("run", "p1"), meta["student"])):
            continue                                   # a DAgger shard whose student was not kept: not replayable
        for i in np.where(D["SPLIT"] == 0)[0]:
            rows.append((core.key_of(D, i, meta), int(D["DIDX"][i]), D["Q"][i], int(D["INC"][i]), D["CAND"][i]))
    rng = np.random.default_rng(7)
    pick = rng.permutation(len(rows))[:a.n]
    items = [(rows[i][0], rows[i][1], 1_000_003 + int(i), a.k) for i in pick]
    print(f"bank: {len(items)} of {len(rows)} dev positions with replay keys, {a.k} draws, all legal buys", flush=True)
    t = time.time()
    with mp.get_context("spawn").Pool(a.workers) as p:
        res = p.map(job, items, chunksize=4)
    ok = [(r, rows[i]) for r, i in zip(res, pick) if r is not None]
    B = {x: np.stack([r[x] for r, _ in ok]) for x in ok[0][0]}
    B["Q16"] = np.stack([row[2] for _, row in ok])         # the original 16-draw label of the same position
    B["CAND16"] = np.stack([row[4] for _, row in ok])
    os.makedirs(os.path.join(ROOT, "data", "bank"), exist_ok=True)
    np.savez_compressed(os.path.join(ROOT, "data", "bank", "bank.npz"), **B)
    Q, inc = B["Q"], B["INC"]
    n = len(Q)
    best = np.nanmax(Q, 1)
    short = np.where(B["CAND16"].astype(bool), Q, np.nan)
    out = {"positions": n, "k": a.k, "secs": round(time.time() - t),
           "incumbent_regret": round(float(np.mean(best - Q[np.arange(n), inc])), 4),
           "shortlist_regret": round(float(np.mean(best - np.nanmax(short, 1))), 4),
           "best_outside_shortlist": round(float(np.mean(np.nanargmax(Q, 1) != np.nanargmax(np.where(np.isnan(short), -1, short), 1))), 4),
           "mean_legal": round(float(B["CAND"].sum(1).mean()), 2)}
    json.dump(out, open(os.path.join(ROOT, "results", "bank.json"), "w"), indent=1)
    print(out)


if __name__ == "__main__":
    main()
