"""The frozen validation bank: held-out label positions regraded with many independent draws and every legal option as
a candidate. Student, controls, the incumbent and the teacher's own labels are scored against it (not against the
noisy draws they were made with). Also measures what the top-5 shortlist misses.

    python experiments/bank.py --n 2000 --k 128 [--workers 7]                               # run p1, dev split (old)
    python experiments/bank.py --run s1 --cohort <id> --split 1 --n 1000 --k 128 --n-hi 100 --k-hi 512 --name s1

Replay identity: a regraded row must be the recorded position -- the same features (exactly), decision kind and
incumbent, and the same legal options where the shard has them (else the recorded candidates must still be legal).
Mismatches are counted per shard and the bank fails on any (--allow-mismatch to only report them).
Out: data/bank/<name>.npz (X, Q, SE, CAND, POL, INC, TURN, K, CAPS, KIND, Q16, CAND16, SE16, OPP, SHARD, HI) +
results/<name>.json. The default name is "bank" (data/bank/bank.npz, results/bank.json, as before).
"""
import argparse, glob, json, multiprocessing as mp, os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FIELDS = ("X", "Q", "SE", "CAND", "POL", "INC", "TURN", "K", "CAPS")


def identity(g, row):
    """None if the regraded record g is the recorded row, else why not."""
    if g is None:
        return "decision not reached"
    if not np.array_equal(g["X"], row["X"]):
        return "features"
    if int(g["INC"]) != int(row["INC"]):
        return "incumbent"
    if "KIND" in g and int(g["KIND"]) != int(row.get("KIND", 1)):
        return "kind"
    legal = g["CAND"].astype(bool)                       # regraded with every legal option as a candidate
    if row.get("LEGAL") is not None and not np.array_equal(legal, row["LEGAL"].astype(bool)):
        return "legal mask"
    if not legal[row["CAND"].astype(bool)].all():
        return "candidate no longer legal"
    return None


def job(item):
    from bv import core
    key, didx, seed, k, kind, row = item
    g = core.regrade(key, didx, k=k, seed=seed, all_legal=True, kind=kind)
    why = identity(g, row)
    if why:
        return {"mismatch": why}
    return {x: g[x] for x in FIELDS + (("KIND",) if "KIND" in g else ())}


def rows_of(run, cohort, split, kinds):
    from bv import core
    rows, shards = [], {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", run, "*.npz"))):
        with np.load(f) as z:
            if "KING" not in z.files:
                continue                                   # old shards have no replay key
            meta = json.loads(str(z["META"]))
            if cohort and meta.get("cohort") != cohort:
                continue
            D = {x: z[x] for x in z.files if x != "META"}
        if meta.get("dagger") and not os.path.exists(core.student_file(meta.get("run", run), meta["student"])):
            continue                                   # a DAgger shard whose student was not kept: not replayable
        kind = D.get("KIND", np.full(len(D["K"]), core.BUY))
        for i in np.where((D["SPLIT"] == split) & np.isin(kind, kinds))[0]:
            row = {"X": D["X"][i], "INC": D["INC"][i], "CAND": D["CAND"][i], "KIND": int(kind[i]),
                   "LEGAL": D["LEGAL"][i] if "LEGAL" in D else None}
            rows.append((core.key_of(D, i, meta), int(D["DIDX"][i]), D["Q"][i], int(D["INC"][i]), D["CAND"][i],
                         int(kind[i]), row, os.path.basename(f), D["SE"][i], int(D["OPP"][i])))
        shards[os.path.basename(f)] = meta.get("cohort")
    return rows, shards


def teacher_at_margin(Q16, SE16, CAND16, inc, Qb, margin, z):
    """The label's own teacher in play (its 16..k-draw grades, the deployment margin and z gate) scored on the bank's
    independent values: per position the true gain of its choice over the incumbent."""
    n = len(inc)
    q = np.where(CAND16.astype(bool), np.nan_to_num(Q16, nan=-1), -1)
    b = q.argmax(1)
    gain16 = q[np.arange(n), b] - q[np.arange(n), inc]
    se = np.nan_to_num(SE16[np.arange(n), b], nan=0)
    pick = np.where((b != inc) & (gain16 > margin) & (gain16 > z * se), b, inc)
    return Qb[np.arange(n), pick] - Qb[np.arange(n), inc], pick != inc


def phase_of(turn):
    return np.where(turn <= 4, "opening", np.where(turn <= 12, "middle", "late"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="p1")
    ap.add_argument("--cohort", default="")
    ap.add_argument("--split", type=int, default=0, help="0 dev, 1 test (held out)")
    ap.add_argument("--kinds", default="buy", help="buy, action or buy,action")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--n-hi", type=int, default=0, help="of these, the first n-hi get k-hi draws (high-draw subset)")
    ap.add_argument("--k-hi", type=int, default=512)
    ap.add_argument("--margin", type=float, default=0.02, help="deployment margin for the teacher-at-margin score")
    ap.add_argument("--z", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--name", default="bank")
    ap.add_argument("--allow-mismatch", action="store_true")
    ap.add_argument("--decisive-share", type=float, default=0.0, help="share of positions whose label spread > 0.2")
    a = ap.parse_args()
    from bv import core
    kinds = [{"buy": core.BUY, "action": core.ACTION}[x] for x in a.kinds.split(",")]
    rows, shards = rows_of(a.run, a.cohort, a.split, kinds)
    rng = np.random.default_rng(7)
    if a.decisive_share > 0:                      # a share of decisive positions (label spread > 0.2), rest ordinary
        spread = np.array([np.nanmax(r[2]) - np.nanmin(r[2]) for r in rows])
        dec, ordi = rng.permutation(np.where(spread > 0.2)[0]), rng.permutation(np.where(spread <= 0.2)[0])
        nd = min(len(dec), int(round(a.n * a.decisive_share)))
        pick = rng.permutation(np.concatenate([dec[:nd], ordi[:a.n - nd]]).astype(int))
    else:
        pick = rng.permutation(len(rows))[:a.n]
    items = [(rows[i][0], rows[i][1], 1_000_003 + int(i), a.k_hi if j < a.n_hi else a.k, rows[i][5], rows[i][6])
             for j, i in enumerate(pick)]
    print(f"bank {a.name}: {len(items)} of {len(rows)} positions (run {a.run}, cohort {a.cohort or 'any'}, split "
          f"{a.split}, {a.kinds}), {a.k} draws ({a.n_hi} at {a.k_hi}), all legal options", flush=True)
    t = time.time()
    with mp.get_context("spawn").Pool(a.workers) as p:
        res = p.map(job, items, chunksize=2)
    mism = {}
    for r, i in zip(res, pick):
        if "mismatch" in r:
            sh = rows[i][7]
            mism.setdefault(sh, {}).setdefault(r["mismatch"], 0)
            mism[sh][r["mismatch"]] += 1
    if mism:
        print("replay mismatches by shard:", json.dumps(mism), flush=True)
        if not a.allow_mismatch:
            json.dump({"name": a.name, "failed": "replay mismatches", "mismatches": mism},
                      open(os.path.join(ROOT, "results", f"{a.name}.json"), "w"), indent=1)
            raise SystemExit(f"bank: {sum(sum(v.values()) for v in mism.values())} replay mismatches")
    ok = [(r, rows[i], j) for j, (r, i) in enumerate(zip(res, pick)) if "mismatch" not in r]
    B = {x: np.stack([r[x] for r, _, _ in ok]) for x in ok[0][0]}
    B["Q16"] = np.stack([row[2] for _, row, _ in ok])         # the original label of the same position
    B["CAND16"] = np.stack([row[4] for _, row, _ in ok])
    B["SE16"] = np.stack([row[8] for _, row, _ in ok])
    B["OPP"] = np.array([row[9] for _, row, _ in ok], np.int8)
    B["SHARD"] = np.array([row[7] for _, row, _ in ok])
    B["DECISIVE"] = np.array([np.nanmax(row[2]) - np.nanmin(row[2]) > 0.2 for _, row, _ in ok])
    B["HI"] = np.array([j < a.n_hi for _, _, j in ok])
    B["KEY"] = np.array([json.dumps({"key": row[0], "didx": row[1], "kind": row[5]}) for _, row, _ in ok])
    if "KIND" not in B:
        B["KIND"] = np.array([row[5] for _, row, _ in ok], np.int32)
    os.makedirs(os.path.join(ROOT, "data", "bank"), exist_ok=True)
    np.savez_compressed(os.path.join(ROOT, "data", "bank", f"{a.name}.npz"), **B)
    out = {"name": a.name, "run": a.run, "cohort": a.cohort, "split": a.split, "kinds": a.kinds, "k": a.k,
           "n_hi": a.n_hi, "k_hi": a.k_hi, "secs": round(time.time() - t), "mismatches": mism,
           "positions": len(B["Q"]), **score(B, a.margin, a.z)}
    json.dump(out, open(os.path.join(ROOT, "results", f"{a.name}.json"), "w"), indent=1)
    print(json.dumps(out)[:2000])


def score(B, margin, z):
    """Incumbent and shortlist regret, strict shortlist misses (the best legal value beats the best shortlisted value),
    and the label teacher at the deployment margin, overall and by phase / opponent / kind."""
    Q, inc = B["Q"], B["INC"]
    n = len(Q)
    best = np.nanmax(Q, 1)
    short_best = np.nanmax(np.where(B["CAND16"].astype(bool), Q, np.nan), 1)
    gain, over = teacher_at_margin(B["Q16"], B["SE16"], B["CAND16"], inc, Q, margin, z)

    def block(m):
        if not m.any():
            return None
        return {"n": int(m.sum()), "incumbent_regret": round(float(np.mean(best[m] - Q[m, inc[m]])), 4),
                "shortlist_regret": round(float(np.mean(best[m] - short_best[m])), 4),
                "strict_outside_shortlist": round(float(np.mean(best[m] - short_best[m] > 1e-6)), 4),
                "teacher_override_rate": round(float(over[m].mean()), 4),
                "teacher_mean_gain": round(float(gain[m].mean()), 4),
                "teacher_harmful": round(float(np.mean(over[m] & (gain[m] < -0.01))), 4),
                "teacher_good": round(float(np.mean(over[m] & (gain[m] > 0.01))), 4)}

    ph = phase_of(B["TURN"])
    opps = ("c2", "c1", "rl1", "ga2")
    return {"margin": margin, "z": z, "mean_legal": round(float(B["CAND"].sum(1).mean()), 2), "all": block(np.ones(n, bool)),
            "high_draw": block(B["HI"]), "decisive": block(B["DECISIVE"]), "ordinary": block(~B["DECISIVE"]),
            "by_phase": {p: block(ph == p) for p in ("opening", "middle", "late")},
            "by_opponent": {o: block(B["OPP"] == i) for i, o in enumerate(opps)},
            "by_kind": {k: block(B["KIND"] == v) for k, v in (("buy", 1), ("action", 0))}}


if __name__ == "__main__":
    main()
