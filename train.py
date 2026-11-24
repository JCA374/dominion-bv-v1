"""The curriculum (run p1): rounds of collect -> fit the student -> dev match, then DAgger, then the promotion match.

    python -u train.py --run p1 [--continue]

Stages (runs/<run>/curriculum.json): 1 labels (c2 plays, the teacher grades), 2 student fit (until the student on c2
beats plain c2 on dev), 3 DAgger (the student plays, the teacher grades), 4 promotion (2,000 reserved kingdoms vs c1
2300 and c2 2654; >= 53% vs both and both 95% intervals above 50%). Files: runs/<run>/{status.json, events.jsonl,
stats.csv, student.npy, student_<round>.npy}.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
CFG = {"pilot_states": 2000, "round_states": 20000, "workers": 7, "k0": 16, "k1": 16, "se": 1.5, "topn": 5,
       "epochs": 6, "batch": 512, "lr": 1e-3, "rank_w": 10.0, "value_w": 0.1, "dev_kingdoms": 300, "margins": [0.02, 0.05, 0.1],
       "dagger_after": 0.505, "promote_at": 0.53, "confirm_kingdoms": 2000, "ga_tolerance": 0.01, "max_rounds": 200}
STAGES = [
    {"id": 1, "name": "Labels", "what": "c2 plays every buy; the teacher grades it with paired rollouts to the end",
     "gate": "pilot (in code): capped and rejected rollouts < 1%, the test suite (reproducible + replayable labels) passes"},
    {"id": 2, "name": "Student fit", "what": "fit the Deep Sets student on all labels; dev match student-on-c2 vs c2",
     "gate": f"dev score >= {CFG['dagger_after']:.1%} at the best margin, confirmed on an independent dev stream"},
    {"id": 3, "name": "DAgger", "what": "the student plays its buys, the teacher grades them (positions the student reaches)",
     "gate": f"dev score >= {CFG['promote_at']:.0%}"},
    {"id": 4, "name": "Promotion", "what": "2,000 fresh confirmation kingdoms, both seats, vs c1 2300, c2 2654 and the GA base",
     "gate": ">= 53% vs both RL-v3, kingdom-block 95% interval above 50%, GA base no more than 1 point below plain c2's"},
]


class Run:
    def __init__(self, name):
        self.d = os.path.join(ROOT, "runs", name)
        os.makedirs(self.d, exist_ok=True)
        self.name = name
        p = os.path.join(self.d, "status.json")
        self.s = json.load(open(p)) if os.path.exists(p) else {"run": name, "round": 0, "stage": 1, "states": 0,
                                                                 "best_dev": None, "started": time.strftime("%Y-%m-%d %H:%M")}

    def save(self, **kw):
        self.s.update(kw, t=time.strftime("%Y-%m-%d %H:%M"))
        json.dump(self.s, open(os.path.join(self.d, "status.json"), "w"), indent=1)

    def event(self, kind, msg, **kw):
        e = {"t": time.strftime("%Y-%m-%d %H:%M"), "kind": kind, "msg": msg, "note": msg, **kw}
        open(os.path.join(self.d, "events.jsonl"), "a").write(json.dumps(e) + "\n")
        print(f"[{e['t']}] {kind}: {msg}", flush=True)

    def row(self, r):
        p = os.path.join(self.d, "stats.csv")
        new = not os.path.exists(p)
        with open(p, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(r))
            if new:
                w.writeheader()
            w.writerow(r)


def cohorts(run):
    """{cohort id: number of shards} of a run's label shards (shards before cohort ids: "legacy")."""
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", run, "*.npz"))):
        with np.load(f) as z:
            c = json.loads(str(z["META"])).get("cohort", "legacy")
        out[c] = out.get(c, 0) + 1
    return out


def load(run, cohort=None, kind=None):
    """All shards in one preallocated array per field (no second copy). CAND (the explicit candidate mask) comes from
    the shard when it has one, else from numpy's isnan(Q) (old shards; numpy keeps NaN checks). POL: the live prior
    recorded with the label when the shard has it (HAS_POL), else recomputed by policy_prior from the mask
    (tests/test_prior.py: identical to the live prior). cohort: only shards of that label cohort ("legacy" = shards
    without an id); None = all (labels of different cohorts are not comparable: compare them separately). kind: only
    rows of that decision kind (bv.core.BUY / ACTION; shards without KIND are buys)."""
    fs = sorted(glob.glob(os.path.join(ROOT, "data", run, "*.npz")))
    if cohort is not None:
        fs = [f for f in fs if json.loads(str(np.load(f)["META"])).get("cohort", "legacy") == cohort]
    if not fs:
        return None
    if kind is not None:                             # rows of one kind: filter shard by shard, then concatenate
        parts = []
        for f in fs:
            with np.load(f) as z:
                m = (z["KIND"] if "KIND" in z.files else np.full(len(z["K"]), 1)) == kind
                parts.append({k: z[k][m] for k in z.files if k != "META" and z[k].ndim and len(z[k]) == len(m)})
        keys = set.intersection(*(set(p) for p in parts))
        out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
        if "CAND" not in out:
            out["CAND"] = ~np.isnan(out["Q"])
        out["HAS_POL"] = np.ones(len(out["K"]), bool)
        return out
    sizes = []
    for f in fs:
        with np.load(f) as z:
            sizes.append(len(z["K"]))
    keys = ["X", "Q", "SE", "INC", "CHOSE", "K", "TURN", "OPP", "SPLIT", "KH", "SLICE", "RESULT"]
    with np.load(fs[0]) as z:
        out = {k: np.empty((sum(sizes),) + z[k].shape[1:], z[k].dtype) for k in keys}
        out["CAND"] = np.empty((sum(sizes), z["Q"].shape[1]), np.uint8)
        out["POL"] = np.zeros((sum(sizes), z["Q"].shape[1]), np.float32)
        out["HAS_POL"] = np.zeros(sum(sizes), bool)
    o = 0
    for f, n in zip(fs, sizes):
        with np.load(f) as z:
            for k in keys:
                out[k][o:o + n] = z[k]
            out["CAND"][o:o + n] = z["CAND"] if "CAND" in z.files else ~np.isnan(z["Q"])
            if "POL" in z.files:
                out["POL"][o:o + n] = z["POL"]
                out["HAS_POL"][o:o + n] = True
        o += n
    return out


def fit(D, init=None, log=print):
    import torch
    from bv.student import Student, export
    torch.set_num_threads(6)
    torch.manual_seed(0)
    m = Student()
    if init is not None:
        m.load_state_dict(init)
    from collect import split_of
    from match import eval_kingdoms
    bad = np.isin(D["KH"], [split_of(k.split(","))[0] for k in eval_kingdoms()])   # never train on evaluation kingdoms
    tr, dv = np.where((D["SPLIT"] == 2) & ~bad)[0], np.where((D["SPLIT"] == 0) & ~bad)[0]
    log(f"  dropped {int(bad.sum())} positions from evaluation kingdoms")
    from bv.core import policy_prior
    X = torch.from_numpy(D["X"])                     # shares memory with the numpy array
    mask = D["CAND"].astype(bool) if "CAND" in D else ~np.isnan(D["Q"])     # explicit mask; numpy isnan for old shards
    pol = policy_prior(D["X"], mask)
    if "HAS_POL" in D and D["HAS_POL"].any():        # the recorded live prior where the shard has it
        pol[D["HAS_POL"]] = D["POL"][D["HAS_POL"]]
    POL = torch.from_numpy(pol)
    Q = torch.from_numpy(np.nan_to_num(D["Q"], nan=0.0))
    M = torch.from_numpy(mask)
    SE = torch.from_numpy(np.nan_to_num(D["SE"], nan=1.0))
    INC = torch.from_numpy(D["INC"].astype(np.int64))
    opt = torch.optim.Adam(m.parameters(), lr=CFG["lr"])

    def loss_of(idx):
        z = m(X[idx], POL[idx])
        p = torch.sigmoid(z)
        mk = M[idx].float()
        bce = torch.nn.functional.binary_cross_entropy_with_logits(z, Q[idx], reduction="none")
        lv = (bce * mk).sum() / mk.sum()
        inc = INC[idx]
        pi = p.gather(1, inc[:, None])
        qi = Q[idx].gather(1, inc[:, None])
        w = 0.0025 / (SE[idx] ** 2 + 0.0025) * mk
        lr_ = (w * ((p - pi) - (Q[idx] - qi)) ** 2).sum() / mk.sum()
        return CFG["value_w"] * lv + CFG["rank_w"] * lr_, z

    for ep in range(CFG["epochs"]):
        perm = tr[np.random.default_rng(ep).permutation(len(tr))]
        m.train()
        tot = 0.0
        for i in range(0, len(perm), CFG["batch"]):
            l_, _ = loss_of(torch.tensor(perm[i:i + CFG["batch"]]))
            opt.zero_grad()
            l_.backward()
            opt.step()
            tot += float(l_.detach()) * min(CFG["batch"], len(perm) - i)
        log(f"  epoch {ep}: train loss {tot / max(len(perm), 1):.4f}")
    m.eval()
    with torch.no_grad():                            # in chunks: one pass over all dev states needs ~1 GB
        zs, ls = [], []
        for i in range(0, len(dv), 4096):
            idx = torch.tensor(dv[i:i + 4096])
            l_, z = loss_of(idx)
            zs.append(z.masked_fill(~M[idx], -1e9).numpy())
            ls.append(float(l_) * len(idx))
        z = np.concatenate(zs)
        l_ = torch.tensor(sum(ls) / max(len(dv), 1))
    q = D["Q"][dv]
    best = np.nanmax(q, 1)
    pick = z.argmax(1)
    reg = float(np.mean(best - q[np.arange(len(dv)), pick]))
    reg_inc = float(np.mean(best - q[np.arange(len(dv)), D["INC"][dv]]))
    agree = float(np.mean(pick == np.nanargmax(q, 1)))
    ev = {"dev_loss": round(float(l_), 4), "dev_regret": round(reg, 4), "dev_regret_incumbent": round(reg_inc, 4),
          "dev_top1": round(agree, 3), "train_states": int(len(tr)), "dev_states": int(len(dv))}
    ev.update(bank_eval(m, margin=CFG["margins"][0]))
    return m, export(m), ev


def bank_eval(m, model_fn=None, margin=0.02, name="bank"):
    """Regret on the frozen validation bank (independent 128-draw grades over all legal buys): the student ranks its own
    shortlist (the bank row's 16-draw candidates) and overrides the incumbent only above the deployment margin, as in
    play (sigmoid values, bv_student). Empty if there is no bank yet."""
    p = os.path.join(ROOT, "data", "bank", f"{name}.npz")
    if not os.path.exists(p):
        return {}
    import torch
    B = dict(np.load(p))
    from bv.core import policy_prior
    short = B["CAND16"].astype(bool)
    pol = policy_prior(B["X"], short)
    with torch.no_grad():
        z = (model_fn or m)(torch.from_numpy(B["X"]), torch.from_numpy(pol)).numpy()
    z[~short] = -1e9
    q = B["Q"]
    n = len(q)
    best = np.nanmax(q, 1)
    inc = B["INC"]
    sq = 1 / (1 + np.exp(-z))
    pick = sq.argmax(1)
    pick = np.where(sq[np.arange(n), pick] - sq[np.arange(n), inc] > margin, pick, inc)
    gain = q[np.arange(n), pick] - q[np.arange(n), inc]
    ov = pick != inc
    return {"bank_regret": round(float(np.mean(best - q[np.arange(n), pick])), 4),
            "bank_regret_incumbent": round(float(np.mean(best - q[np.arange(n), inc])), 4),
            "bank_harmful_override": round(float(np.mean(ov & (gain < -0.01))), 4),
            "bank_good_override": round(float(np.mean(ov & (gain > 0.01))), 4)}


def match(name, student, margin, stream, kingdoms, opp="c2", bv=3):
    cmd = [PY, "-u", os.path.join(ROOT, "match.py"), "--name", name, "--bv", str(bv), "--opp", opp,
           "--margin", str(margin), "--stream", stream, "--kingdoms", str(kingdoms), "--workers", str(CFG["workers"])]
    if student:
        cmd += ["--student", student]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return json.load(open(os.path.join(ROOT, "results", f"match_{name}.json")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="p1")
    ap.add_argument("--continue", dest="cont", action="store_true")
    ap.add_argument("--fit-first", action="store_true", help="first round: no new labels, refit on what exists")
    a = ap.parse_args()
    R = Run(a.run)
    json.dump(CFG, open(os.path.join(R.d, "config.json"), "w"), indent=1)
    json.dump({"note": "Distil a paired-rollout buy teacher into a Deep Sets student on top of RL-v3 c2.",
               "stages": [{"name": f"{x['id']} {x['name']}", "world": x["what"], "pass": x["gate"]} for x in STAGES]},
              open(os.path.join(R.d, "curriculum.json"), "w"), indent=1)
    from collect import collect
    import torch
    init = None
    ck = os.path.join(R.d, "student.pt")
    if a.cont and os.path.exists(ck):
        init = torch.load(ck)
        from bv.student import Student
        try:
            Student().load_state_dict(init)
        except RuntimeError:
            init = None                                  # the student changed shape: refit from scratch
    while R.s["round"] < CFG["max_rounds"]:
        rnd = R.s["round"] + 1
        n = CFG["pilot_states"] if rnd == 1 else CFG["round_states"]
        if a.fit_first:
            n, a.fit_first = 0, False
        dag = R.s["stage"] >= 3
        stud = os.path.join(R.d, "student.npy") if dag else ""
        R.event("collect", f"round {rnd}: collecting {n} states ({'DAgger: student plays' if dag else 'c2 plays'})")
        st = collect({"run": a.run, "round": rnd, "states": n, "workers": CFG["workers"], "k0": CFG["k0"], "k1": CFG["k1"],
                      "se": CFG["se"], "topn": CFG["topn"], "margin": R.s.get("margin", 0.0), "student": stud,
                      "dagger": dag})
        R.s["states"] += st["states"]
        cap = st["capped"] / max(st["rollouts"], 1)
        R.event("labels", f"round {rnd}: {st['states']} states in {st['secs']} s ({st['states'] / max(st['secs'], 1):.1f}/s), "
                f"{st['rollouts']} rollouts, capped {cap:.2%}, errors {st['errors']}")
        D = load(a.run)
        t = time.time()
        m, flat, ev = fit(D, init, log=print)
        init = m.state_dict()
        torch.save(init, ck)
        np.save(os.path.join(R.d, "student.npy"), flat)
        np.save(os.path.join(R.d, f"student_{rnd}.npy"), flat)
        R.event("fit", f"round {rnd}: {ev['train_states']} train states, dev regret {ev['dev_regret']:.4f} "
                f"(incumbent {ev['dev_regret_incumbent']:.4f}), top-1 {ev['dev_top1']:.0%}", secs=round(time.time() - t))
        sp = os.path.join(R.d, f"student_{rnd}.npy")
        dev = {}
        for mg in CFG["margins"]:
            r = match(f"{a.run}_r{rnd}_dev_m{mg}", sp, mg, "dev", CFG["dev_kingdoms"])
            dev[mg] = r
        bm = max(dev, key=lambda k: dev[k]["score"])
        b = dev[bm]
        R.event("dev", f"round {rnd}: student on c2 vs c2 " + ", ".join(
            f"m{k}: {100 * v['score']:.1f}% (override {v['override_rate']:.0%})" for k, v in dev.items()))
        row = {"round": rnd, "t": time.strftime("%Y-%m-%d %H:%M"), "stage": R.s["stage"], "states": R.s["states"],
               "states_per_s": round(st["states"] / max(st["secs"], 1), 2), "capped": round(cap, 4), **ev,
               **{f"dev_m{k}": v["score"] for k, v in dev.items()}, "best_margin": bm, "override": b["override_rate"]}
        R.row(row)
        R.save(round=rnd, margin=bm, last_dev=b["score"], last_ci=b["ci95"],
               best_dev=max(R.s["best_dev"] or 0, b["score"]))
        if R.s["stage"] == 1:                           # pilot gate: rates and reproducibility, in code
            rej = st.get("rejected", 0) / max(st["states"], 1)
            rep = subprocess.run([PY, "-m", "pytest", "-q", "tests"], cwd=ROOT, capture_output=True).returncode == 0
            if cap < 0.01 and rej < 0.01 and rep:
                R.save(stage=2)
                R.event("stage", f"stage 2: student fit (pilot: capped {cap:.2%}, rejected {rej:.2%}, tests passed)")
            else:
                R.event("stage", f"pilot gate failed: capped {cap:.2%}, rejected {rej:.2%}, tests {'ok' if rep else 'FAILED'}")
                raise SystemExit("pilot gate failed")
        if R.s["stage"] == 2 and b["score"] >= CFG["dagger_after"]:
            c = match(f"{a.run}_r{rnd}_dev2_m{bm}", sp, bm, "dev2", CFG["dev_kingdoms"])   # independent check
            R.event("dev2", f"round {rnd}: independent dev check at margin {bm}: {100 * c['score']:.1f}% {c['ci95']}")
            if c["score"] >= CFG["dagger_after"]:
                R.save(stage=3)
                R.event("stage", f"stage 3: DAgger (dev {100 * b['score']:.1f}%, dev2 {100 * c['score']:.1f}% at margin {bm})")
        if R.s["stage"] >= 3 and b["score"] >= CFG["promote_at"] and not R.s.get("promotion_tried_round") == rnd:
            c = match(f"{a.run}_r{rnd}_dev2p_m{bm}", sp, bm, "dev2", CFG["dev_kingdoms"])
            if c["score"] < CFG["promote_at"] - 0.01:
                R.event("promotion", f"round {rnd}: dev2 {100 * c['score']:.1f}% does not confirm dev; no promotion match")
                continue
            n = CFG["confirm_kingdoms"]
            R.event("promotion", f"round {rnd}: promotion match on {n} fresh confirmation kingdoms (vs c2, c1, GA base)")
            res = {o: match(f"{a.run}_r{rnd}_promo_{o}", sp, bm, "confirm", n, opp=o) for o in ("c2", "c1", "ga2")}
            base = match(f"c2_vs_ga2_confirm_{n}", "", 0.0, "confirm", n, opp="ga2", bv=0)   # plain c2 vs GA base
            ga_ok = res["ga2"]["score"] >= base["score"] - CFG["ga_tolerance"]
            ok = ga_ok and all(res[o]["score"] >= 0.53 and res[o]["ci95"][0] > 0.5 for o in ("c2", "c1"))
            R.save(promotion_tried_round=rnd, promotion={o: {"score": v["score"], "ci95": v["ci95"],
                                                             "latency": v.get("latency_s")} for o, v in res.items()})
            R.event("promotion", f"round {rnd}: vs c2 {100 * res['c2']['score']:.1f}% {res['c2']['ci95']}, vs c1 "
                    f"{100 * res['c1']['score']:.1f}% {res['c1']['ci95']}, vs GA base {100 * res['ga2']['score']:.1f}% "
                    f"(plain c2 {100 * base['score']:.1f}%), latency {res['c2'].get('latency_s')} -> "
                    f"{'PROMOTED' if ok else 'not yet'}")
            if ok:
                R.save(stage=4, promoted_round=rnd)
                np.save(os.path.join(R.d, "promoted.npy"), flat)

if __name__ == "__main__":
    main()
