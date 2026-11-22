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
CFG = {"pilot_states": 2000, "round_states": 20000, "workers": 5, "k0": 16, "k1": 16, "se": 1.5, "topn": 5,
       "epochs": 6, "batch": 512, "lr": 1e-3, "rank_w": 10.0, "dev_kingdoms": 300, "margins": [0.02, 0.05, 0.1],
       "dagger_after": 0.505, "promote_at": 0.53, "reserved_kingdoms": 2000, "max_rounds": 200}
STAGES = [
    {"id": 1, "name": "Labels", "what": "c2 plays every buy; the teacher grades it with paired rollouts to the end",
     "gate": "pilot: cap/error rate < 1%, labels reproducible"},
    {"id": 2, "name": "Student fit", "what": "fit the Deep Sets student on all labels; dev match student-on-c2 vs c2",
     "gate": f"dev score >= {CFG['dagger_after']:.1%} at the best margin"},
    {"id": 3, "name": "DAgger", "what": "the student plays its buys, the teacher grades them (positions the student reaches)",
     "gate": f"dev score >= {CFG['promote_at']:.0%}"},
    {"id": 4, "name": "Promotion", "what": "2,000 reserved kingdoms, both seats, vs c1 2300 and c2 2654",
     "gate": ">= 53% vs both, kingdom-block 95% interval above 50%"},
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
        e = {"t": time.strftime("%Y-%m-%d %H:%M"), "kind": kind, "msg": msg, **kw}
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


def load(run):
    fs = sorted(glob.glob(os.path.join(ROOT, "data", run, "*.npz")))
    parts = [dict(np.load(f)) for f in fs]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} if parts else None


def fit(D, init=None, log=print):
    import torch
    from bv.student import Student, export
    torch.set_num_threads(6)
    torch.manual_seed(0)
    m = Student()
    if init is not None:
        m.load_state_dict(init)
    tr, dv = np.where(D["SPLIT"] == 2)[0], np.where(D["SPLIT"] == 0)[0]
    from bv.core import policy_prior
    X = torch.tensor(D["X"])
    POL = torch.tensor(policy_prior(D["X"], D["Q"]))
    Q = torch.tensor(np.nan_to_num(D["Q"], nan=0.0))
    M = torch.tensor(~np.isnan(D["Q"]))
    SE = torch.tensor(np.nan_to_num(D["SE"], nan=1.0))
    INC = torch.tensor(D["INC"].astype(np.int64))
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
        return lv + CFG["rank_w"] * lr_, z

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
    with torch.no_grad():
        idx = torch.tensor(dv)
        l_, z = loss_of(idx)
        z = z.masked_fill(~M[idx], -1e9).numpy()
    q = D["Q"][dv]
    best = np.nanmax(q, 1)
    pick = z.argmax(1)
    reg = float(np.mean(best - q[np.arange(len(dv)), pick]))
    reg_inc = float(np.mean(best - q[np.arange(len(dv)), D["INC"][dv]]))
    agree = float(np.mean(pick == np.nanargmax(q, 1)))
    return m, export(m), {"dev_loss": round(float(l_), 4), "dev_regret": round(reg, 4),
                          "dev_regret_incumbent": round(reg_inc, 4), "dev_top1": round(agree, 3),
                          "train_states": int(len(tr)), "dev_states": int(len(dv))}


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
    json.dump(STAGES, open(os.path.join(R.d, "curriculum.json"), "w"), indent=1)
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
        st = collect({"run": a.run, "states": n, "workers": CFG["workers"], "k0": CFG["k0"], "k1": CFG["k1"],
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
        if R.s["stage"] == 1:
            R.save(stage=2)
            R.event("stage", "stage 2: student fit")
        if R.s["stage"] == 2 and b["score"] >= CFG["dagger_after"]:
            R.save(stage=3)
            R.event("stage", f"stage 3: DAgger (dev {100 * b['score']:.1f}% at margin {bm})")
        if R.s["stage"] >= 3 and b["score"] >= CFG["promote_at"] and not R.s.get("promotion_tried_round") == rnd:
            R.event("promotion", f"round {rnd}: promotion match on {CFG['reserved_kingdoms']} reserved kingdoms")
            res = {o: match(f"{a.run}_r{rnd}_promo_{o}", sp, bm, "reserved", CFG["reserved_kingdoms"], opp=o)
                   for o in ("c2", "c1")}
            ok = all(v["score"] >= 0.53 and v["ci95"][0] > 0.5 for v in res.values())
            R.save(promotion_tried_round=rnd, promotion={o: {"score": v["score"], "ci95": v["ci95"]} for o, v in res.items()})
            R.event("promotion", f"round {rnd}: vs c2 {100 * res['c2']['score']:.1f}% {res['c2']['ci95']}, vs c1 "
                    f"{100 * res['c1']['score']:.1f}% {res['c1']['ci95']} -> {'PROMOTED' if ok else 'not yet'}")
            if ok:
                R.save(stage=4, promoted_round=rnd)
                np.save(os.path.join(R.d, "promoted.npy"), flat)


if __name__ == "__main__":
    main()
