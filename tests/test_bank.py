"""The bank's replay identity check: a regraded row must be the recorded position (features, kind, incumbent, legal
options). Holds for a position with known top cards (Sentry / Harbinger / Artisan), for an action decision, and for a
position after a student's override (DAgger); a wrong replay key is caught. Collection skips errored source games and
stamps a cohort id."""
import importlib.util
import os
import random
import shutil

import numpy as np

from bv import core

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("bank", os.path.join(ROOT, "experiments", "bank.py"))
bank = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bank)


def _row(rec, i):
    return {"X": rec["X"][i], "INC": rec["INC"][i], "CAND": rec["CAND"][i], "KIND": int(rec["KIND"][i]),
            "LEGAL": rec["LEGAL"][i]}


def test_known_top_and_action_rows_replay_identically():
    k = core.random_kingdom(random.Random(9), ["Sentry", "Harbinger", "Artisan"])
    core.configure(k0=2, k1=2, topn=3, act=1)
    try:
        found = {"known": 0, "action": 0}
        for seed in range(300, 306):
            key = {"kingdom": k, "seed": seed, "seat": 0, "opp": "c1", "act": 1, "topn": 3}
            seats, kg = core.replay_game(key)
            rec = core.play(seats, kg, seed, seed ^ 0x5EED)["rec"]
            for want, rows in (("known", np.where((rec["KN"] > 0) & (rec["KIND"] == core.BUY))[0]),
                               ("action", np.where(rec["KIND"] == core.ACTION)[0])):
                for i in rows[:2]:
                    g = core.regrade(key, int(rec["DIDX"][i]), k=2, seed=5, kind=int(rec["KIND"][i]))
                    assert bank.identity(g, _row(rec, i)) is None
                    found[want] += 1
            bad = core.regrade(dict(key, seed=seed + 1000), int(rec["DIDX"][len(rec["DIDX"]) // 2]), k=2, seed=5,
                               kind=int(rec["KIND"][len(rec["DIDX"]) // 2]))
            assert bank.identity(bad, _row(rec, len(rec["DIDX"]) // 2)) is not None
        assert found["known"] >= 2 and found["action"] >= 4
    finally:
        core.configure(act=0)


def test_position_after_a_student_override_replays(tmp_path):
    from bv.student import Student, export
    import torch
    torch.manual_seed(3)
    f = tmp_path / "st.npy"
    np.save(f, export(Student()))
    st = np.load(f).astype(np.float32)
    core.configure(k0=2, k1=2, topn=3, margin=-1.0, student=st, act=0)
    k = core.random_kingdom(random.Random(5))
    r = core.play([core.seat_for("c2", k, 4), core.seat_for("c1", k)], k, 777, 777 ^ 0x5EED)["rec"]
    over = np.where(r["CHOSE"] != r["INC"])[0]
    i = int(over[0]) + 3
    key = {"kingdom": k, "seed": 777, "seat": 0, "opp": "c1", "topn": 3, "margin": -1.0, "student": str(f)}
    g = core.regrade(key, int(r["DIDX"][i]), k=2, seed=1)
    assert bank.identity(g, _row(r, i)) is None
    plain = core.regrade({"kingdom": k, "seed": 777, "seat": 0, "opp": "c1"}, int(r["DIDX"][i]), k=2, seed=1)
    assert bank.identity(plain, _row(r, i)) is not None                # without the student: another position
    core.configure(k0=2, k1=2, topn=3)


def test_collect_stamps_a_cohort_and_skips_errored_games():
    import collect
    run = "_test_cohort"
    out = os.path.join(ROOT, "data", run)
    try:
        st = collect.collect({"run": run, "states": 40, "workers": 1, "k0": 2, "k1": 2, "se": 1.5, "topn": 3,
                              "margin": 0.0, "student": "", "dagger": False, "act": 1, "kval": 2})
        assert st["states"] >= 40 and st["skipped_error"] == 0
        import glob
        import json
        fs = glob.glob(os.path.join(out, "*.npz"))
        metas = [json.loads(str(np.load(x)["META"])) for x in fs]
        assert len({m["cohort"] for m in metas}) == 1 and metas[0]["act"] == 1
        z = np.load(fs[0])
        assert {"LEGAL", "KIND", "VG", "VSE", "VK", "KN"} <= set(z.files)
        assert (z["KIND"] == core.ACTION).any() and (z["KIND"] == core.BUY).any()
        a = collect.meta({"run": "x", "k0": 2, "k1": 2, "se": 1.5, "topn": 3, "margin": 0.0, "student": "",
                          "dagger": False, "act": 1, "kval": 2})
        b = dict(a)
        c = collect.meta({"run": "y", "k0": 2, "k1": 4, "se": 1.5, "topn": 3, "margin": 0.0, "student": "",
                          "dagger": False, "act": 1, "kval": 2})
        assert a["cohort"] == b["cohort"] == metas[0]["cohort"] and c["cohort"] != a["cohort"]
    finally:
        shutil.rmtree(out, ignore_errors=True)
