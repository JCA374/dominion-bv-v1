"""Labels are replayable: a recorded label's key (kingdom, seed, seat, opponent) and decision index reproduce the same
position exactly, and the observation-equivalence self-test holds (same observation -> identical grades, a different
observation -> different grades)."""
import random

import numpy as np

from bv import core


def test_regrade_reproduces_the_position():
    core.configure(k0=4, k1=4, topn=3)
    k = core.random_kingdom(random.Random(21))
    key = {"kingdom": k, "seed": 4242, "seat": 1, "opp": "c1"}
    seats, kg = core.replay_game(key)
    r = core.play(seats, kg, key["seed"], key["seed"] ^ 0x5EED)["rec"]
    assert len(r["DIDX"]) > 5
    i = len(r["DIDX"]) // 2
    g = core.regrade(key, int(r["DIDX"][i]), k=8, seed=99, all_legal=True)
    assert g is not None
    assert np.array_equal(g["X"], r["X"][i]) and g["INC"] == r["INC"][i] and g["K"] == 8
    assert g["CAND"].sum() >= r["CAND"][i].sum()          # all legal buys >= the shortlist


def test_observation_equivalence():
    hits = 0
    for seed in (1, 2, 3, 4):
        res = core.canon_test(seed)
        if res is None:
            continue
        same, diff = res
        assert same == 0.0
        hits += diff > 0
    assert hits >= 2
