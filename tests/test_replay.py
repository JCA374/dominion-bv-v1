"""Labels are replayable: a recorded label's key (kingdom, seed, seat, opponent) and decision index reproduce the same
position exactly (also for DAgger labels, which replay with the student that played them), and the
observation-equivalence self-test holds (same observation -> identical grades, a different observation -> different
grades), also with known top cards (Sentry/Harbinger/Artisan/Bureaucrat), which the sampler must keep in place."""
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
        same, diff, same_known, diff_known = res
        assert same == 0.0 and same_known == 0.0
        hits += diff > 0 and diff_known > 0
    assert hits >= 2


def test_dagger_label_replays_with_its_student(tmp_path):
    """A student that overrides buys changes the game: replaying with plain c2 reaches other positions, replaying with
    the stored student reaches the recorded one."""
    from bv.student import Student
    import torch
    torch.manual_seed(3)
    f = tmp_path / "st.npy"
    from bv.student import export
    np.save(f, export(Student()))
    st = np.load(f).astype(np.float32)
    core.configure(k0=4, k1=4, topn=3, margin=-1.0, student=st)     # margin -1: the student always takes its pick
    k = core.random_kingdom(random.Random(5))
    seats = [core.seat_for("c2", k, 4), core.seat_for("c1", k)]
    r = core.play(seats, k, 777, 777 ^ 0x5EED)["rec"]
    over = np.where(r["CHOSE"] != r["INC"])[0]
    assert len(over) and over[0] + 3 < len(r["DIDX"])
    i = int(over[0]) + 3                                             # a position after an override
    key = {"kingdom": k, "seed": 777, "seat": 0, "opp": "c1"}
    plain = core.regrade(key, int(r["DIDX"][i]), k=4, seed=1)
    key_d = dict(key, topn=3, margin=-1.0, student=str(f))
    g = core.regrade(key_d, int(r["DIDX"][i]), k=4, seed=1)
    assert np.array_equal(g["X"], r["X"][i])
    assert plain is None or not np.array_equal(plain["X"], r["X"][i])


def test_known_top_cards_are_tracked():
    """In kingdoms with Sentry, Harbinger, Artisan and Bureaucrat, some graded buys see known top cards."""
    core.configure(k0=2, k1=2, topn=2)
    k = core.random_kingdom(random.Random(9), ["Sentry", "Harbinger", "Artisan", "Bureaucrat"])
    n0 = core.known_stats()
    for s in range(6):
        core.play([core.seat_for("c2", k, 2), core.seat_for("c2", k, 2)], k, 100 + s, 5 + s)
    n1 = core.known_stats()
    assert n1[0] - n0[0] > 50 and n1[1] - n0[1] > 0
