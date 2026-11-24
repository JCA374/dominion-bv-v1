"""bv_advise: the teacher on a position given from outside a game (the Steam advisor), with a time cap."""
import random

import numpy as np

from bv import core

K = ["Village", "Smithy", "Market", "Militia", "Moat", "Cellar", "Chapel", "Festival", "Laboratory", "Witch"]
ID = core.ID
C2 = core.params("rl3_c2_2654.npy")


def supply():
    s = [-1] * core.NUM_CARDS
    for c, n in {"Copper": 46, "Silver": 40, "Gold": 30, "Estate": 8, "Duchy": 8, "Province": 8, "Curse": 10}.items():
        s[ID[c]] = n
    for c in K:
        s[ID[c]] = 10
    return s


def ids(*names):
    return [ID[n] for n in names]


START = ids(*["Copper"] * 7, *["Estate"] * 3)


def turn1(**kw):
    """My first turn, buy phase with $5 (5 Coppers in play); the opponent has its starting 10, all unseen."""
    zones = [ids("Copper", "Copper", "Estate", "Estate", "Estate"), [], [], ids(*["Copper"] * 5),
             START[:5], START[5:], [], []]
    return core.advise(C2, supply(), 0, zones, [0, 0], core.BUY, coins=5, **kw)


def test_buy_candidates_and_grades():
    core.configure(k0=8, k1=8, topn=5)
    r = turn1(secs=30)
    assert r["draws"] == 8
    legal = {i for i, c in enumerate(supply()) if c > 0 and core.lib and i < core.NUM_CARDS}
    cand = set(np.where(r["cand"])[0])
    assert r["inc"] in cand and core.PASS in cand
    assert all(a == core.PASS or a in legal for a in cand)
    assert all(0 <= r["q"][a] <= 1 for a in cand)
    assert np.isnan(r["q"][~r["cand"]]).all()


def test_same_seed_same_grades():
    core.configure(k0=8, k1=8, topn=5)
    a, b = turn1(secs=30, seed=7), turn1(secs=30, seed=7)
    assert np.array_equal(a["q"], b["q"], equal_nan=True)


def test_time_cap():
    core.configure(k0=256, k1=256, topn=5)
    r = turn1(secs=0.2)
    assert 1 <= r["draws"] < 256
    assert r["ms"] < 600                                  # the cap, plus at most one draw over it


def test_inc_is_the_callers_pick():
    core.configure(k0=4, k1=4, topn=2)
    r = turn1(secs=30, inc=ID["Moat"])
    assert r["inc"] == ID["Moat"] and r["cand"][ID["Moat"]]
    assert turn1(secs=30, inc=ID["Province"])["inc"] == r["net"]   # not legal at $5: the net's pick


def test_opponent_zones_pooled():
    """The public sampler: only the opponent's collection and zone sizes matter, not which card is in which zone."""
    core.configure(k0=8, k1=8, topn=3)
    me = [ids("Copper", "Copper", "Estate", "Estate", "Estate"), [], [], ids(*["Copper"] * 5)]
    a = core.advise(C2, supply(), 0, me + [START[:5], START[5:], [], []], [0, 0], core.BUY, coins=5, seed=3)
    b = core.advise(C2, supply(), 0, me + [START[::-1][:5], START[::-1][5:], [], []], [0, 0], core.BUY,
                    coins=5, seed=3)
    assert np.array_equal(a["q"], b["q"], equal_nan=True)


def test_actions_and_one_option():
    core.configure(k0=4, k1=4, topn=5)
    hand = ids("Village", "Smithy", "Copper", "Copper", "Estate")
    deck = [ID["Copper"]] * 5 + [ID["Estate"]] * 2
    zones = [deck, hand, [], [], START[:5], START[5:], [], []]
    r = core.advise(C2, supply(), 1, zones, [2, 3], core.ACTION, actions=1)
    assert set(np.where(r["cand"])[0]) == {ID["Village"], ID["Smithy"], core.PASS}
    none = [deck, ids("Copper", "Copper", "Estate", "Copper", "Copper"), [], [], START[:5], START[5:], [], []]
    assert core.advise(C2, supply(), 1, none, [2, 3], core.ACTION)["draws"] == -2


def test_main_game_unchanged_by_the_cap():
    """No deadline outside bv_advise: a graded game is the same before and after an advise call."""
    k = core.random_kingdom(random.Random(5))
    core.configure(k0=4, k1=4, topn=3, act=0)
    play = lambda: core.play([core.seat_for("c2", k, 1), core.seat_for("c2", k)], k, 11, 12)  # noqa: E731
    a = play()
    turn1(secs=0.01)
    b = play()
    assert a["scores"] == b["scores"] and np.array_equal(a["rec"]["Q"], b["rec"]["Q"], equal_nan=True)
