"""Teacher and label fixes from the c2 search-training handoff (2026-11-23):

- action label-only mode: c2 keeps control, every graded action decision is a full row (features, legal and graded
  masks, incumbent, grades and paired SE per candidate, draws, caps, kind and ordinal), PASS included;
- one rollout-failure policy for buys and actions: an engine error in any rollout -> no label and no override;
- a tie with zero observed SE is unresolved: the adaptive grader draws on;
- validation: the graded winner is checked against the incumbent on independent draws before it is taught;
- the public sampler (only what a player can reconstruct about the opponent) and public top cards;
- one hidden-state preparation per draw gives the same grades as one per candidate;
- the label buffer reports overflow.
"""
import random

import numpy as np
import pytest

from bv import core

K = core.random_kingdom(random.Random(5))


def _game(bv, act, opp="c2", seed=11, **cfg):
    core.configure(**{"k0": 4, "k1": 4, "topn": 3, "act": act, **cfg})
    try:
        return core.play([core.seat_for("c2", K, bv), core.seat_for(opp, K)], K, seed, seed + 1)
    finally:
        core.lib.bv_set_fault(0)
        core.configure(act=0)


def test_action_labels_only_mode_keeps_c2_in_control():
    plain = _game(0, 0)
    lab = _game(2, 1)
    assert plain["scores"] == lab["scores"] and plain["turns"] == lab["turns"]
    r = lab["rec"]
    act = r["KIND"] == core.ACTION
    assert act.sum() > 10 and (~act).sum() > 10
    assert (r["CHOSE"] == r["INC"]).all()                                # labels only: the incumbent plays
    a = np.where(act)[0]
    assert (r["CAND"][a] == r["LEGAL"][a]).all()                        # every legal action and PASS is graded
    assert r["CAND"][a, core.PASS].all()
    assert (r["DIDX"][a] == np.arange(len(a))).all()                    # ordinal within the kind
    for i in a:
        q = r["Q"][i]
        assert not np.isnan(q[r["INC"][i]]) and np.isnan(q[~r["CAND"][i].astype(bool)]).all()
        assert r["K"][i] == 4 and r["LEGAL"][i, r["INC"][i]]
    best = np.array([np.nanargmax(r["Q"][i]) for i in a])
    assert (best == r["INC"][a]).any()                                  # non-overrides are rows too


def test_action_labels_replay():
    lab = _game(2, 1, opp="c1", seed=31)
    r = lab["rec"]
    a = np.where(r["KIND"] == core.ACTION)[0]
    i = a[len(a) // 2]
    core.configure(k0=4, k1=4, topn=3, act=1)
    key = {"kingdom": K, "seed": 31, "seat": 0, "opp": "c1", "act": 1, "topn": 3}
    seats, kg = core.replay_game(key)
    core.lib.bv_set_regrade(int(r["DIDX"][i]), 8, 99, 1)
    core.lib.bv_set_regrade_kind(core.ACTION)
    try:
        g = core.play(seats, kg, 31, 32)["rec"]
    finally:
        core.lib.bv_set_regrade(-1, 0, 0, 0)
        core.lib.bv_set_regrade_kind(core.BUY)
        core.configure(act=0)
    j = np.where((g["KIND"] == core.ACTION) & (g["DIDX"] == r["DIDX"][i]))[0]
    assert len(j) == 1 and len(g["K"]) == 1                             # only the target was graded
    assert np.array_equal(g["X"][j[0]], r["X"][i]) and g["INC"][j[0]] == r["INC"][i] and g["K"][j[0]] == 8


def test_rollout_errors_mean_no_label_and_no_override():
    plain = _game(0, 0)
    before = int(core.lib.bv_rejected())
    core.lib.bv_set_fault(1)                                            # every rollout errors
    t = _game(1, 1, margin=-1.0)                                        # a teacher that would override everything
    assert t["scores"] == plain["scores"] and t["turns"] == plain["turns"]
    assert len(t["rec"]["K"]) == 0 and int(core.lib.bv_rejected()) > before
    core.lib.bv_set_fault(97)                                           # some errors: those decisions are dropped
    lab = _game(2, 1)
    full = _game(2, 1)
    assert 0 < len(lab["rec"]["K"]) < len(full["rec"]["K"])


def test_zero_se_ties_draw_on():
    """With adaptive draws (4 then up to 20), a decision whose two best candidates are tied with zero paired SE is not
    resolved: it gets every draw."""
    r = _game(2, 1, k0=4, k1=20, se_mult=1.5)["rec"]
    seen = 0
    for i in range(len(r["K"])):
        q = r["Q"][i][r["CAND"][i].astype(bool)]
        if len(q) >= 2 and np.all(q == q[0]) and np.all(np.nan_to_num(r["SE"][i]) == 0):
            seen += 1
            assert r["K"][i] == 20
    assert seen > 0


def test_validation_draws_only_for_a_different_winner():
    r = _game(2, 0, kval=8)["rec"]
    inc_q = r["Q"][np.arange(len(r["K"])), r["INC"]]
    other = np.array([np.nanmax(np.where(r["CAND"][i].astype(bool), r["Q"][i], -1)) > inc_q[i]
                      for i in range(len(r["K"]))])                    # a strictly better graded candidate
    assert other.any() and (r["VK"][other] == 8).all() and (r["VK"][~other] == 0).all()
    assert (np.abs(r["VG"][other]) <= 1).all() and (r["VSE"][other] >= 0).all()
    t = _game(1, 0, kval=8, margin=-1.0)                                # validation also gates teacher play:
    r = t["rec"]                                                        # an override passed both checks
    over = np.where(r["CHOSE"] != r["INC"])[0]
    assert len(over) and (r["VK"][over] == 8).all() and (r["VG"][over] > -1.0).all()


def test_prepare_once_equals_per_candidate():
    a = _game(2, 1, prep=1)["rec"]
    b = _game(2, 1, prep=0)["rec"]
    for x in ("Q", "SE", "K", "INC", "CAND", "CAPS", "X"):
        assert np.array_equal(np.nan_to_num(a[x], nan=-9), np.nan_to_num(b[x], nan=-9)), x


def test_public_sampler_observation_equivalence():
    """Public view: an opponent discard card swapped with a hidden draw-pile card is the same observation (identical
    grades), but not for the simulator sampler; cards below a public top card may be reordered, the public top card
    itself may not."""
    hits = [0, 0, 0]
    for s in range(1, 15):
        for t in (5, 7, 9, 11):
            r = core.obs_test(s, turn=t, k=16)
            if r is None:
                continue
            pub_swap, sim_swap, below_top, top = r
            assert pub_swap in (0.0, -1.0) and below_top in (0.0, -1.0)
            hits[0] += sim_swap > 0
            hits[1] += top > 0
            hits[2] += 1
            break
    assert hits[2] >= 5 and hits[0] >= 3 and hits[1] >= 3


@pytest.mark.parametrize("cards,public", [(["Harbinger", "Artisan", "Sentry"], False), (["Bureaucrat"], True)])
def test_public_top_cards(cards, public):
    """Harbinger, Artisan and Sentry put cards back that only their owner sees; Bureaucrat's are public."""
    k = core.random_kingdom(random.Random(9), cards)
    core.configure(k0=1, k1=1, topn=2)
    n0 = core.pub_stats()
    for c in cards:                                     # both players' first buys forced to 2 copies of the card
        core.lib.rl_set_force(core.ID[c], core.ID[c], 2, 6)
        try:
            for s in range(4):
                core.play([core.seat_for("c2", k, 2), core.seat_for("c2", k, 2)], k, 200 + s, 9 + s)
        finally:
            core.lib.rl_set_force(-1, -1, 2, 6)
    n1 = core.pub_stats()
    assert n1[0] - n0[0] > 100
    assert (n1[1] - n0[1] > 0) == public


def test_public_sampler_and_mixture_grade_legal_reproducible_labels():
    mix = [core.seat_for("c1", K), core.seat_for("ga2", K), core.seat_for("rl1", K)]
    out = []
    for _ in range(2):
        core.set_mix(mix)
        try:
            out.append(_game(2, 0, obs=1)["rec"])
        finally:
            core.set_mix([])
    a, b = out
    assert np.array_equal(np.nan_to_num(a["Q"], nan=-1), np.nan_to_num(b["Q"], nan=-1))
    plain = _game(2, 0)["rec"]
    assert len(plain["K"]) == len(a["K"]) and not np.array_equal(np.nan_to_num(plain["Q"], nan=-1),
                                                                 np.nan_to_num(a["Q"], nan=-1))
    q = a["Q"][~np.isnan(a["Q"])]
    assert ((q >= 0) & (q <= 1)).all()


def test_label_overflow_is_reported(monkeypatch):
    monkeypatch.setattr(core, "MAX_REC", 3)
    r = _game(2, 1)
    assert r["overflow"] and len(r["rec"]["K"]) == 3
