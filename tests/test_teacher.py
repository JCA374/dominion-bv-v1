import random

import numpy as np

from bv import core


def _game(bv):
    core.configure(k0=4, k1=4, topn=3)
    k = core.random_kingdom(random.Random(5))
    return core.play([core.seat_for("c2", k, bv), core.seat_for("c2", k)], k, 11, 12)


def test_labels_reproducible_and_legal():
    a, b = _game(2), _game(2)
    assert len(a["rec"]["K"]) > 5
    assert np.array_equal(np.nan_to_num(a["rec"]["Q"], nan=-1), np.nan_to_num(b["rec"]["Q"], nan=-1))
    r = a["rec"]
    for i in range(len(r["K"])):
        q = r["Q"][i]
        assert not np.isnan(q[r["INC"][i]])            # the incumbent is always graded
        assert r["CHOSE"][i] == r["INC"][i]            # mode 2: the incumbent plays
        assert np.all((q[~np.isnan(q)] >= 0) & (q[~np.isnan(q)] <= 1))


def test_labels_only_mode_plays_like_plain_c2():
    plain = _game(0)
    lab = _game(2)
    assert plain["scores"] == lab["scores"] and plain["turns"] == lab["turns"]
