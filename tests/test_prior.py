"""The student's prior input must be the same in training and in play: the explicit candidate mask recorded at
collection, c2's probabilities only on it (exact zeros elsewhere), and the training-side recomputation identical to
the live prior. Uses real recorded states from teacher-labelled games under the production build flags."""
import random

import numpy as np

from bv import core


def test_training_prior_equals_live_prior():
    core.configure(k0=4, k1=4, topn=3)
    rng = random.Random(9)
    recs = []
    for i in range(3):
        k = core.random_kingdom(rng)
        r = core.play([core.seat_for("c2", k, 2), core.seat_for("c2", k)], k, 100 + i, 200 + i)
        recs.append(r["rec"])
    X = np.concatenate([r["X"] for r in recs])
    Q = np.concatenate([r["Q"] for r in recs])
    CAND = np.concatenate([r["CAND"] for r in recs]).astype(bool)
    POL = np.concatenate([r["POL"] for r in recs])
    assert len(X) > 20
    assert np.array_equal(CAND, ~np.isnan(Q))              # the mask is the graded set
    assert np.all(POL[~CAND] == 0)                         # exact zeros off the mask
    assert np.allclose(POL.sum(1), 1, atol=1e-5)
    train = core.policy_prior(X, CAND)
    assert np.array_equal(train, POL)                      # bit-identical to the live prior
