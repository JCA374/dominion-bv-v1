# dominion-bv-v1: a buy-value challenger

Can a network that **learns the value of each buy** from paired rollouts beat our best model, RL-v3 (c1 iteration 2300, c2 iteration 2654)?

- **Teacher** (`c/bv.c`, `bv_grade`): at each buy, every candidate (c2's top 5 buys, its own choice and PASS) is played to the end of the game on the same 16–64 hidden-state draws. Frozen policies continue for both players: c2 for us, the real opponent for them. The label is the mean final score (win 1, tie ½). Hidden zones are put in a canonical order before `de_hide`, so equal observations give equal samples.
- **Student** (`bv/student.py`, C twin `bv_student`): a Deep Sets network over the 33 card tokens (two rounds of mean+max pooled context, width 64). It scores every legal buy and PASS in one pass, and it overrides c2's buy only by more than a margin chosen on dev kingdoms.
- **Curriculum** (`train.py`): labels (c2 plays) → student fit → DAgger (the student plays, the teacher grades) → promotion (2,000 reserved kingdoms, both seats, vs c1 and c2; ≥53% vs both with the 95% interval above 50%).

From the proposal `DOMINION_CHALLENGER_PROPOSAL.md` (the workspace root). The engine is pinned as the `engine/` submodule. RL-v3 (`rl-v3/`, which brings az-v1 and dominion-v1) supplies the net, the features and the seats. No rules are copied.

```
python match.py --name gate --bv 1 --k0 16 --k1 32 --margin 0.04 --kingdoms 120   # teacher vs c2 (the gate)
python -u train.py --run p1 [--continue]                                          # the curriculum
python -m pytest -q tests
```
Python: rl-v3's venv (torch CPU). Frozen models are in `models/` (`SHA256SUMS`).
