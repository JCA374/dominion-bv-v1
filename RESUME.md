# Paused 2026-11-23 21:46 (resume here)

Stopped at a clean point: round 25's labels are saved (20,082 states, the first with replay keys) and the student
was refit on them (dev regret 0.0465 vs incumbent 0.0454, top-1 33%); its dev match had not started.

Resume, in this order (each as a systemd user unit, as before):

1. Training: `python -u train.py --run p1 --continue --fit-first` (refits on all labels, then dev at margins
   0.02/0.05/0.1; the next collection is round 26 with the known-top sampler).
2. Teacher suite (frozen old build `bv_suite.so`, picked automatically for `--stream suite`):
   `match.py --name suite_action --act 2 --bv 1 --opp c2 --stream suite --k0 32 --k1 32 --margin 0.02 --z 2 --kingdoms 150 --workers 3 --resume`
   (214 of 300 games done), then the same with `--name suite_both --act 1` (not started).
   Buy-only is done: 59.0% [54.7, 63.2] vs c2, 300 games.
3. Validation bank and controls (restart from zero; deterministic): `experiments/bank.py --n 1000 --k 128 --workers 2`,
   then `experiments/controls.py`.

Workers in train.py are back at 7; drop to 4 while the suite runs beside it. Decide on scaling training from the bank,
the controls and a fresh match, not from the reused dev suite.
