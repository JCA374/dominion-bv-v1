# CLAUDE.md

bv-v1: buy-value challenger (proposal: ../../DOMINION_CHALLENGER_PROPOSAL.md). Rollout teacher in `c/bv.c` (includes
rl-v3's rl.c), Deep Sets student `bv/student.py` (C twin `bv_student`, parity test), `match.py` (paired matches,
streams gate/dev/reserved), `collect.py` (labels), `train.py` (curriculum run, `runs/<run>/`).

- Python: `/home/<user>/code/games/dominion/trainers/rl-v3/.venv/bin/python` (rl-v3's venv; use the absolute path).
- Run as systemd user units (oomd killed terminal runs before): `systemd-run --user --unit=bv-p1 --working-directory=$PWD
  bash -c 'exec <venv python> -u train.py --run p1 --continue >> logs/p1.log 2>&1'`. Stop: `systemctl --user stop bv-p1`.
- Teacher cost: ~5 ms per rollout per core; a buy grades ~6 candidates x 16-64 draws.
- Never put personal names or home paths on the site, in commits or podcasts.
