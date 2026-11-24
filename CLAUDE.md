# CLAUDE.md

bv-v1: buy-value challenger (proposal: ../../DOMINION_CHALLENGER_PROPOSAL.md). Rollout teacher in `c/bv.c` (includes
rl-v3's rl.c), Deep Sets student `bv/student.py` (C twin `bv_student`, parity test), `match.py` (paired matches,
streams gate/dev/reserved), `collect.py` (labels), `train.py` (curriculum run, `runs/<run>/`).

- Python: `/home/<user>/code/games/dominion/trainers/rl-v3/.venv/bin/python` (rl-v3's venv; use the absolute path).
- Run as systemd user units (oomd killed terminal runs before): `systemd-run --user --unit=bv-p1 --working-directory=$PWD
  bash -c 'exec <venv python> -u train.py --run p1 --continue >> logs/p1.log 2>&1'`. Stop: `systemctl --user stop bv-p1`.
- Teacher cost: ~5 ms per rollout per core; a buy grades ~6 candidates x 16-64 draws.
- Never put personal names or home paths on the site, in commits or podcasts.
- Handoff fixes (2026-11-24, tests/test_handoff_fixes.py, test_bank.py): buys and actions share one grader
  (bv_decide_main); bv_set_act 1/2 with bv mode 2 = action labels only (c2 plays); any rollout error -> no label, no
  override; zero-SE ties draw on; bv_set_kval = independent validation draws (gate play too); bv_set_obs 1 = public
  sampler (opponent's deck+hand+discard pooled, public Bureaucrat top cards kept, g_pub); bv_set_mix = rollout
  opponent mixture; bv_set_prep (one hidden-state preparation per draw: identical grades, no measurable speed-up);
  records carry LEGAL, KIND, VG/VSE/VK, KN; shards carry a cohort id (collect.meta); match.py writes its manifest
  (sources, library sha, game list sha) before game one, validates resumed rows, scores completed pairs, --games file;
  bank.py requires replay identity (fails on mismatches), scores at the deployment margin, strict shortlist misses.
