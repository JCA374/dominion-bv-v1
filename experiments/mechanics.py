"""Game-mechanics analysis: which decisions exist, which the challenger touches, and where c2's buys are systematically
improvable. Writes docs/reports/game-mechanics.html (+ results/mechanics.json).

    python experiments/mechanics.py [--games 200]

1. Decision mix: c2 vs c2 games through rl-v3's recorder (every decision with 2+ options, by kind).
2. Buy labels (data/p1): per (c2's buy -> alternative) pair, the mean paired difference across positions. Averaging one
   fixed contrast over many positions is unbiased (unlike the max over candidates in one position, which is inflated by
   noise), so a pair whose mean is clearly above 0 is a systematic c2 mistake the student could learn.
3. By game phase, coins, multi-buy turns and PASS.
"""
import argparse, base64, glob, io, json, math, os, random, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def decision_mix(n):
    sys.path.insert(0, os.path.join(ROOT, "rl-v3"))
    from rl import core as rc
    p = np.load(os.path.join(ROOT, "models", "rl3_c2_2654.npy")).astype(np.float32)
    rng = random.Random("mechanics mix")
    kinds, opts, turns = [], [], 0
    for i in range(n):
        k = rc.random_kingdom(rng)
        g = rc.base_genome(k)["ints"]
        r = rc.play([rc.Seat(rc.NET, g, p, record=True), rc.Seat(rc.NET, g, p, record=True)], k, rng.getrandbits(32), i + 1)
        kinds += list(r["rec"]["KIND"])
        opts += list(r["rec"]["LEGAL"].sum(1))
        turns += sum(r["turns"])
    kinds, opts = np.array(kinds), np.array(opts)
    rows = []
    for kd in sorted(set(kinds), key=lambda x: -(kinds == x).sum()):
        m = kinds == kd
        rows.append({"kind": rc.KINDS[kd], "per_game": round(m.sum() / n, 2), "share": round(m.mean(), 4),
                     "mean_options": round(float(opts[m].mean()), 2)})
    return {"games": n, "decisions_per_game": round(len(kinds) / n, 1), "turns_per_game": round(turns / n, 1), "kinds": rows}


def labels():
    import importlib.util
    spec = importlib.util.spec_from_file_location("bv_train", os.path.join(ROOT, "train.py"))
    tr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tr)
    load = tr.load
    from bv.core import NAME, NA
    D = load("p1")
    Q, SE, INC, X = D["Q"], D["SE"], D["INC"], D["X"]
    N = 33
    coins = np.rint(X[:, 5 * N] * 8).astype(int)
    buys = np.rint(X[:, 5 * N + 1] * 2).astype(int)
    turn = D["TURN"]
    name = lambda a: "PASS" if a == NA - 1 else NAME[a]
    n = len(Q)
    qi = Q[np.arange(n), INC]
    diff = Q - qi[:, None]                       # paired difference to c2's buy (NaN = not a candidate)
    # per pair
    pairs = {}
    for a in range(NA):
        col = diff[:, a]
        ok = ~np.isnan(col) & (INC != a)
        for inc in np.unique(INC[ok]):
            m = ok & (INC == inc)
            if m.sum() < 300:
                continue
            v = col[m]
            pairs[(name(inc), name(a))] = (float(v.mean()), float(v.std() / math.sqrt(len(v))), int(len(v)))
    top = sorted(pairs.items(), key=lambda kv: -kv[1][0] / max(kv[1][1], 1e-9))
    best_pairs = [{"c2": k[0], "alt": k[1], "gain": round(v[0], 4), "se": round(v[1], 4), "n": v[2],
                   "z": round(v[0] / max(v[1], 1e-9), 1)} for k, v in top[:15]]
    worst_pairs = [{"c2": k[0], "alt": k[1], "gain": round(v[0], 4), "se": round(v[1], 4), "n": v[2],
                    "z": round(v[0] / max(v[1], 1e-9), 1)} for k, v in top[-6:]]

    def by(groups):
        out = []
        for label, m in groups:
            if m.sum() < 200:
                continue
            g = np.nanmax(diff[m], 1)            # noisy-best gain (inflated) for reference
            seb = np.take_along_axis(np.nan_to_num(SE[m], nan=9), np.nanargmax(np.where(np.isnan(diff[m]), -9, diff[m]), 1)[:, None], 1)[:, 0]
            clear = (g > 0.02) & (g > 2 * seb)
            out.append({"group": label, "n": int(m.sum()), "clear_override_share": round(float(clear.mean()), 4),
                        "mean_candidates": round(float((~np.isnan(Q[m])).sum(1).mean()), 2)})
        return out

    phase = by([("turns 1-2", turn <= 2), ("turns 3-8", (turn >= 3) & (turn <= 8)), ("turns 9-14", (turn >= 9) & (turn <= 14)),
                ("turns 15+", turn >= 15)])
    coin = by([(f"${c}", coins == c) for c in range(0, 9)] + [("$9+", coins >= 9)])
    multi = by([("1 buy", buys <= 1), ("2+ buys", buys >= 2)])
    pas = by([("c2 passes", INC == NA - 1), ("c2 buys a card", INC != NA - 1)])
    inc_counts = {}
    for a in INC:
        inc_counts[name(a)] = inc_counts.get(name(a), 0) + 1
    return {"positions": int(n), "pairs_tested": len(pairs), "best_pairs": best_pairs, "worst_pairs": worst_pairs,
            "phase": phase, "coins": coin, "multi_buy": multi, "pass": pas,
            "c2_buys": sorted(inc_counts.items(), key=lambda kv: -kv[1])[:14]}


def act_result():
    p = os.path.join(ROOT, "results", "match_act_teacher_c2.jsonl")
    if not os.path.exists(p):
        return None
    r = [json.loads(l) for l in open(p)]
    s = json.load(open(p[:-1])) if os.path.exists(p[:-1]) else {}
    tot = {k: sum(x[k] for x in r) for k in ("act_graded", "act_overrides", "act_play_not_stop", "act_stop_not_play", "act_other_action")}
    return {"games": len(r), "score": s.get("score"), "ci95": s.get("ci95"), **tot}


def bars(rows, key, label, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(rows) + 1.2))
    ys = [r[label] for r in rows][::-1]
    vs = [100 * r[key] for r in rows][::-1]
    ax.barh(ys, vs, color="#4a7fd0")
    for y, v in zip(ys, vs):
        ax.text(v, y, f" {v:.1f}%", va="center", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("% of positions")
    fig.tight_layout()
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=110)
    plt.close(fig)
    return f'<img style="max-width:100%" src="data:image/png;base64,{base64.b64encode(b.getvalue()).decode()}">'


def table(rows, cols):
    h = "".join(f"<th>{c}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{r.get(c, '')}</td>" for c in cols) + "</tr>" for r in rows)
    return f"<table><tr>{h}</tr>{body}</table>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    mix = decision_mix(a.games)
    lab = labels()
    act = act_result()
    json.dump({"mix": mix, "labels": lab, "action_teacher": act}, open(os.path.join(ROOT, "results", "mechanics.json"), "w"), indent=1)
    print(json.dumps({"mix": mix["kinds"][:6], "best_pairs": lab["best_pairs"][:8], "phase": lab["phase"], "act": act}, indent=0)[:3000])


if __name__ == "__main__" and "--report" not in sys.argv:
    main()


def report():
    """docs/reports/game-mechanics.html from results/mechanics.json (+ the action teacher's newest numbers)."""
    d = json.load(open(os.path.join(ROOT, "results", "mechanics.json")))
    d["action_teacher"] = act_result() or d.get("action_teacher")
    mix, lab, act = d["mix"], d["labels"], d["action_teacher"]
    kinds = mix["kinds"]
    buy = next(r for r in kinds if r["kind"] == "buy")
    play = next(r for r in kinds if r["kind"] == "play action")
    gains = sum(r["per_game"] for r in kinds if r["kind"] in ("Artisan gain", "Remodel gain", "Workshop gain", "Mine gain"))
    for r in kinds:
        r["touched by bv-v1"] = "yes (teacher + student)" if r["kind"] == "buy" else ("experiment: action teacher" if r["kind"] == "play action" else "no: c2 decides")
    act_txt = "still running"
    if act and act.get("games"):
        sc = f"{100 * act['score']:.1f}%" if act.get("score") is not None else "pending"
        ci = f" (95% interval {100 * act['ci95'][0]:.1f}–{100 * act['ci95'][1]:.1f}%)" if act.get("ci95") else ""
        act_txt = (f"{act['games']} games so far: score {sc}{ci} against plain c2. It graded {act['act_graded']} action decisions and "
                   f"overrode {act['act_overrides']} ({100 * act['act_overrides'] / max(act['act_graded'], 1):.1f}%): "
                   f"{act['act_other_action']} picked a different action, {act['act_stop_not_play']} stopped where c2 would play, "
                   f"{act['act_play_not_stop']} played where c2 would stop.")
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Game mechanics: what the challenger touches</title>
<style>:root{{--bg:#fff;--fg:#1b1b1b;--mut:#666;--line:#ddd}}@media (prefers-color-scheme:dark){{:root{{--bg:#16181c;--fg:#e8e8e8;--mut:#9a9a9a;--line:#333}}}}
body{{background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,sans-serif;max-width:960px;margin:0 auto;padding:16px}}
table{{border-collapse:collapse;width:100%;font-size:14px;display:block;overflow-x:auto}}td,th{{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left}}
.box{{border:1px solid var(--line);border-radius:8px;padding:10px 14px;margin:12px 0}}.mut{{color:var(--mut)}}</style></head><body>
<p class="mut"><a href="../log.html">← back to the site</a></p>
<h1>Game mechanics: what the challenger touches, and where c2 errs</h1>
<p>Questions: does the challenger only look at buys? Are playing actions, their order, buying nothing and playing nothing taken into account? And why has the student stopped improving?</p>
<div class="box"><b>Answers in short.</b>
<ul><li><b>Buys only.</b> The teacher and the student change only buy decisions. Buying nothing (PASS) is always one of the graded candidates. Every other decision is c2's, in the real game and inside every rollout: which action to play, in what order, when to stop playing actions, Throne Room targets, trashing, gains, discards.</li>
<li><b>The rest is the bigger share.</b> Of {mix['decisions_per_game']} decisions per game with two or more options (both players), buys are {buy['per_game']:.0f} ({100 * buy['share']:.0f}%). Action plays are {play['per_game']:.0f} ({100 * play['share']:.0f}%), with {play['mean_options']} options on average. Gains from Artisan, Remodel, Workshop and Mine ({gains:.1f} per game, 10–13 options each) are buy-like choices that the challenger does not touch.</li>
<li><b>c2's buys have no card-level mistakes.</b> Over {lab['positions']:,} labelled positions, {lab['pairs_tested']} pairs "c2 bought X, the alternative was Y" (at least 300 positions each) were tested. No pair beats c2 on average beyond chance: the best has z = {lab['best_pairs'][0]['z']} among {lab['pairs_tested']} tests. The teacher's wins (55–59%) come from <i>rare, position-specific</i> exceptions. That is why the student, which learns from labels with 16 draws (paired SE about 0.1 per position), has stalled near c2.</li>
<li><b>Action teacher (experiment).</b> {act_txt}</li></ul></div>
<h2>1. Every decision kind (c2 vs c2, {mix['games']} games, both players; {mix['turns_per_game']} turns per game)</h2>
{table(kinds, ['kind', 'per_game', 'share', 'mean_options', 'touched by bv-v1'])}
{bars(kinds[:12], 'share', 'kind', 'Share of decisions with 2+ options')}
<h2>2. Is any of c2's buys systematically wrong?</h2>
<p>For one position, taking the best of about 6 noisy candidates overstates its gain (the winner's curse). Averaging <i>one fixed contrast</i>, c2's buy against one alternative, over many positions is unbiased. Highest average gains over c2's choice (positive = c2 should have taken the alternative):</p>
{table(lab['best_pairs'], ['c2', 'alt', 'gain', 'se', 'n', 'z'])}
<p>The strongest pairs are against c2, for example (negative = c2 is right):</p>
{table(lab['worst_pairs'], ['c2', 'alt', 'gain', 'se', 'n', 'z'])}
<h2>3. Where the teacher sees a clear gain (gain > 2 SE and > 0.02, at 16 draws)</h2>
<p>This is noisy per position. The shares show where the opportunities sit, not how many are real.</p>
{table(lab['phase'], ['group', 'n', 'clear_override_share', 'mean_candidates'])}
{table(lab['coins'], ['group', 'n', 'clear_override_share', 'mean_candidates'])}
{table(lab['multi_buy'] + lab['pass'], ['group', 'n', 'clear_override_share', 'mean_candidates'])}
{bars(lab['coins'], 'clear_override_share', 'group', 'Clear-gain share by coins at the buy')}
<h2>4. What this means</h2>
<ol><li>The student's plateau is a signal-to-noise problem, not a missing input. The gains are rare (a few % of positions), small, and depend on the position. More positions alone move the regret slowly (0.062 → 0.051 over 340k positions).</li>
<li>The gated teacher already captures them (54.9% vs c2, 59.2% vs c1 on reserved kingdoms). It is the model to use for play now, at about 1 second per buy.</li>
<li>The next gains are likely outside buys: action plays are twice as many decisions, and gains are buy-like. The action teacher measures whether rollouts improve c2 there too. If they do, the same gated teacher can cover both, and gains can be added once the engine has a resume point for gain decisions (an engine change, no rule change).</li></ol>
<p class="mut">Made by experiments/mechanics.py from data/p1 and fresh c2 games; numbers in results/mechanics.json.</p>
</body></html>"""
    os.makedirs(os.path.join(ROOT, "docs", "reports"), exist_ok=True)
    open(os.path.join(ROOT, "docs", "reports", "game-mechanics.html"), "w").write(html)
    print("wrote docs/reports/game-mechanics.html")


if __name__ == "__main__" and "--report" in sys.argv:
    report()
