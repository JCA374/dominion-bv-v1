"""The C side through ctypes: engine/c/engine.c + c/bv.c (which includes rl-v3's rl.c, az-v1's az.c and
dominion-v1's ga_agent.c) built into bv.so. Kingdoms, genomes and bots come from az-v1 (submodule rl-v3/az-v1)."""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RL3 = os.path.join(ROOT, "rl-v3")
AZ = os.path.join(RL3, "az-v1")
ENGINE_C = os.path.join(ROOT, "engine", "c")
for p in (AZ, os.path.join(AZ, "dominion-v1")):
    if p not in sys.path:
        sys.path.insert(0, p)

from az.base import base_genome, random_kingdom  # noqa: E402,F401
from dominion_engine.cards import ID, NAME, NUM_CARDS  # noqa: E402,F401

SO = os.environ.get("BV_SO") or os.path.join(ROOT, "bv.so")     # BV_SO: a test build beside the live one
SRC = [os.path.join(ENGINE_C, "engine.c"), os.path.join(ROOT, "c", "bv.c")]
DEPS = SRC + [os.path.join(ROOT, "c", "fastnet.h"), os.path.join(RL3, "c", "rl.c"), os.path.join(AZ, "c", "az.c"), os.path.join(ENGINE_C, "engine.h")]


def build() -> str:
    if os.environ.get("BV_SO") and os.path.exists(SO):
        return SO                                    # a frozen build (match arms that must share one library)
    if not os.path.exists(SO) or max(os.path.getmtime(d) for d in DEPS) > os.path.getmtime(SO):
        tmp = f"{SO}.{os.getpid()}"
        subprocess.check_call(["gcc", "-O3", "-march=native", "-ffast-math", "-fno-finite-math-only", "-shared", "-fPIC", "-Wall", "-Wextra",
                               "-Wno-unused-parameter", "-Wno-unused-function", "-I", ENGINE_C, "-o", tmp, *SRC, "-lm"])
        os.replace(tmp, SO)
    return SO


lib = ctypes.CDLL(build())
NF = lib.rl_num_features()
NA = lib.rl_num_actions()
PASS = NA - 1
NP_STUDENT = lib.bv_student_params()
TOK, GLB = lib.bv_tok(), lib.bv_glb()
_I, _F = ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float)
lib.bv_set_cfg.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_float]
lib.bv_set_student.argtypes = [_F]
lib.bv_student.argtypes = [_F, _F, _F, _F]
lib.bv_tokens.argtypes = [_F, _F, _F, _F]
_U8 = ctypes.POINTER(ctypes.c_ubyte)
lib.bv_pol_of.argtypes = [_F, ctypes.c_int, ctypes.c_int, _F, _U8, _F]
lib.bv_stats.argtypes = [ctypes.POINTER(ctypes.c_long)]
lib.bv_play.restype = ctypes.c_int
_NEW = hasattr(lib, "bv_set_obs")             # the frozen suite build (bv_suite.so) predates the new label fields
lib.bv_play.argtypes = [_I, _F, _F, _I, _I, _I, _I, _I, _I, ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64,
                        _F, _F, _F, _I, _I, _I, _I, _I, _U8, _F, _I, _U8] + \
                       ([_U8, _I, _F, _F, _I, _I] if _NEW else []) + [ctypes.c_int, _I]
if _NEW:
    lib.bv_set_obs.argtypes = [ctypes.c_int]
    lib.bv_set_kval.argtypes = [ctypes.c_int]
    lib.bv_set_prep.argtypes = [ctypes.c_int]
    lib.bv_preps.restype = ctypes.c_long
    lib.bv_set_regrade_kind.argtypes = [ctypes.c_int]
    lib.bv_pub_stats.argtypes = [ctypes.POINTER(ctypes.c_long)]
    lib.bv_set_mix.argtypes = [ctypes.c_int, _I, ctypes.POINTER(_F), _I, _I, ctypes.POINTER(_I)]
lib.bv_set_regrade.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_int]
lib.bv_rejected.restype = ctypes.c_long
lib.bv_lat.argtypes = [_F, ctypes.c_int]
lib.bv_act_log.argtypes = [_F, ctypes.c_int]
if hasattr(lib, "bv_known_stats"):              # absent in the frozen suite build
    lib.bv_known_stats.argtypes = [ctypes.POINTER(ctypes.c_long)]
if _NEW:
    lib.bv_set_fault.argtypes = [ctypes.c_int]
    lib.bv_obs_test.argtypes = [_F, ctypes.c_int, ctypes.c_int, _I, ctypes.c_int, ctypes.c_uint64, ctypes.c_int, ctypes.c_int, _F]
lib.bv_canon_test.argtypes = [_F, ctypes.c_int, ctypes.c_int, _I, ctypes.c_int, ctypes.c_uint64, ctypes.c_int, ctypes.c_int, _F]
lib.rl_num_params.argtypes = [ctypes.c_int, ctypes.c_int]
GA, NET = 0, 2
MAX_REC = 800                                  # labels per game (buys and, with the action teacher, action decisions)
BUY, ACTION = 1, 0                             # decision kinds (engine DE_BUY, DE_PLAY_ACTION)
MODELS = os.path.join(ROOT, "models")
_P: dict = {}


def params(name):
    """A net's weights by file name in models/ or by absolute path."""
    if name not in _P:
        _P[name] = np.ascontiguousarray(np.load(os.path.join(MODELS, name)).astype(np.float32))
    return _P[name]


def sha(name):
    return hashlib.sha256(open(os.path.join(MODELS, name), "rb").read()).hexdigest()[:16]


def net_shape(p):
    for nf in (NF, lib.rl_num_features_base()):
        b = nf + NA + 4
        nh = int(round((-b + (b * b + 4 * (p.size - NA - 1)) ** 0.5) / 2))
        if nh > 0 and lib.rl_num_params(nh, nf) == p.size:
            return nh, nf
    raise ValueError(p.size)


def _ints(xs):
    return (ctypes.c_int * len(xs))(*xs)


def _f(a):
    return None if a is None else a.ctypes.data_as(_F)


class Seat:
    """mode NET (params = an rl net, every decision) or GA (genome plays all). bv: 0 plain, 1 teacher plays,
    2 teacher labels (incumbent plays), 3 student plays, 4 student plays + teacher labels."""

    def __init__(self, mode, genome, prm=None, bv=0):
        self.mode, self.genome, self.prm, self.bv = mode, genome, prm, bv


lib.bv_set_z.argtypes = [ctypes.c_float]
lib.bv_set_fast.argtypes = [ctypes.c_int]
lib.bv_set_act.argtypes = [ctypes.c_int]
lib.bv_act_stats.argtypes = [ctypes.POINTER(ctypes.c_long)]


def configure(k0=16, k1=64, se_mult=1.5, topn=5, margin=0.0, student=None, z=0.0, act=None, obs=0, kval=0,
              prep=1):
    """Teacher settings. act: None = leave as is, 0 buys, 1 buys + actions, 2 actions only. obs: the sampler's view of
    the opponent (0 simulator, 1 public). kval: independent validation draws for the graded winner (0 = off; in play
    mode the override must also pass on them). prep: 1 = each hidden-state draw prepared once (0 = per candidate)."""
    configure.last = dict(k0=k0, k1=k1, se_mult=se_mult, topn=topn, margin=margin, student=student, z=z, act=act,
                          obs=obs, kval=kval, prep=prep)
    lib.bv_set_cfg(k0, k1, se_mult, topn, margin)
    lib.bv_set_z(z)
    configure.student = student                    # keep a reference alive
    lib.bv_set_student(_f(student))
    if act is not None:
        lib.bv_set_act(act)
    if _NEW:
        lib.bv_set_obs(obs)
        lib.bv_set_kval(kval)
        lib.bv_set_prep(prep)
    elif obs or kval or not prep:
        raise RuntimeError("this build has no obs / kval / prep settings")


def set_mix(seats):
    """Rollouts continue the opponent with seats[k % n] at draw k (a fixed mixture, independent of the actual
    opponent); [] = the actual opponent. seats: Seat objects (NET or GA) for the game's kingdom."""
    set_mix.keep = seats
    n = len(seats)
    if not n:
        lib.bv_set_mix(0, None, None, None, None, None)
        return
    shapes = [net_shape(s.prm) if s.mode == NET else (256, NF) for s in seats]
    PF = (_F * n)(*[_f(s.prm) if s.prm is not None else None for s in seats])
    PG = (_I * n)(*[s.genome if s.genome is not None else None for s in seats])
    lib.bv_set_mix(n, _ints([s.mode for s in seats]), PF, _ints([a for a, _ in shapes]), _ints([b for _, b in shapes]), PG)


def stats():
    o = (ctypes.c_long * 6)()
    lib.bv_stats(o)
    return dict(zip(["rollouts", "capped", "errors", "graded", "overrides", "buys"], list(o)))


def play(seats, kingdom, seed, rng_seed=1):
    k = _ints([ID[c] for c in kingdom])
    X = np.zeros((MAX_REC, NF), np.float32)
    Q = np.zeros((MAX_REC, NA), np.float32)
    SE = np.zeros((MAX_REC, NA), np.float32)
    INC, CH, K, SEAT, TURN = (np.zeros(MAX_REC, np.int32) for _ in range(5))
    CAND = np.zeros((MAX_REC, NA), np.uint8)
    CAPS = np.zeros((MAX_REC, NA), np.uint8)
    POL = np.zeros((MAX_REC, NA), np.float32)
    DIDX = np.zeros(MAX_REC, np.int32)
    LEGAL = np.zeros((MAX_REC, NA), np.uint8)
    KIND = np.full(MAX_REC, BUY, np.int32)
    VG, VSE = np.zeros(MAX_REC, np.float32), np.zeros(MAX_REC, np.float32)
    VK = np.zeros(MAX_REC, np.int32)
    KN = np.zeros(MAX_REC, np.int32)
    shapes = [net_shape(s.prm) if s.mode == NET else (256, NF) for s in seats]
    out = (ctypes.c_int * 6)()
    pi = lambda a: a.ctypes.data_as(_I)  # noqa: E731
    n = lib.bv_play(_ints([s.mode for s in seats]), _f(seats[0].prm), _f(seats[1].prm),
                    _ints([a for a, _ in shapes]), _ints([b for _, b in shapes]), seats[0].genome, seats[1].genome,
                    _ints([s.bv for s in seats]), k, len(kingdom), seed & (2**64 - 1), rng_seed & (2**64 - 1),
                    _f(X), _f(Q), _f(SE), pi(INC), pi(CH), pi(K), pi(SEAT), pi(TURN),
                    CAND.ctypes.data_as(_U8), _f(POL), pi(DIDX), CAPS.ctypes.data_as(_U8),
                    *([LEGAL.ctypes.data_as(_U8), pi(KIND), _f(VG), _f(VSE), pi(VK), pi(KN)] if _NEW else []), MAX_REC, out)
    m = out[0]
    rec = {"X": X[:n], "Q": Q[:n], "SE": SE[:n], "INC": INC[:n], "CHOSE": CH[:n], "K": K[:n], "SEAT": SEAT[:n],
           "TURN": TURN[:n], "CAND": CAND[:n], "POL": POL[:n], "DIDX": DIDX[:n], "CAPS": CAPS[:n]}
    if _NEW:
        rec.update(LEGAL=LEGAL[:n], KIND=KIND[:n], VG=VG[:n], VSE=VSE[:n], VK=VK[:n], KN=KN[:n])
    return {"winners": [i for i in range(2) if m >> i & 1], "scores": [out[1], out[2]], "turns": [out[3], out[4]],
            "capped": bool(out[5] & 1), "error": bool(out[5] & 2), "overflow": bool(out[5] & 4), "rec": rec}


if hasattr(lib, "bv_advise"):
    lib.bv_advise.restype = ctypes.c_int
    lib.bv_advise.argtypes = [_F, ctypes.c_int, ctypes.c_int, _I, ctypes.c_int, _I, _I, _I, ctypes.c_int, ctypes.c_int,
                              ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_uint64,
                              _F, _F, _U8, _F]


def advise(prm, supply, me, zones, turns, kind, coins=0, buys=1, actions=1, merchants=0, inc=-1, secs=3.0, seed=1):
    """Grade one position from outside a game (the Steam advisor; C bv_advise), stopped after `secs` seconds. Draws:
    configure()'s k0/k1/se_mult/topn. zones: 8 card-id lists [my deck, hand, discard, play, their deck, hand, discard,
    play]; supply: per card (-1 = not in the game); kind: BUY or ACTION; inc: the pick to compare against (-1 = the
    net's). -> {"draws", "ms", "inc", "net", "value", "q", "se", "cand"}; draws 0: no time, -1: engine error, -2: one
    option."""
    nh, nf = net_shape(prm)
    flat = [c for z in zones for c in z]
    q, se = np.full(NA, np.nan, np.float32), np.full(NA, np.nan, np.float32)
    cand, out = np.zeros(NA, np.uint8), np.zeros(5, np.float32)
    K = lib.bv_advise(_f(prm), nh, nf, _ints(list(supply)), me, _ints(flat or [0]), _ints([len(z) for z in zones]),
                      _ints(list(turns)), kind, coins, buys, actions, merchants, inc, secs, seed & (2**64 - 1),
                      _f(q), _f(se), cand.ctypes.data_as(_U8), _f(out))
    return {"draws": K, "ms": float(out[3]), "inc": int(out[0]), "net": int(out[1]), "value": float(out[4]),
            "q": q, "se": se, "cand": cand.astype(bool)}


def points(r, seat):
    w = r["winners"]
    return 0.5 if len(w) == 2 else (1.0 if seat in w else 0.0)


def net_file(spec):
    """The weights file of a net spec: 'c2' / 'c1' / 'rl1' (models/) or 'net:<path to .npy>' (any rl-v3 net)."""
    if spec.startswith("net:"):
        return os.path.abspath(spec[4:])
    return os.path.join(MODELS, {"c2": "rl3_c2_2654.npy", "c1": "rl3_c1_2300.npy", "rl1": "rl1_r2_2200.npy"}[spec])


def seat_for(spec, kingdom, bv=0):
    """spec: 'c2' / 'c1' / 'rl1' (nets), 'net:<path.npy>' (any rl-v3 net), 'ga' (az-v1's per-kingdom base genome) or
    'ga2' (portfolio-v3 base)."""
    if spec.startswith("net:"):
        return Seat(NET, None, params(net_file(spec)), bv)
    if spec == "ga":
        return Seat(GA, base_genome(kingdom)["ints"])
    if spec == "ga2":
        from az.base import base_genome_v2
        return Seat(GA, base_genome_v2(kingdom)["ints"])
    name = {"c2": "rl3_c2_2654.npy", "c1": "rl3_c1_2300.npy", "rl1": "rl1_r2_2200.npy"}[spec]
    return Seat(NET, None, params(name), bv)     # a net seat never reads the genome


def policy_prior(X, mask, net="rl3_c2_2654.npy"):
    """c2's probabilities over each recorded state's graded candidates (the student's prior input). mask: the explicit
    candidate mask [n, NA] (for old shards without one: ~np.isnan(Q), computed in numpy, which keeps NaN checks)."""
    p = params(net)
    nh, nf = net_shape(p)
    P = np.zeros((len(X), NA), np.float32)
    X = np.ascontiguousarray(X, np.float32)
    M = np.ascontiguousarray(mask, np.uint8)
    U8 = ctypes.POINTER(ctypes.c_ubyte)
    for i in range(len(X)):
        lib.bv_pol_of(_f(p), nh, nf, _f(X[i]), M[i].ctypes.data_as(U8), _f(P[i]))
    return P


def latencies():
    """Seconds per graded / student / action-teacher decision since the last call."""
    b = np.zeros(8192, np.float32)
    return b[:lib.bv_lat(_f(b), 8192)].copy()


def action_log():
    """Action overrides since the last call: rows (turn, c2's choice, teacher's choice, paired gain, SE)."""
    b = np.zeros((512, 5), np.float32)
    return b[:lib.bv_act_log(_f(b), 512)].copy()


def replay_game(key):
    """The seats and kingdom of a recorded label (key: kingdom, seed, seat, opp, and for DAgger labels the student
    file, margin and topn it played with) as collect.py played it. Configures the student (or none) to match."""
    k = key["kingdom"]
    st = np.load(key["student"]).astype(np.float32) if key.get("student") else None
    c = dict(getattr(configure, "last", {}))
    c.update(topn=key.get("topn", 5), margin=key.get("margin", 0.02), student=st)
    if "act" in key:                               # action labels: the action teacher must count decisions the same
        c["act"] = key["act"]
    configure(**c)
    me = seat_for("c2", k, 4 if st is not None else 2)
    op = seat_for(key["opp"], k)
    return ([me, op] if key["seat"] == 0 else [op, me]), k


def student_file(run, sha):
    """Where collect.py keeps the exact student a DAgger round played with (content-addressed by its sha)."""
    return os.path.join(ROOT, "data", run, "students", f"{sha}.npy")


def key_of(D, i, meta=None, opps=("c2", "c1", "rl1", "ga2")):
    """The replay key of label row i of a loaded/np.load-ed shard dict; meta = the shard's META (needed for DAgger
    rows: they replay with the same student)."""
    key = {"kingdom": str(D["KING"][i]).split(","), "seed": int(D["SEED"][i]), "seat": int(D["SEAT0"][i]),
           "opp": opps[int(D["OPP"][i])]}
    if meta is not None:
        key.update(topn=meta["topn"], margin=meta["margin"])
        if meta.get("dagger"):
            key["student"] = student_file(meta.get("run", "p1"), meta["student"])
        if meta.get("act"):
            key["act"] = meta["act"]
    return key


def regrade(key, didx, k, seed, all_legal=True, kind=BUY):
    """Replay a recorded game and grade only its decision number didx (counted within its kind) with k draws from
    seed (all legal buys if all_legal). Returns that record (dict of arrays, one row) or None."""
    seats, kg = replay_game(key)
    lib.bv_set_regrade(int(didx), int(k), int(seed) & (2**64 - 1), int(all_legal))
    if _NEW:
        lib.bv_set_regrade_kind(int(kind))
    try:
        r = play(seats, kg, key["seed"], key["seed"] ^ 0x5EED)
    finally:
        lib.bv_set_regrade(-1, 0, 0, 0)
        if _NEW:
            lib.bv_set_regrade_kind(BUY)
    rec = r["rec"]
    i = np.where((rec["DIDX"] == didx) & (rec.get("KIND", np.full(len(rec["DIDX"]), BUY)) == kind))[0]
    return {x: v[i[0]] for x, v in rec.items()} if len(i) else None


def canon_test(seed, turn=5, k=8, kingdom=None):
    """bv_canon_test: max |q diff| for (same observation, different observation, same observation with 2 known top
    cards, a different known top card), or None."""
    import random as _r
    kg = kingdom or random_kingdom(_r.Random(seed))
    p = params("rl3_c2_2654.npy")
    nh, nf = net_shape(p)
    out = np.zeros(4, np.float32)
    ok = lib.bv_canon_test(_f(p), nh, nf, _ints([ID[c] for c in kg]), len(kg), seed, turn, k, _f(out))
    return tuple(float(v) for v in out) if ok else None


def pub_stats():
    """(graded decisions, of which the opponent had public top cards) since the library loaded."""
    o = (ctypes.c_long * 2)()
    lib.bv_pub_stats(o)
    return int(o[0]), int(o[1])


def obs_test(seed, turn=6, k=8, kingdom=None):
    """bv_obs_test: max |q diff| for (public sampler: discard <-> hidden deck swap, simulator sampler: the same swap,
    public sampler: two hidden cards below a public top swapped, public sampler: the public top card changed), or
    None (-1 entries: no such swap in that state)."""
    import random as _r
    kg = kingdom or random_kingdom(_r.Random(seed))
    p = params("rl3_c2_2654.npy")
    nh, nf = net_shape(p)
    out = np.zeros(4, np.float32)
    ok = lib.bv_obs_test(_f(p), nh, nf, _ints([ID[c] for c in kg]), len(kg), seed, turn, k, _f(out))
    return tuple(float(v) for v in out) if ok else None


def known_stats():
    """(graded decisions, of which with known top cards in my draw pile) since the library loaded."""
    o = (ctypes.c_long * 2)()
    lib.bv_known_stats(o)
    return int(o[0]), int(o[1])
