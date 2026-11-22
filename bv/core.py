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

SO = os.path.join(ROOT, "bv.so")
SRC = [os.path.join(ENGINE_C, "engine.c"), os.path.join(ROOT, "c", "bv.c")]
DEPS = SRC + [os.path.join(RL3, "c", "rl.c"), os.path.join(AZ, "c", "az.c"), os.path.join(ENGINE_C, "engine.h")]


def build() -> str:
    if not os.path.exists(SO) or max(os.path.getmtime(d) for d in DEPS) > os.path.getmtime(SO):
        tmp = f"{SO}.{os.getpid()}"
        subprocess.check_call(["gcc", "-O3", "-march=native", "-ffast-math", "-shared", "-fPIC", "-Wall", "-Wextra",
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
lib.bv_pol_of.argtypes = [_F, ctypes.c_int, ctypes.c_int, _F, _F, _F]
lib.bv_stats.argtypes = [ctypes.POINTER(ctypes.c_long)]
lib.bv_play.restype = ctypes.c_int
lib.bv_play.argtypes = [_I, _F, _F, _I, _I, _I, _I, _I, _I, ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64,
                        _F, _F, _F, _I, _I, _I, _I, _I, ctypes.c_int, _I]
lib.rl_num_params.argtypes = [ctypes.c_int, ctypes.c_int]
GA, NET = 0, 2
MAX_REC = 200
MODELS = os.path.join(ROOT, "models")
_P: dict = {}


def params(name):
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


def configure(k0=16, k1=64, se_mult=1.5, topn=5, margin=0.0, student=None, z=0.0):
    lib.bv_set_cfg(k0, k1, se_mult, topn, margin)
    lib.bv_set_z(z)
    configure.student = student                    # keep a reference alive
    lib.bv_set_student(_f(student))


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
    shapes = [net_shape(s.prm) if s.mode == NET else (256, NF) for s in seats]
    out = (ctypes.c_int * 6)()
    pi = lambda a: a.ctypes.data_as(_I)  # noqa: E731
    n = lib.bv_play(_ints([s.mode for s in seats]), _f(seats[0].prm), _f(seats[1].prm),
                    _ints([a for a, _ in shapes]), _ints([b for _, b in shapes]), seats[0].genome, seats[1].genome,
                    _ints([s.bv for s in seats]), k, len(kingdom), seed & (2**64 - 1), rng_seed & (2**64 - 1),
                    _f(X), _f(Q), _f(SE), pi(INC), pi(CH), pi(K), pi(SEAT), pi(TURN), MAX_REC, out)
    m = out[0]
    return {"winners": [i for i in range(2) if m >> i & 1], "scores": [out[1], out[2]], "turns": [out[3], out[4]],
            "capped": bool(out[5] & 1), "error": bool(out[5] & 2),
            "rec": {"X": X[:n], "Q": Q[:n], "SE": SE[:n], "INC": INC[:n], "CHOSE": CH[:n], "K": K[:n],
                    "SEAT": SEAT[:n], "TURN": TURN[:n]}}


def points(r, seat):
    w = r["winners"]
    return 0.5 if len(w) == 2 else (1.0 if seat in w else 0.0)


def seat_for(spec, kingdom, bv=0):
    """spec: 'c2' / 'c1' / 'rl1' (nets) or 'ga' (az-v1's per-kingdom base genome) or 'ga2' (portfolio-v3 base)."""
    if spec == "ga":
        return Seat(GA, base_genome(kingdom)["ints"])
    if spec == "ga2":
        from az.base import base_genome_v2
        return Seat(GA, base_genome_v2(kingdom)["ints"])
    name = {"c2": "rl3_c2_2654.npy", "c1": "rl3_c1_2300.npy", "rl1": "rl1_r2_2200.npy"}[spec]
    return Seat(NET, None, params(name), bv)     # a net seat never reads the genome


def policy_prior(X, Q, net="rl3_c2_2654.npy"):
    """c2's probabilities over each recorded state's graded candidates (the student's prior input)."""
    p = params(net)
    nh, nf = net_shape(p)
    P = np.zeros((len(X), NA), np.float32)
    X = np.ascontiguousarray(X, np.float32)
    Q = np.ascontiguousarray(Q, np.float32)
    for i in range(len(X)):
        lib.bv_pol_of(_f(p), nh, nf, _f(X[i]), _f(Q[i]), _f(P[i]))
    return P
