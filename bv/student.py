"""The student: a Deep Sets network over card tokens that scores every legal buy (and PASS) in one pass. Mirrors
bv_student in c/bv.c exactly (tests/test_student.py checks parity); export() writes the flat C layout."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from bv.core import NA, NF, NUM_CARDS, lib  # noqa: F401

N, TOK, GLB, W, EMB = NUM_CARDS, 17, 16, 64, 8
AZ_NF = 5 * N + 10
_C = __import__("dominion_engine.cards", fromlist=["CARDS"])


def _rules():
    """Per card: cost/8, +cards/3, +actions/2, +coins/3, +buys, action, attack, victory, treasure (= bv_tokens)."""
    import ctypes
    out = np.zeros((N, 9), np.float32)
    tok = np.zeros(N * TOK, np.float32)
    glb = np.zeros(GLB, np.float32)
    x = np.zeros(NF, np.float32)
    pol = np.zeros(NA, np.float32)
    F = ctypes.POINTER(ctypes.c_float)
    lib.bv_tokens(x.ctypes.data_as(F), pol.ctypes.data_as(F), tok.ctypes.data_as(F), glb.ctypes.data_as(F))
    out[:] = tok.reshape(N, TOK)[:, 7:16]
    return torch.tensor(out)


RULES = _rules()


def tokens(x: torch.Tensor, pol: torch.Tensor):
    """x [B, NF] rl features, pol [B, NA] c2's candidate probabilities -> tok [B, N, TOK], glb [B, GLB], in_game [B, N]"""
    B = x.shape[0]
    zone = [x[:, i * N:(i + 1) * N] for i in range(5)] + [x[:, AZ_NF:AZ_NF + N], x[:, AZ_NF + N:AZ_NF + 2 * N]]
    tok = torch.cat([torch.stack(zone, -1), RULES.unsqueeze(0).expand(B, N, 9), pol[:, :N, None]], -1)
    glb = torch.cat([x[:, 5 * N:5 * N + 10], x[:, AZ_NF + 2 * N:AZ_NF + 2 * N + 1], pol[:, N:N + 1],
                     x.new_zeros(B, GLB - 12)], -1)
    return tok, glb, x[:, 4 * N:5 * N]


def _pool(h, m):
    mm = m.unsqueeze(-1)
    mean = (h * mm).sum(1) / mm.sum(1).clamp(min=1)
    mx = (h * mm).max(1).values          # h >= 0 (relu), so masked zeros never win over a positive
    return torch.cat([mean, mx], -1)


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Parameter(torch.randn(N, EMB) * 0.1)
        self.g0 = nn.Linear(TOK + EMB + GLB, W)
        self.u = nn.ModuleList([nn.Linear(3 * W, W) for _ in range(2)])
        self.h1, self.h2 = nn.Linear(3 * W + GLB, W), nn.Linear(W, 1)
        self.p1, self.p2 = nn.Linear(2 * W + GLB, W), nn.Linear(W, 1)

    def forward(self, x, pol):
        """logits [B, NA] (sigmoid = expected final score)"""
        tok, glb, m = tokens(x, pol)
        B = x.shape[0]
        G = glb.unsqueeze(1).expand(B, N, GLB)
        h = torch.relu(self.g0(torch.cat([tok, self.emb.unsqueeze(0).expand(B, N, EMB), G], -1)))
        for u in self.u:
            p = _pool(h, m).unsqueeze(1).expand(B, N, 2 * W)
            h = torch.relu(u(torch.cat([h, p], -1))) + h
        p = _pool(h, m)
        card = self.h2(torch.relu(self.h1(torch.cat([h, p.unsqueeze(1).expand(B, N, 2 * W), G], -1)))).squeeze(-1)
        ps = self.p2(torch.relu(self.p1(torch.cat([p, glb], -1))))
        return torch.cat([card, ps], -1)


def export(model: Student) -> np.ndarray:
    parts = [model.emb]
    for lin in [model.g0, *model.u, model.h1, model.h2, model.p1, model.p2]:
        parts += [lin.weight, lin.bias]
    flat = np.concatenate([p.detach().cpu().numpy().ravel() for p in parts]).astype(np.float32)
    assert flat.size == lib.bv_student_params(), (flat.size, lib.bv_student_params())
    return flat
