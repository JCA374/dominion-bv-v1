import ctypes

import numpy as np
import torch

from bv.core import NA, NF, lib
from bv.student import Student, export

F = ctypes.POINTER(ctypes.c_float)


def test_c_matches_torch():
    torch.manual_seed(0)
    m = Student()
    flat = export(m)
    rng = np.random.default_rng(0)
    x = rng.random((5, NF)).astype(np.float32)
    x[:, 4 * 33:5 * 33] = rng.random((5, 33)) < 0.5
    want = torch.sigmoid(m(torch.tensor(x))).detach().numpy()
    for i in range(5):
        q = np.zeros(NA, np.float32)
        lib.bv_student(flat.ctypes.data_as(F), np.ascontiguousarray(x[i]).ctypes.data_as(F), q.ctypes.data_as(F))
        assert np.allclose(q, want[i], atol=1e-5), (q, want[i])
