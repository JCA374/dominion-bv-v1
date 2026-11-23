"""The proposal's controls on identical labels, scored on the frozen validation bank (data/bank/bank.npz):
incumbent (c2 imitation: always c2's buy), the card-set student, the same student trained on the paired gain only, and
a flat MLP of the same input (413 features + c2's candidate prior) and a similar parameter count.

    python experiments/controls.py [--epochs 6]
Out: results/controls.json (also listed on the site).
"""
import argparse, importlib.util, json, os, sys
import numpy as np
import torch
import torch.nn as nn
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location("bv_train", os.path.join(ROOT, "train.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)


class Flat(nn.Module):
    """413 features + 34 prior -> 96 -> 96 -> 34 (about 50k parameters, like the student)."""
    def __init__(self, nf=413, na=34, h=96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nf + na, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, na))

    def forward(self, x, pol):
        return self.net(torch.cat([x, pol], -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    a = ap.parse_args()
    T.CFG["epochs"] = a.epochs
    D = T.load("p1")
    out = {}
    from bv.student import Student
    for name, value_w, cls in (("card-set student", 0.1, Student), ("card-set, paired gain only", 0.0, Student),
                               ("flat MLP", 0.1, Flat)):
        T.CFG["value_w"] = value_w
        import bv.student as bs
        keep, keep_x = bs.Student, bs.export
        bs.Student, bs.export = cls, (lambda m: None)      # fit() builds bv.student.Student; no C export needed
        try:
            m, _, ev = T.fit(D, None, log=lambda *x: None)
        finally:
            bs.Student, bs.export = keep, keep_x
        out[name] = {k: v for k, v in ev.items() if k.startswith(("bank", "dev_regret", "dev_top1"))}
        print(name, out[name], flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "controls.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
