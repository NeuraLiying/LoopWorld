#!/usr/bin/env python3
"""Run Stage A geometry & interface checks. Exit 0 iff all pass."""

from __future__ import annotations

import argparse
import os
import sys

# allow running from repo root without install
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from loopworld.checks import run_all_checks
from loopworld.model import LoopWorldModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    print(f"Stage A checks on device={device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")

    # Param counts for frozen configs
    for w, h, label in [(256, 4, "smoke_256"), (640, 10, "main_640")]:
        m = LoopWorldModel(image_size=64, patch_size=4, width=w, heads=h)
        n = m.count_parameters()
        print(f"  params[{label}]: {n:,} ({n/1e6:.2f}M)")

    ok, results = run_all_checks(device)
    print("\n=== Stage A results ===")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name}: {r.detail}")
    print("=== summary: {} ===".format("ALL PASSED" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
