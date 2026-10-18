#!/usr/bin/env python3
"""Print README headline tables from results/*.json (no GPU)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"

def main():
    orbit = json.loads((R / "orbit_stage_e_step083864.json").read_text())
    g = orbit["loop_gains_R4_minus_R1_S5"]
    print("=== Orbit Stage E (24 held-out, 5 Euler steps) ===")
    for mode in ["S", "T", "AR"]:
        cfg = orbit["configs"]["%s_R4_S5" % mode]
        print(
            "  %s: R4 PSNR=%.2f  R4-R1=%.2f dB  AbsRel=%.3f"
            % (mode, cfg["psnr"]["all"], g[mode]["R4_minus_R1"], cfg["absrel"]["all"])
        )
    print("  Headline AR gain (raw): %.2f dB" % g["AR"]["R4_minus_R1"])

    maze = json.loads((R / "maze_layout_test_seeds700000_700031.json").read_text())
    print("\n=== Maze layout test (seeds 700000-700031) ===")
    for row in maze["results"]:
        rev = row["revisit"]["psnr_pred_vs_gt_at_revisit"]
        print("  %s: AR=%.2f  revisit_vs_GT=%.2f" % (row["mode"], row["psnr_all"], rev))

    grid = json.loads((R / "orbit_loop_rxsteps_grid.json").read_text())
    print("\n=== R x steps peak (from grid) ===")
    print(" ", grid.get("peak_S"))

if __name__ == "__main__":
    main()
