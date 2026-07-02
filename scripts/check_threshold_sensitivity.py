"""R4 threshold-sensitivity check: frac(dNLL > t) for Joint MAP vs Smoothed MAP
at several cuts, from the PERSISTED per-cell dNLL arrays (no new computation).
Read-only. Canonical +48h (phase_4_fc48_14ep_step8) and AE recon (phase_4_real)."""
import numpy as np
from pathlib import Path

RUNS = Path("/users/harvey_bermingham/weather-sampler-research/outputs/runs")
CUTS = [0.03125, 0.0625, 0.125, 0.25, 0.5]  # 0.25σ, ~0.35σ, 0.5σ, ~0.7σ, 1σ equivalents

for label, run in [("fc48 +48h canonical (14ep step8)", "phase_4_fc48_14ep_step8"),
                   ("AE reconstruction", "phase_4_real")]:
    d = RUNS / run
    with np.load(d / "delta_per_cell.npz") as f:
        m1, blur = np.asarray(f["m1_star"], float), np.asarray(f["smoothed_map_n10"], float)
    masks = {}
    if (d / "masks.npz").exists():
        with np.load(d / "masks.npz") as f:
            masks = {k: np.asarray(f[k], bool) for k in f.files if f[k].shape == m1.shape}
    strata = {"global": np.ones(m1.shape[0], bool)}
    for k in ("bimodal", "bimodal_1sigma"):
        if k in masks: strata[k] = masks[k]
    print(f"\n=== {label} ({run}) N={m1.shape[0]} ===")
    for sname, sel in strata.items():
        a, b = m1[sel], blur[sel]
        print(f"  stratum {sname:15s} n={sel.sum():6d}  mean dNLL: m1={a.mean():.4f} blur={b.mean():.4f} gap={a.mean()-b.mean():+.4f}")
        for t in CUTS:
            fa, fb = float((a > t).mean()), float((b > t).mean())
            tag = " <-- headline cut" if t == 0.125 else ""
            print(f"    t={t:<8g} frac: m1={fa:.4f} blur={fb:.4f} gap={fa-fb:+.4f}{tag}")
