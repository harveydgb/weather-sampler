"""Decompose the fc48 bimodal-stratum DEEP tail (dNLL > 0.5 nat): is each such
value resting ON a secondary mode (mis-scored by the best-mode convention) or
genuinely smeared into the gap between modes? Read-only over persisted artifacts."""
import numpy as np

D = "outputs/runs/phase_4_fc48_14ep_step8/"
with np.load(D + "delta_per_cell.npz") as f:
    d_m1, d_bl = f["m1_star"], f["smoothed_map_n10"]
with np.load(D + "masks.npz") as f:
    bimodal = f["bimodal"].astype(bool)
with np.load(D + "modes.npz") as f:
    mv, valid, mu_u = f["mode_values"], f["valid_mask"].astype(bool), f["mode_unary"]
with np.load(D + "method1_sensitivity.npz") as f:
    x_m1 = f["field_star"]
with np.load(D + "anchors.npz") as f:
    x_bl = f["smoothed_map_n10"]

SIGMA = 0.15  # App A: jitter 0.15 "close to the median component spread"

def decompose(x, d, label, cut=0.5):
    sel = bimodal & (d > cut)
    idx = np.where(sel)[0]
    on_secondary = mid_gap = other = 0
    dists = []
    for i in idx:
        v = valid[i]
        modes, u = mv[i][v], mu_u[i][v]
        best = np.argmin(u)
        dist = np.abs(modes - x[i])
        nn = np.argmin(dist)
        dists.append(dist[nn] / SIGMA)
        if nn != best and dist[nn] < 0.5 * SIGMA:
            on_secondary += 1          # hugging a non-best mode
        elif dist[nn] >= 0.5 * SIGMA:
            mid_gap += 1               # >0.5 sigma from EVERY mode: true smear
        else:
            other += 1                 # near best mode yet dNLL>0.5 (odd cases)
    n = len(idx)
    print(f"{label}: deep-tail n={n} ({n/bimodal.sum():.1%} of stratum)")
    if n:
        print(f"  on a SECONDARY mode (<0.5 sigma from a non-best mode): {on_secondary} ({on_secondary/n:.1%})")
        print(f"  off EVERY mode by >0.5 sigma (genuine smear):          {mid_gap} ({mid_gap/n:.1%})")
        print(f"  near best mode (other):                                {other} ({other/n:.1%})")
        print(f"  median dist to nearest mode: {np.median(dists):.2f} sigma")

print(f"bimodal stratum n={bimodal.sum()}")
decompose(x_m1, d_m1, "Joint MAP @ lambda*")
decompose(x_bl, d_bl, "Smoothed MAP (n=10)")
