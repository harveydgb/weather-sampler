"""Solve Joint MAP at the faithfulness-budget operating point (steering plan W3).

The budget rule: the largest smoothness weight lambda that keeps at least 95%
of cells within 0.125 nat (~ half a local sigma) of their best emitted peak,
i.e. a global off-mode fraction (dNLL > 0.125) of at most 5%. Lambda is found
by log-lambda interpolation between the persisted sweep rows that bracket the
5% crossing (read from scores.csv, never re-derived), solved once at the
production Method 1 settings, with at most ONE corrective re-interpolation if
the achieved fraction lands more than half a percentage point off the budget.

Writes `budget_point.npz` into the run dir. NO existing artifact is touched;
lambda* and every headline number are unaffected (DEC-R46, report_steering_plan.md).

Run from the repo root:
    .venv/bin/python scripts/run_budget_point.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from sampler_research.baselines import mode_field
from sampler_research.graph import scale_free_roughness, sparse_laplacian
from sampler_research.io import load_real_marginal
from sampler_research.method4_mrf import delta_nll_to_best_mode
from sampler_research.regularised_map import minimise_at_lambda, objective

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "outputs" / "runs" / "phase_4_fc48_14ep_step8"

# Production Method 1 settings, mirrored from scripts/run_phase4_real.py.
N_RESTARTS = 4
N_STEPS = 400
LR = 0.05
SEED = 7  # fixed, documented; independent of the sweep's seed+idx sequence

KEEP_FRAC = 0.95           # the budget: >= 95% of cells within the threshold
DNLL_THRESHOLD = 0.125     # nat; the report's standing off-mode cut (~0.5 sigma)
FRAC_TOL = 0.005           # accept within +-0.5 pp of the 5% budget


def _sweep_frac_rows(run_dir):
    """(lambda, off-mode frac) for the persisted m1 sweep rows, from scores.csv."""
    rows = []
    with open(run_dir / "scores.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["name"].startswith("m1_lam"):
                lam = float(row["name"][len("m1_lam"):])
                rows.append((lam, float(row["dnll_frac_gt_0p125"])))
    return sorted(rows)


def _interp_lambda(lo, hi, target):
    """Log-lambda linear interpolation of the off-mode fraction to `target`."""
    (lam_lo, f_lo), (lam_hi, f_hi) = lo, hi
    t = (target - f_lo) / (f_hi - f_lo)
    return float(10 ** (np.log10(lam_lo) + t * (np.log10(lam_hi) - np.log10(lam_lo))))


def _solve(pi, mu, sigma, lam, edges, lap, restart_scale):
    """Sanity-guarded production solve (run_phase4_real._solve_lambda convention)."""
    warm, _ = mode_field(pi, mu, sigma)
    warm_energy = objective(warm, pi, mu, sigma, lam, laplacian=lap, n_edges=len(edges))
    lr = LR
    for attempt in range(2):
        res = minimise_at_lambda(
            pi, mu, sigma, lam,
            n_restarts=N_RESTARTS, n_steps=N_STEPS, lr=lr,
            restart_scale=restart_scale, rng=np.random.default_rng(SEED),
            laplacian=lap, edges=edges,
        )
        if bool(np.isfinite(res.energy) and res.energy <= warm_energy + 1e-9):
            return res, lr
        if attempt == 0:
            print(f"[budget] lambda={lam:g}: energy above warm start; halving lr once")
            lr = LR / 2.0
    sys.exit(f"[budget] solve at lambda={lam:g} failed the warm-start sanity guard twice")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    args = parser.parse_args()
    run_dir = args.run_dir

    provenance = json.loads((run_dir / "provenance.json").read_text())
    data = load_real_marginal(Path(provenance["data_path"]))
    pi, mu, sigma = data["pi"], data["mu"], data["sigma"]
    restart_scale = float(np.median(sigma))
    with np.load(run_dir / "masks.npz") as f:
        edges = f["edges"]
    lap = sparse_laplacian(pi.shape[0], edges)
    with np.load(run_dir / "modes.npz") as f:
        mode_values, valid_mask = f["mode_values"], f["valid_mask"]

    sweep = _sweep_frac_rows(run_dir)
    budget = 1.0 - KEEP_FRAC
    below = [r for r in sweep if r[1] <= budget and r[0] > 0]
    above = [r for r in sweep if r[1] > budget]
    if not below or not above:
        sys.exit(f"[budget] sweep rows do not bracket the {budget:.0%} budget: {sweep}")
    lo, hi = max(below), min(above)
    print(f"[budget] bracketing sweep rows: lambda={lo[0]:g} ({lo[1]:.2%}) / "
          f"lambda={hi[0]:g} ({hi[1]:.2%}); target frac <= {budget:.2%}")

    solves = []
    lam = _interp_lambda(lo, hi, budget)
    for attempt in range(2):
        t0 = time.perf_counter()
        res, lr_used = _solve(pi, mu, sigma, lam, edges, lap, restart_scale)
        delta = delta_nll_to_best_mode(res.field, pi, mu, sigma, mode_values, valid_mask)
        frac = float(np.mean(delta["per_cell"] > DNLL_THRESHOLD))
        r_tilde, collapsed = scale_free_roughness(res.field, edges)
        wall = time.perf_counter() - t0
        solves.append((lam, frac))
        print(f"[budget] lambda={lam:.3f}: NLL/N={res.nll_over_n:.4f} "
              f"R~={r_tilde:.5f} frac_off={frac:.4f} collapsed={collapsed} ({wall:.1f} s)")
        if collapsed:
            sys.exit("[budget] variance collapse at the budget point (unexpected)")
        if abs(frac - budget) <= FRAC_TOL:
            break
        if attempt == 0:
            # One corrective step: re-interpolate between the new point and the
            # sweep row on the other side of the budget.
            other = lo if frac > budget else hi
            lam = _interp_lambda(min((lam, frac), other), max((lam, frac), other), budget)
            print(f"[budget] {frac:.2%} outside {budget:.2%}+-{FRAC_TOL:.1%}; "
                  f"one corrective solve at lambda={lam:.3f}")
    else:
        print(f"[budget] WARNING: settled at frac={solves[-1][1]:.2%} after the "
              f"single corrective step (documented as achieved value)")

    out = run_dir / "budget_point.npz"
    np.savez(
        out,
        field=res.field,
        lambda_budget=np.float64(solves[-1][0]),
        nll_over_n=np.float64(res.nll_over_n),
        r_tilde=np.float64(r_tilde),
        frac_off=np.float64(solves[-1][1]),
        keep_frac=np.float64(KEEP_FRAC),
        dnll_threshold=np.float64(DNLL_THRESHOLD),
        solves=np.asarray(solves, dtype=float),
        seed=np.int64(SEED),
        n_steps=np.int64(N_STEPS),
        n_restarts=np.int64(N_RESTARTS),
        lr_used=np.float64(lr_used),
        restart_scale=np.float64(restart_scale),
    )
    print(f"[budget] wrote {out}")


if __name__ == "__main__":
    main()
