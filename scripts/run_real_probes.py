"""Phase 4 robustness probes -> outputs/runs/phase_4_real/robustness_probes.json

Deterministic re-derivation of the 11 Jun post-implementation review probes, so
the report quotes persisted numbers instead of throwaway /tmp evidence. Run
after `scripts/run_real_eval.py` (all stages + --lambda-star):

    .venv/bin/python scripts/run_real_probes.py

For a converged forecast run, the lambda* seed-stability probe alone (CPU-only,
no model inference) runs under --seed-stability-only: it reads the run-specific
restart_scale from that run's lambda_star.json (NOT the AE 0.15), loads the
shared o96 kNN graph for edges, and writes a robustness_probes.json holding only
that one block into the run dir:

    .venv/bin/python scripts/run_real_probes.py --seed-stability-only \
        --run-dir outputs/runs/phase_4_fc48_14ep_step8 \
        --data-npz outputs/data/phase_4_fc48_14ep_step8_2t.npz

Reads the persisted run artifacts (masks/modes/anchors/method1_sweep/
lambda_star) plus the converted real npz; writes ONE small JSON artifact and
touches nothing else. Probes (review findings in brackets):

  m4_beta_scale       solve_value_mrf at beta in {10, 100} -- two decades above
                      the swept [0.01, 1] grid, at/above the matched-energy
                      scale of lambda* ~ 94 -- plus the unary-gap distribution
                      over multi-mode cells. Evidence that the pinned beta
                      sweep is a data property (near-one-hot unary gaps), not
                      an under-scaled grid. [I1]
  lambda_star_seed_stability
                      re-solve the two rows bracketing lambda* (lambda 30 and
                      100, production settings) at restart-seed offsets
                      +1000/+2000 and re-run the S6 selection rule. [C]
  blur_stability      lambda_max(L) of the real unit-weight Laplacian and
                      step * lambda_max vs the stability bound 2 (monotone
                      bound 1). [I2]
  iid_wrap_seam       analytic iid expectation E[(x_i - x_j)^2] =
                      Var_i + Var_j + (m_i - m_j)^2 on wrap vs all edges, plus
                      endpoint mixture variance -- why the observed iid
                      wrap-seam ratio ~ 0.45 is expected, not a near-miss. [M3]
  weighted_graph      declared Gaussian distance-decay ablation as a probe,
                      not a pipeline variant [B]: w = exp(-d^2 / 2 ell^2) with
                      ell = median 1st-neighbour arc; weighted smoothed-MAP n10
                      target, weighted Method 1 rows at lambda in {30,100,300},
                      interpolated lambda*_w and its scores. R~ is always the
                      unit-weight metric so both sides share one convention.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import eigsh

from sampler_research.baselines import gmm_nll_per_cell, smoothed_map_baseline
from sampler_research.graph import (
    edge_arc_km,
    scale_free_roughness,
    sparse_laplacian,
)
from sampler_research.io import load_real_marginal
from sampler_research.method4_mrf import (
    ModeExtraction,
    _unary_best_assignment,
    delta_nll_to_best_mode,
    solve_value_mrf,
)
from sampler_research.phase4_eval import select_lambda_star
from sampler_research.regularised_map import minimise_at_lambda

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_NPZ = REPO_ROOT / "outputs" / "data" / "phase_4_real_2t.npz"
RUN_DIR = REPO_ROOT / "outputs" / "runs" / "phase_4_real"
GRAPH_NPZ = REPO_ROOT / "outputs" / "runs" / "o96_knn_k8_graph.npz"

# Production Method 1 / Method 4 settings (mirrors scripts/run_real_eval.py).
N_RESTARTS = 4
N_STEPS = 400
LR = 0.05
RESTART_SCALE = 0.15
M4_RESTARTS = 2
M4_MAX_SWEEPS = 30
SMOOTH_STEP = 0.1
SMOOTH_N_ITERS = 10
# Runner seed convention: sweep row seed = base 0 + index in the lambda grid
# (0, 0.5, 2, 8, 30, 100, 300, 1000), so lambda=30 -> 4 and lambda=100 -> 5.
BASE_ROW_SEEDS = {30.0: 4, 100.0: 5}
SEED_OFFSETS = (1000, 2000)
PROBE_BETAS = (10.0, 100.0)
WEIGHTED_LAMBDAS = ((30.0, 4), (100.0, 5), (300.0, 6))
WEIGHTED_STAR_SEED = 200  # mirrors the runner's sensitivity-solve seed


def _solve_m1(data, lam, seed, laplacian, edges, restart_scale=RESTART_SCALE):
    return minimise_at_lambda(
        data["pi"], data["mu"], data["sigma"], lam,
        n_restarts=N_RESTARTS, n_steps=N_STEPS, lr=LR,
        restart_scale=restart_scale, rng=np.random.default_rng(seed),
        laplacian=laplacian, edges=edges,
    )


def probe_m4_beta_scale(data, extraction, edges):
    unary_best = _unary_best_assignment(extraction)
    multi = extraction.mode_counts >= 2
    # mode_unary is +inf on padded slots, so the first two sorted columns are
    # the best and second-best real modes wherever mode_counts >= 2.
    sorted_unary = np.sort(extraction.mode_unary, axis=1)
    gaps = (sorted_unary[:, 1] - sorted_unary[:, 0])[multi]
    out = {
        "n_cells": int(extraction.mode_counts.size),
        "n_multi_mode_cells": int(multi.sum()),
        "unary_gap_nats": {
            "median": float(np.median(gaps)),
            "p10": float(np.percentile(gaps, 10)),
            "p90": float(np.percentile(gaps, 90)),
        },
        "settings": {"n_restarts": M4_RESTARTS, "max_sweeps": M4_MAX_SWEEPS, "seed": 0},
        "betas": {},
    }
    for beta in PROBE_BETAS:
        res = solve_value_mrf(
            data["pi"], data["mu"], data["sigma"], extraction, beta,
            n_restarts=M4_RESTARTS, edges=edges,
            rng=np.random.default_rng(0), max_sweeps=M4_MAX_SWEEPS,
        )
        moved = int(np.sum(np.asarray(res.assignment) != unary_best))
        out["betas"][f"{beta:g}"] = {
            "cells_moved_off_unary_best": moved,
            "nll_over_n": float(res.nll_over_n),
            "r_tilde": float(res.r_tilde),
            "variance_collapsed": bool(res.variance_collapsed),
        }
        print(f"[m4-beta] beta={beta:g}: moved {moved}/{out['n_cells']} cells, "
              f"R~={res.r_tilde:.5f}, NLL/N={res.nll_over_n:.4f}")
    print(f"[m4-beta] unary gap (multi-mode cells): median "
          f"{out['unary_gap_nats']['median']:.1f} nats "
          f"(p10 {out['unary_gap_nats']['p10']:.1f}, p90 {out['unary_gap_nats']['p90']:.1f})")
    return out


def probe_lambda_star_seed_stability(data, edges, laplacian, sweep, star,
                                     restart_scale=RESTART_SCALE):
    target = float(star["target_r_tilde"])
    out = {
        "base_lambda_star": float(star["lambda_star"]),
        "resolved_lambdas": sorted(BASE_ROW_SEEDS),
        "settings": {"n_restarts": N_RESTARTS, "n_steps": N_STEPS, "lr": LR,
                     "restart_scale": restart_scale},
        "offsets": {},
    }
    for offset in SEED_OFFSETS:
        r_tildes = np.array(sweep["r_tilde"], dtype=float)
        collapsed = np.array(sweep["variance_collapsed"], dtype=bool)
        valid = np.array(sweep["warm_start_sane"], dtype=bool)
        for lam, base_seed in BASE_ROW_SEEDS.items():
            res = _solve_m1(data, lam, base_seed + offset, laplacian, edges,
                            restart_scale=restart_scale)
            row = int(np.flatnonzero(np.isclose(sweep["lambdas"], lam))[0])
            r_tildes[row] = res.r_tilde
            collapsed[row] = res.variance_collapsed
        sel = select_lambda_star(sweep["lambdas"], r_tildes, collapsed, target, valid=valid)
        out["offsets"][f"+{offset}"] = {
            "lambda_star": float(sel.lambda_star),
            "bracketed": bool(sel.bracketed),
            "clauses": list(sel.clauses),
        }
        print(f"[lambda*-seeds] offset +{offset}: lambda* = {sel.lambda_star:.4f} "
              f"(bracketed={sel.bracketed}, clauses={sel.clauses or 'none'})")
    return out


def _lambda_max(laplacian):
    return float(eigsh(laplacian, k=1, which="LM", return_eigenvectors=False)[0])


def probe_blur_stability(n_cells, edges):
    lmax = _lambda_max(sparse_laplacian(n_cells, edges))
    out = {
        "lambda_max": lmax,
        "step": SMOOTH_STEP,
        "step_times_lambda_max": SMOOTH_STEP * lmax,
        "stability_bound": 2.0,
        "monotone_bound": 1.0,
    }
    print(f"[blur] lambda_max(L) = {lmax:.2f}; step*lambda_max = "
          f"{out['step_times_lambda_max']:.2f} (stable < 2, monotone < 1)")
    return out


def probe_iid_wrap_seam(data, edges, lon_threshold_deg=350.0):
    pi, mu, sigma, latlons = data["pi"], data["mu"], data["sigma"], data["latlons"]
    m = np.sum(pi * mu, axis=1)
    var = np.sum(pi * (sigma**2 + mu**2), axis=1) - m**2
    expected_sq = var[edges[:, 0]] + var[edges[:, 1]] + (m[edges[:, 0]] - m[edges[:, 1]]) ** 2
    dlon = np.abs(latlons[edges[:, 0], 1] - latlons[edges[:, 1], 1])
    wrap = dlon > lon_threshold_deg
    out = {
        "n_wrap_edges": int(wrap.sum()),
        "analytic_iid_wrap_over_all_ratio": float(expected_sq[wrap].mean() / expected_sq.mean()),
        "mean_mixture_variance_wrap_endpoints": float(np.mean(var[edges[wrap]])),
        "mean_mixture_variance_all_endpoints": float(np.mean(var[edges])),
    }
    print(f"[wrap-seam] analytic iid wrap/all ratio = "
          f"{out['analytic_iid_wrap_over_all_ratio']:.3f} over {out['n_wrap_edges']} wrap edges; "
          f"endpoint mixture variance {out['mean_mixture_variance_wrap_endpoints']:.3f} (wrap) vs "
          f"{out['mean_mixture_variance_all_endpoints']:.3f} (all)")
    return out


def probe_weighted_graph(data, edges, modes):
    pi, mu, sigma = data["pi"], data["mu"], data["sigma"]
    n = pi.shape[0]
    arc = edge_arc_km(data["latlons"], edges)
    nearest = np.full(n, np.inf)
    np.minimum.at(nearest, edges[:, 0], arc)
    np.minimum.at(nearest, edges[:, 1], arc)
    ell = float(np.median(nearest))
    weights = np.exp(-(arc**2) / (2.0 * ell**2))
    lap_w = sparse_laplacian(n, edges, weights=weights)

    blur_w = smoothed_map_baseline(
        pi, mu, sigma, step=SMOOTH_STEP, n_iters=SMOOTH_N_ITERS, laplacian=lap_w
    )
    target_w, target_collapsed = scale_free_roughness(blur_w, edges)
    if target_collapsed:
        raise RuntimeError("weighted smoothed-MAP target is variance-collapsed")
    blur_delta = delta_nll_to_best_mode(
        blur_w, pi, mu, sigma, modes["mode_values"], modes["valid_mask"]
    )

    rows = []
    for lam, seed in WEIGHTED_LAMBDAS:
        res = _solve_m1(data, lam, seed, lap_w, edges)
        rows.append({
            "lambda": lam,
            "nll_over_n": float(res.nll_over_n),
            "r_tilde": float(res.r_tilde),
            "variance_collapsed": bool(res.variance_collapsed),
        })
        print(f"[weighted] lambda={lam:g}: NLL/N={res.nll_over_n:.4f} R~={res.r_tilde:.5f}")
    sel = select_lambda_star(
        [r["lambda"] for r in rows], [r["r_tilde"] for r in rows],
        [r["variance_collapsed"] for r in rows], target_w,
    )
    res_star = _solve_m1(data, sel.lambda_star, WEIGHTED_STAR_SEED, lap_w, edges)
    star_delta = delta_nll_to_best_mode(
        res_star.field, pi, mu, sigma, modes["mode_values"], modes["valid_mask"]
    )

    out = {
        "ell_km": ell,
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "lambda_max_weighted": _lambda_max(lap_w),
        "note": "R~ is the unit-weight metric throughout; weights enter the "
                "blur operator and the Method 1 penalty only",
        "smoothed_map_n10_weighted": {
            "r_tilde": float(target_w),
            "nll_over_n": float(np.mean(gmm_nll_per_cell(blur_w, pi, mu, sigma))),
            "dnll_frac_gt_0p125": float(blur_delta["frac_over"][0]),
        },
        "method1_rows": rows,
        "lambda_star_weighted": float(sel.lambda_star),
        "lambda_star_bracketed": bool(sel.bracketed),
        "m1_at_lambda_star_weighted": {
            "nll_over_n": float(res_star.nll_over_n),
            "r_tilde": float(res_star.r_tilde),
            "dnll_frac_gt_0p125": float(star_delta["frac_over"][0]),
            "variance_collapsed": bool(res_star.variance_collapsed),
        },
    }
    print(f"[weighted] ell = {ell:.1f} km, weights {out['weight_min']:.2f}-{out['weight_max']:.2f}, "
          f"lambda_max(L_w) = {out['lambda_max_weighted']:.1f}")
    print(f"[weighted] blur_w target R~ = {target_w:.5f}; lambda*_w = {sel.lambda_star:.1f} "
          f"(bracketed={sel.bracketed})")
    print(f"[weighted] M1@lambda*_w: NLL/N={res_star.nll_over_n:.4f} "
          f"tail>0.125={star_delta['frac_over'][0]:.3f} vs blur_w "
          f"NLL/N={out['smoothed_map_n10_weighted']['nll_over_n']:.4f} "
          f"tail>0.125={out['smoothed_map_n10_weighted']['dnll_frac_gt_0p125']:.3f}")
    return out


def _run_full_ae(run_dir, data_npz):
    """All five AE-regime probes -> run_dir/robustness_probes.json."""
    t0 = time.perf_counter()
    data = load_real_marginal(data_npz)
    with np.load(run_dir / "masks.npz") as f:
        edges = f["edges"]
    with np.load(run_dir / "modes.npz") as f:
        modes = {k: f[k] for k in f.files}
    extraction = ModeExtraction(
        mode_values=modes["mode_values"], valid_mask=modes["valid_mask"],
        mode_unary=modes["mode_unary"], mode_counts=modes["mode_counts"],
    )
    sweep = dict(np.load(run_dir / "method1_sweep.npz"))
    star = json.loads((run_dir / "lambda_star.json").read_text())
    laplacian = sparse_laplacian(data["pi"].shape[0], edges)

    probes = {
        "m4_beta_scale": probe_m4_beta_scale(data, extraction, edges),
        "lambda_star_seed_stability": probe_lambda_star_seed_stability(
            data, edges, laplacian, sweep, star
        ),
        "blur_stability": probe_blur_stability(data["pi"].shape[0], edges),
        "iid_wrap_seam": probe_iid_wrap_seam(data, edges),
        "weighted_graph": probe_weighted_graph(data, edges, modes),
        "wall_s": None,
    }
    probes["wall_s"] = round(time.perf_counter() - t0, 1)
    out_path = run_dir / "robustness_probes.json"
    out_path.write_text(json.dumps(probes, indent=2) + "\n")
    print(f"wrote {out_path} ({probes['wall_s']} s)")


def _run_seed_stability_only(run_dir, data_npz, graph_npz):
    """Forecast mode: only the lambda* seed-stability probe (CPU-only).

    Threads the run-specific restart_scale (read from the run's lambda_star.json,
    NOT the AE default 0.15) into the Method-1 re-solves, and takes edges from the
    SHARED o96 kNN graph. Writes a robustness_probes.json holding only the
    lambda_star_seed_stability block, so make_real_figures.fig_robustness keeps
    skipping gracefully on runs where the other probes are absent."""
    t0 = time.perf_counter()
    data = load_real_marginal(data_npz)
    with np.load(graph_npz) as f:
        edges = f["edges"]
    sweep = dict(np.load(run_dir / "method1_sweep.npz"))
    star = json.loads((run_dir / "lambda_star.json").read_text())
    restart_scale = float(star["restart_scale"])
    laplacian = sparse_laplacian(data["pi"].shape[0], edges)
    print(f"[seed-stability] {run_dir.name}: lambda*={float(star['lambda_star']):.2f}, "
          f"restart_scale={restart_scale:.4f} (run-specific; AE default is {RESTART_SCALE})")
    probes = {
        "lambda_star_seed_stability": probe_lambda_star_seed_stability(
            data, edges, laplacian, sweep, star, restart_scale=restart_scale
        ),
        "wall_s": None,
    }
    probes["wall_s"] = round(time.perf_counter() - t0, 1)
    out_path = run_dir / "robustness_probes.json"
    out_path.write_text(json.dumps(probes, indent=2) + "\n")
    print(f"wrote {out_path} ({probes['wall_s']} s)")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--data-npz", type=Path, default=DATA_NPZ)
    parser.add_argument("--graph-npz", type=Path, default=GRAPH_NPZ,
                        help="shared kNN graph (edges) used in seed-stability mode")
    parser.add_argument("--seed-stability-only", action="store_true",
                        help="forecast mode: run only the lambda* seed-stability "
                             "probe, reading the run-specific restart_scale from "
                             "the run's lambda_star.json")
    args = parser.parse_args()
    if args.seed_stability_only:
        _run_seed_stability_only(args.run_dir, args.data_npz, args.graph_npz)
    else:
        _run_full_ae(args.run_dir, args.data_npz)


if __name__ == "__main__":
    main()
