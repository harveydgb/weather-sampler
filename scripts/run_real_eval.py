"""Phase 4 — anchors, Method 1 sweep, lambda*, Method 4, and stratified scores
on the real O96 GMM output (phase_4_plan.md S5-S6; data facts in
phase_4_data_audit.md).

Run from the repo root with the local (numpy/scipy-only) venv:

    .venv/bin/python scripts/run_real_eval.py                  # all stages
    .venv/bin/python scripts/run_real_eval.py --stages m1      # one stage
    .venv/bin/python scripts/run_real_eval.py --lambda-star    # S6 rule + sensitivity
    .venv/bin/python scripts/run_real_eval.py --quick --out-dir /tmp/p4smoke

Stages (separately invokable; later stages load earlier artifacts):
  graph   -> masks.npz (stratum masks + k-NN edges + edge_arc_km)
  anchors -> anchors.npz (MAP / mixture mean / iid seeds / smoothed-MAP)
  modes   -> modes.npz (mean-shift mode extraction, heteroscedastic update)
  m1      -> method1_sweep.npz (bracket-guarded lambda sweep)
  m4      -> method4_sweep.npz (value-space MRF beta sweep; SHOULD)
  scores  -> delta_per_cell.npz + scores.csv/.md + variograms.npz + spectra.npz
             (native O96 angular power spectrum, rung-3 bracket) + timings.json
  spectrum-> spectra.npz ONLY, from the persisted method fields (opt-in, not in
             the default run): refreshes the angular power spectrum's curve set
             -- e.g. picking up the W3 faithfulness-budget curve once
             budget_point.npz exists -- without re-scoring or resampling variograms

`--lambda-star` applies the S6 roughness-matching rule to the persisted sweep
(target = smoothed-MAP n_iters=10 R-tilde), extends the sweep decade-by-decade
if unbracketed, and persists lambda_star.json + method1_sensitivity.npz.

The Method 1 grid {0, 0.5, 2, 8, 30, 100, 300, 1000} is the updated
phase_4_plan grid, pre-verified (11 Jun review probe) to bracket smoothed-MAP's
R-tilde ~ 0.003; the 3-point coarse probe {2, 20, 200} + abort converts that
calibration into a permanent guard.
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from sampler_research.baselines import (
    gmm_nll_per_cell,
    mixture_mean_field,
    mode_field,
    smoothed_map_baseline,
)
from sampler_research.diagnostics import (
    sampled_spherical_power_spectrum,
    sampled_spherical_variogram,
)
from sampler_research.gmm import sample_iid_gmm
from sampler_research.graph import (
    edge_arc_km,
    knn_sphere_edges,
    scale_free_roughness,
    sparse_laplacian,
)
from sampler_research.io import load_real_marginal
from sampler_research.method4_mrf import (
    ModeExtraction,
    beta_sweep,
    delta_nll_to_best_mode,
    extract_gmm_modes,
)
from sampler_research.phase4_eval import (
    practically_bimodal_mask,
    select_lambda_star,
    stratified_scores,
    stratum_masks,
    wrap_seam_ratio,
)
from sampler_research.regularised_map import minimise_at_lambda, objective

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_NPZ = REPO_ROOT / "outputs" / "data" / "phase_4_real_2t.npz"
RUN_DIR = REPO_ROOT / "outputs" / "runs" / "phase_4_real"

# The default full-run pipeline. `spectrum` is an extra, opt-in stage (not in the
# default) that regenerates ONLY spectra.npz from the persisted method fields --
# used to refresh the angular power spectrum's curve set (e.g. adding the W3
# faithfulness-budget curve) without re-scoring or re-sampling the variograms.
DEFAULT_STAGES = ("graph", "anchors", "modes", "m1", "m4", "scores")
STAGES = DEFAULT_STAGES + ("spectrum",)

# Updated phase_4_plan S5 grid (log-spaced through the verified target region).
LAMBDA_GRID = (0.0, 0.5, 2.0, 8.0, 30.0, 100.0, 300.0, 1000.0)
PROBE_LAMBDAS = (2.0, 20.0, 200.0)
IID_SEEDS = (0, 1, 2)
SMOOTH_N_ITERS = (5, 10, 20)
SMOOTH_STEP = 0.1
N_RESTARTS = 4
N_STEPS = 400
LR = 0.05
# Default Method 1 restart jitter, kept only as a fallback if sigma is unavailable.
# The runner now derives the operating value from the loaded data (median sigma);
# on the AE data this resolves to ~0.148, matching the historical 0.15.
RESTART_SCALE = 0.15
BETAS = tuple(np.logspace(-2.0, 0.0, 8))
M4_RESTARTS = 2
M4_MAX_SWEEPS = 30
VARIOGRAM_PAIRS = 60_000
TARGET_ANCHOR = "smoothed_map_n10"


def _update_timings(out_dir, **updates):
    path = out_dir / "timings.json"
    timings = json.loads(path.read_text()) if path.exists() else {}
    timings.update({k: round(v, 3) for k, v in updates.items()})
    path.write_text(json.dumps(timings, indent=2) + "\n")


def _content_hash(path):
    """SHA-256 of the resolved data file's bytes (streamed)."""

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _data_regime(data_path):
    """Regime/lead label for `data_path`, read from its converter `_meta.json`.

    Forecast leads carry `lead_hours` (and friends) in the sidecar meta written
    by convert_real_gmm_pt_to_npz.py; the legacy AE files do not, so they fall
    back to the reconstruction-regime label.
    """

    data_path = Path(data_path)
    meta_path = data_path.with_name(data_path.stem + "_meta.json")
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except (ValueError, OSError):
            meta = {}
    if "lead_hours" in meta:
        label = {"regime": "forecast", "lead_hours": meta["lead_hours"]}
        for key in (
            "valid_datetime", "init_datetime", "forecast_step", "from_run_id",
            "mini_epoch", "norm_mean_channel", "norm_std_channel",
        ):
            if meta.get(key) is not None:
                label[key] = meta[key]
        return label
    return {"regime": "reconstruction_step0"}


def ensure_provenance(out_dir, data_path):
    """Stamp/verify the out-dir against `data_path` (path + content hash + regime).

    The first stage to run in a fresh out-dir writes provenance.json; every
    later stage re-checks it and aborts on mismatch, so step-0 anchors can never
    be silently mixed with a forecast-lead sweep (or vice versa). Returns the
    regime label dict for stamping into downstream artifacts.
    """

    data_path = Path(data_path).resolve()
    regime = _data_regime(data_path)
    record = {
        "data_path": str(data_path),
        "data_sha256": _content_hash(data_path),
        "regime": regime,
    }
    path = out_dir / "provenance.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("data_sha256") != record["data_sha256"]:
            sys.exit(
                f"[provenance] out-dir {out_dir} was built from\n"
                f"  {existing.get('data_path')} (sha256 {existing.get('data_sha256', '')[:12]})\n"
                f"but --data is\n"
                f"  {record['data_path']} (sha256 {record['data_sha256'][:12]}).\n"
                "Refusing to mix regimes in one out-dir; use a fresh --out-dir per lead."
            )
        return existing.get("regime", regime)
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"[provenance] stamped {path} (regime={regime}, sha256={record['data_sha256'][:12]})")
    return regime


def stage_graph(data, out_dir, args):
    # The k-NN graph (edges + arc lengths) depends only on latlons, which are
    # identical across all forecast leads. A --graph-cache lets every lead dir
    # reuse one computed graph instead of recomputing it 8 times; the stratum
    # masks below still depend on the per-lead GMM params and are recomputed.
    cache = getattr(args, "graph_cache", None)
    if cache is not None and Path(cache).exists():
        with np.load(cache) as f:
            edges, arc = f["edges"], f["edge_arc_km"]
        elapsed = 0.0
        print(f"[graph] reused cached k-NN graph from {cache}")
    else:
        t0 = time.perf_counter()
        edges = knn_sphere_edges(data["latlons"], k=8)
        arc = edge_arc_km(data["latlons"], edges)
        elapsed = time.perf_counter() - t0
        if cache is not None:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, edges=edges, edge_arc_km=arc)
            print(f"[graph] cached k-NN graph to {cache}")
    print(f"[graph] |E| = {len(edges)} (audit S5 expects 162,406 at k=8)")
    print(
        f"[graph] edge arc km min/median/max = "
        f"{arc.min():.1f}/{np.median(arc):.1f}/{arc.max():.1f}"
    )
    masks = stratum_masks(data["pi"], data["mu"], data["sigma"], data["latlons"])
    masks["bimodal_1sigma"] = practically_bimodal_mask(
        data["pi"], data["mu"], data["sigma"], min_separation=1.0
    )
    for name, mask in masks.items():
        print(f"[graph] stratum {name}: {int(mask.sum())} cells")
    np.savez(out_dir / "masks.npz", edges=edges, edge_arc_km=arc, **masks)
    _update_timings(out_dir, knn_graph_s=elapsed)
    print(f"[graph] wrote {out_dir / 'masks.npz'}")


def _load_masks(out_dir):
    with np.load(out_dir / "masks.npz") as f:
        return {k: f[k] for k in f.files}


def stage_anchors(data, out_dir, args):
    pi, mu, sigma = data["pi"], data["mu"], data["sigma"]
    masks = _load_masks(out_dir)
    edges = masks["edges"]
    lap = sparse_laplacian(pi.shape[0], edges)

    t0 = time.perf_counter()
    fields = {}
    fields["mode_map"], _ = mode_field(pi, mu, sigma)
    fields["mixture_mean"] = mixture_mean_field(pi, mu)
    for seed in IID_SEEDS[: 1 if args.quick else len(IID_SEEDS)]:
        sample, _ = sample_iid_gmm(pi, mu, sigma, np.random.default_rng(seed))
        fields[f"iid_seed{seed}"] = sample
    for n_iters in SMOOTH_N_ITERS:
        fields[f"smoothed_map_n{n_iters}"] = smoothed_map_baseline(
            pi, mu, sigma, step=SMOOTH_STEP, n_iters=n_iters, laplacian=lap
        )
    elapsed = time.perf_counter() - t0

    # Anchor-table continuity DIAGNOSTIC (phase_4_plan S5): with near-one-hot pi
    # the mixture mean nearly coincides with the MAP field. At step 0 this gap is
    # tiny (< 0.02); under forecast-policy GMMs the pi spread, so the gap GROWING
    # is the expected finding, not an error. We record the value and never abort.
    nll_map = float(np.mean(gmm_nll_per_cell(fields["mode_map"], pi, mu, sigma)))
    nll_mean = float(np.mean(gmm_nll_per_cell(fields["mixture_mean"], pi, mu, sigma)))
    gap = abs(nll_mean - nll_map)
    near_one_hot = gap < 0.02
    print(
        f"[anchors] NLL/N(MAP) = {nll_map:.4f}, NLL/N(mixture_mean) = {nll_mean:.4f}, "
        f"|gap| = {gap:.4f} -> near-one-hot regime signature "
        f"{'PRESENT' if near_one_hot else 'ABSENT'} (gap {'<' if near_one_hot else '>='} 0.02)"
    )
    anchors_meta = {
        "nll_over_n_map": nll_map,
        "nll_over_n_mixture_mean": nll_mean,
        "nll_gap_mean_minus_map": gap,
        "near_one_hot_signature": bool(near_one_hot),
        "regime": getattr(args, "regime", None),
    }
    (out_dir / "anchors_meta.json").write_text(json.dumps(anchors_meta, indent=2) + "\n")

    np.savez(out_dir / "anchors.npz", **fields)
    _update_timings(out_dir, anchors_s=elapsed)
    print(f"[anchors] wrote {out_dir / 'anchors.npz'} ({len(fields)} fields)")


def stage_modes(data, out_dir, args):
    pi, mu, sigma = data["pi"], data["mu"], data["sigma"]
    t0 = time.perf_counter()
    ext = extract_gmm_modes(pi, mu, sigma)
    elapsed = time.perf_counter() - t0
    counts = np.bincount(ext.mode_counts, minlength=5)
    print(f"[modes] extraction {elapsed:.1f} s; mode-count census 1/2/3/4 = {counts[1:].tolist()}")
    np.savez(
        out_dir / "modes.npz",
        mode_values=ext.mode_values,
        valid_mask=ext.valid_mask,
        mode_unary=ext.mode_unary,
        mode_counts=ext.mode_counts,
    )
    _update_timings(out_dir, mode_extraction_s=elapsed)
    print(f"[modes] wrote {out_dir / 'modes.npz'}")


def _load_extraction(out_dir):
    with np.load(out_dir / "modes.npz") as f:
        return ModeExtraction(
            mode_values=f["mode_values"],
            valid_mask=f["valid_mask"],
            mode_unary=f["mode_unary"],
            mode_counts=f["mode_counts"],
        )


def _solve_lambda(pi, mu, sigma, lam, edges, lap, seed, n_steps, n_restarts,
                  restart_scale=RESTART_SCALE):
    """Sanity-guarded Method 1 solve: halve lr once on divergence, never more."""

    warm, _ = mode_field(pi, mu, sigma)
    n_edges = len(edges)
    warm_energy = objective(warm, pi, mu, sigma, lam, laplacian=lap, n_edges=n_edges)
    lr = LR
    for attempt in range(2):
        res = minimise_at_lambda(
            pi, mu, sigma, lam,
            n_restarts=n_restarts, n_steps=n_steps, lr=lr,
            restart_scale=restart_scale, rng=np.random.default_rng(seed),
            laplacian=lap, edges=edges,
        )
        sane = bool(np.isfinite(res.energy) and res.energy <= warm_energy + 1e-9)
        if sane:
            break
        if attempt == 0:
            print(f"[m1] lambda={lam:g}: energy above warm start; halving lr once")
            lr = LR / 2.0
    return res, sane, lr


def _target_r_tilde(out_dir, edges):
    with np.load(out_dir / "anchors.npz") as f:
        target_field = f[TARGET_ANCHOR]
    target, collapsed = scale_free_roughness(target_field, edges)
    if collapsed:
        sys.exit(f"[m1] {TARGET_ANCHOR} is variance-collapsed: lambda* target undefined (hard stop)")
    return target


def stage_m1(data, out_dir, args):
    pi, mu, sigma = data["pi"], data["mu"], data["sigma"]
    masks = _load_masks(out_dir)
    edges = masks["edges"]
    lap = sparse_laplacian(pi.shape[0], edges)
    n_steps = 50 if args.quick else N_STEPS
    n_restarts = 2 if args.quick else N_RESTARTS

    target = _target_r_tilde(out_dir, edges)
    with np.load(out_dir / "anchors.npz") as f:
        print("[m1] anchor R-tilde:")
        for name in f.files:
            r, c = scale_free_roughness(f[name], edges)
            print(f"[m1]   {name}: R~ = {r:.5f}{' (collapsed)' if c else ''}")

    # Bracket-check guard (phase_4_plan S5): coarse probe, abort if the
    # production grid cannot bracket the smoothed-MAP target.
    probe_r = []
    for i, lam in enumerate(PROBE_LAMBDAS):
        res, _, _ = _solve_lambda(pi, mu, sigma, lam, edges, lap, 1000 + i,
                                  n_steps=max(50, n_steps // 2), n_restarts=2,
                                  restart_scale=args.restart_scale)
        probe_r.append(res.r_tilde)
        print(f"[m1] probe lambda={lam:g}: NLL/N={res.nll_over_n:.4f} R~={res.r_tilde:.5f}")
    bracket_ok = min(probe_r) <= target <= max(probe_r)
    print(f"[m1] target R~({TARGET_ANCHOR}) = {target:.5f}; probe bracket ok = {bracket_ok}")
    if not bracket_ok and not args.quick:
        sys.exit(
            "[m1] BRACKET GUARD: production grid does not bracket the smoothed-MAP "
            f"R~ target {target:.5f} (probe range {min(probe_r):.5f}..{max(probe_r):.5f}); aborting"
        )

    lambdas = (0.0, 0.5) if args.quick else LAMBDA_GRID
    rows = {key: [] for key in (
        "energy", "nll_over_n", "r_tilde", "variance_collapsed", "restart_spread",
        "restart_field_spread", "restart_nll_min", "restart_nll_max", "lr_used",
        "warm_start_sane", "wall_s",
    )}
    fields = []
    t_sweep = time.perf_counter()
    for idx, lam in enumerate(lambdas):
        t0 = time.perf_counter()
        res, sane, lr_used = _solve_lambda(
            pi, mu, sigma, lam, edges, lap, args.seed + idx, n_steps, n_restarts,
            restart_scale=args.restart_scale,
        )
        wall = time.perf_counter() - t0
        fields.append(res.field)
        rows["energy"].append(res.energy)
        rows["nll_over_n"].append(res.nll_over_n)
        rows["r_tilde"].append(res.r_tilde)
        rows["variance_collapsed"].append(res.variance_collapsed)
        rows["restart_spread"].append(res.restart_spread)
        rows["restart_field_spread"].append(res.restart_field_spread)
        rows["restart_nll_min"].append(float(res.restart_nll_over_n.min()))
        rows["restart_nll_max"].append(float(res.restart_nll_over_n.max()))
        rows["lr_used"].append(lr_used)
        rows["warm_start_sane"].append(sane)
        rows["wall_s"].append(wall)
        print(
            f"[m1] lambda={lam:g}: NLL/N={res.nll_over_n:.4f} R~={res.r_tilde:.5f} "
            f"collapsed={res.variance_collapsed} sane={sane} ({wall:.1f} s)"
        )

    np.savez(
        out_dir / "method1_sweep.npz",
        lambdas=np.asarray(lambdas, dtype=float),
        fields=np.stack(fields),
        probe_lambdas=np.asarray(PROBE_LAMBDAS, dtype=float),
        probe_r_tilde=np.asarray(probe_r, dtype=float),
        target_r_tilde=np.float64(target),
        restart_scale=np.float64(args.restart_scale),
        **{k: np.asarray(v) for k, v in rows.items()},
    )
    print(f"[m1] restart_scale = {args.restart_scale:.4f} (median sigma of loaded data)")
    _update_timings(out_dir, m1_sweep_s=time.perf_counter() - t_sweep)
    print(f"[m1] wrote {out_dir / 'method1_sweep.npz'}")


def run_lambda_star(data, out_dir, args):
    pi, mu, sigma = data["pi"], data["mu"], data["sigma"]
    masks = _load_masks(out_dir)
    edges = masks["edges"]
    lap = sparse_laplacian(pi.shape[0], edges)
    target = _target_r_tilde(out_dir, edges)
    n_steps = 50 if args.quick else N_STEPS
    n_restarts = 2 if args.quick else N_RESTARTS

    sweep_path = out_dir / "method1_sweep.npz"
    if not sweep_path.exists():
        sys.exit(
            "[lambda*] needs the m1 + anchors stages first; "
            "run --stages graph,anchors,modes,m1"
        )
    sweep = dict(np.load(sweep_path))
    extended = 0
    while True:
        try:
            sel = select_lambda_star(
                sweep["lambdas"], sweep["r_tilde"], sweep["variance_collapsed"],
                target, valid=sweep["warm_start_sane"],
            )
        except ValueError:
            # No valid positive-lambda row at all: report the smallest-lambda
            # row as lambda* and flag the mismatch (S6 final fallback).
            positive = sweep["lambdas"] > 0
            idx = int(np.argmin(np.where(positive, sweep["lambdas"], np.inf)))
            sel = None
            lambda_star = float(sweep["lambdas"][idx])
            record = {
                "lambda_star": lambda_star,
                "target_r_tilde": float(target),
                "matched_r_tilde": float(sweep["r_tilde"][idx]),
                "bracketed": False,
                "extend_direction": None,
                "clauses": ["no_valid_rows_smallest_lambda_fallback"],
            }
            break
        if sel.bracketed or extended >= 3:
            lambda_star = sel.lambda_star
            record = {
                "lambda_star": sel.lambda_star,
                "target_r_tilde": sel.target_r_tilde,
                "matched_r_tilde": sel.matched_r_tilde,
                "bracketed": sel.bracketed,
                "extend_direction": sel.extend_direction,
                "clauses": sel.clauses,
            }
            break
        # Decade-by-decade extension (S6 clause d), persisted back into the sweep.
        extended += 1
        base = sweep["lambdas"].max() if sel.extend_direction == "up" else (
            sweep["lambdas"][sweep["lambdas"] > 0].min()
        )
        factor = 10.0 if sel.extend_direction == "up" else 0.1
        new_lams = [base * factor ** 0.5, base * factor]
        print(f"[lambda*] unbracketed ({sel.extend_direction}); extending with {new_lams}")
        for lam in new_lams:
            res, sane, lr_used = _solve_lambda(
                pi, mu, sigma, lam, edges, lap, args.seed + 100 + extended, n_steps, n_restarts,
                restart_scale=args.restart_scale,
            )
            sweep["lambdas"] = np.append(sweep["lambdas"], lam)
            sweep["fields"] = np.vstack([sweep["fields"], res.field[None]])
            sweep["energy"] = np.append(sweep["energy"], res.energy)
            sweep["nll_over_n"] = np.append(sweep["nll_over_n"], res.nll_over_n)
            sweep["r_tilde"] = np.append(sweep["r_tilde"], res.r_tilde)
            sweep["variance_collapsed"] = np.append(
                sweep["variance_collapsed"], res.variance_collapsed
            )
            sweep["restart_spread"] = np.append(sweep["restart_spread"], res.restart_spread)
            sweep["restart_field_spread"] = np.append(
                sweep["restart_field_spread"], res.restart_field_spread
            )
            sweep["restart_nll_min"] = np.append(
                sweep["restart_nll_min"], res.restart_nll_over_n.min()
            )
            sweep["restart_nll_max"] = np.append(
                sweep["restart_nll_max"], res.restart_nll_over_n.max()
            )
            sweep["lr_used"] = np.append(sweep["lr_used"], lr_used)
            sweep["warm_start_sane"] = np.append(sweep["warm_start_sane"], sane)
            sweep["wall_s"] = np.append(sweep["wall_s"], np.nan)
        np.savez(sweep_path, **sweep)

    print(
        f"[lambda*] lambda* = {record['lambda_star']:g} "
        f"(target R~ {record['target_r_tilde']:.5f}, bracketed={record['bracketed']}, "
        f"clauses={record['clauses']})"
    )

    sensitivity = {}
    sens_fields = {}
    for tag, lam in (("half", lambda_star / 2), ("star", lambda_star), ("double", 2 * lambda_star)):
        res, sane, lr_used = _solve_lambda(
            pi, mu, sigma, lam, edges, lap, args.seed + 200, n_steps, n_restarts,
            restart_scale=args.restart_scale,
        )
        sens_fields[f"field_{tag}"] = res.field
        sensitivity[tag] = {
            "lambda": lam,
            "nll_over_n": res.nll_over_n,
            "r_tilde": res.r_tilde,
            "variance_collapsed": bool(res.variance_collapsed),
            "restart_nll_min": float(res.restart_nll_over_n.min()),
            "restart_nll_max": float(res.restart_nll_over_n.max()),
            "warm_start_sane": sane,
            "lr_used": lr_used,
        }
        print(
            f"[lambda*] {tag}: lambda={lam:g} NLL/N={res.nll_over_n:.4f} "
            f"R~={res.r_tilde:.5f} collapsed={res.variance_collapsed}"
        )

    record["target_anchor"] = TARGET_ANCHOR
    record["sensitivity"] = sensitivity
    record["regime"] = getattr(args, "regime", None)
    record["restart_scale"] = float(args.restart_scale)
    (out_dir / "lambda_star.json").write_text(json.dumps(record, indent=2) + "\n")
    np.savez(
        out_dir / "method1_sensitivity.npz",
        lambdas=np.asarray([lambda_star / 2, lambda_star, 2 * lambda_star]),
        **sens_fields,
    )
    print(f"[lambda*] wrote {out_dir / 'lambda_star.json'} and method1_sensitivity.npz")


def stage_m4(data, out_dir, args):
    pi, mu, sigma = data["pi"], data["mu"], data["sigma"]
    masks = _load_masks(out_dir)
    edges = masks["edges"]
    ext = _load_extraction(out_dir)
    betas = BETAS[:2] if args.quick else BETAS
    max_sweeps = 3 if args.quick else M4_MAX_SWEEPS

    t0 = time.perf_counter()
    points = beta_sweep(
        pi, mu, sigma, betas,
        n_restarts=M4_RESTARTS, seed=args.seed, extraction=ext,
        max_sweeps=max_sweeps, edges=edges,
    )
    elapsed = time.perf_counter() - t0
    total_sweeps = int(sum(p.restart_n_sweeps.sum() for p in points))
    s_per_sweep = elapsed / max(total_sweeps, 1)
    for p in points:
        print(
            f"[m4] beta={p.beta:.4g}: NLL/N={p.nll_over_n:.4f} R~={p.r_tilde:.5f} "
            f"dNLL frac>0.125={p.delta_to_mode['frac_over'][0]:.4f} "
            f"sweeps={p.restart_n_sweeps.tolist()}"
        )
    print(f"[m4] {total_sweeps} ICM sweeps in {elapsed:.0f} s -> {s_per_sweep:.2f} s/sweep")

    # Non-fatal grid-health monitors (no behaviour change). The step-0 sweep is
    # pinned by near-one-hot unary gaps; under spread-pi forecast GMMs we expect
    # it to move, so flag if it is still pinned or if no beta reaches the
    # smoothed-MAP R-tilde target (the matched-coherence operating point).
    sweep_fields = np.stack([p.field for p in points])
    field_span = float(np.max(np.ptp(sweep_fields, axis=0))) if len(points) > 1 else 0.0
    delta_span = float(np.ptp([p.nll_over_n for p in points])) if len(points) > 1 else 0.0
    if field_span <= 1e-9 or delta_span <= 1e-9:
        print(
            f"[m4] WARNING: beta sweep appears PINNED (max field span {field_span:.2e}, "
            f"NLL/N span {delta_span:.2e} across betas {betas[0]:g}..{betas[-1]:g}); "
            "every beta returns essentially the same field. Expected at step 0; "
            "if seen on forecast GMMs, the grid may need extending."
        )
    try:
        target = _target_r_tilde(out_dir, edges)
        r_tildes = np.asarray([p.r_tilde for p in points])
        nearest = float(np.min(np.abs(r_tildes - target)))
        if nearest > 0.1 * abs(target):
            print(
                f"[m4] WARNING: no beta row approaches the smoothed-MAP R~ target "
                f"{target:.5f} (nearest M4 R~ off by {nearest:.5f}, "
                f">10% rel); consider extending the beta grid for a matched-R~ comparison."
            )
    except SystemExit:
        # _target_r_tilde hard-stops only on a collapsed target; leave that to
        # the Method 1 stage and skip the R~-bracket monitor here.
        print("[m4] (smoothed-MAP target collapsed; skipping R~-bracket monitor)")

    np.savez(
        out_dir / "method4_sweep.npz",
        betas=np.asarray([p.beta for p in points]),
        fields=np.stack([p.field for p in points]),
        assignments=np.stack([p.assignment for p in points]),
        energy=np.asarray([p.energy for p in points]),
        nll_over_n=np.asarray([p.nll_over_n for p in points]),
        r_tilde=np.asarray([p.r_tilde for p in points]),
        variance_collapsed=np.asarray([p.variance_collapsed for p in points]),
        restart_spread=np.asarray([p.restart_spread for p in points]),
        n_sweeps_max=np.asarray([int(p.restart_n_sweeps.max()) for p in points]),
        wall_s=np.float64(elapsed),
    )
    _update_timings(out_dir, m4_sweep_s=elapsed, icm_s_per_sweep=s_per_sweep)
    print(f"[m4] wrote {out_dir / 'method4_sweep.npz'}")


def _collect_fields(out_dir):
    """All persisted fields keyed by name, with (kind, param) metadata."""

    fields = {}
    with np.load(out_dir / "anchors.npz") as f:
        for name in f.files:
            param = name.split("_n")[-1] if name.startswith("smoothed") else (
                name.split("seed")[-1] if name.startswith("iid") else ""
            )
            kind = "anchor"
            fields[name] = (f[name], kind, param)
    with np.load(out_dir / "method1_sweep.npz") as f:
        for lam, field in zip(f["lambdas"], f["fields"]):
            fields[f"m1_lam{lam:g}"] = (field, "method1", f"{lam:g}")
    sens = out_dir / "method1_sensitivity.npz"
    if sens.exists():
        with np.load(sens) as f:
            for tag, lam in zip(("half", "star", "double"), f["lambdas"]):
                name = "m1_star" if tag == "star" else f"m1_star_{tag}"
                fields[name] = (f[f"field_{tag}"], "method1_sensitivity", f"{lam:g}")
    m4 = out_dir / "method4_sweep.npz"
    if m4.exists():
        with np.load(m4) as f:
            for beta, field in zip(f["betas"], f["fields"]):
                fields[f"m4_beta{beta:.4g}"] = (field, "method4", f"{beta:.4g}")
    return fields


def _maybe_load_era5_reference(regime, latlons, out_dir, args):
    """ERA5 rung-3 direction-of-realism reference, or None on ANY failure.

    Never raises into the runner (cut criterion C4, spectrum_era5_plan S8): on
    --no-era5, a missing valid_datetime, missing data access, or a grid mismatch,
    it logs `[scores] ERA5 reference skipped: <reason>` and returns None so the
    spectrum and variogram still ship sample-only. ERA5 is a *direction*
    reference among siblings sharing the decoder mean -- never a target, never
    validation.

    Resolution order: a pre-computed `era5_reference.npz` in the run dir (dumped
    by load_era5_reference.py under the WeatherGenerator venv, so the sampler venv
    never needs the anemoi/zarr stack) is preferred; otherwise the loader is
    called live (requires anemoi + the zarr reachable in this interpreter).
    """
    if getattr(args, "no_era5", False):
        print("[scores] ERA5 reference skipped: --no-era5")
        return None
    cached = Path(out_dir) / "era5_reference.npz"
    if cached.exists():
        with np.load(cached) as f:
            z = np.asarray(f["era5"], dtype=float).reshape(-1)
        n = np.asarray(latlons).shape[0]
        if z.shape[0] != n:
            print(f"[scores] ERA5 reference skipped: cached era5_reference.npz has "
                  f"{z.shape[0]} points, expected {n}")
            return None
        print(f"[scores] ERA5 reference loaded from cache {cached.name}")
        return z
    valid_dt = (regime or {}).get("valid_datetime")
    if not valid_dt:
        print("[scores] ERA5 reference skipped: no valid_datetime in this regime")
        return None
    try:
        import importlib.util

        loader_path = Path(__file__).resolve().parent / "load_era5_reference.py"
        spec = importlib.util.spec_from_file_location("load_era5_reference", loader_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.load_era5_2t_on_o96(valid_dt, latlons, meta_dir=out_dir)
    except Exception as exc:  # noqa: BLE001 - any failure -> sample-only spectrum
        print(f"[scores] ERA5 reference skipped: {exc}")
        return None


# Headline field set shared by the variogram and the spectrum: the iid/MAP/mean
# baselines, the Joint MAP at lambda*, and (when the sweep exists) the Method 4
# row nearest to lambda* in roughness. Kept as a helper so stage_scores and the
# stand-alone spectrum stage select the SAME curves.
_SPECTRUM_BASE_FIELDS = ("iid_seed0", "mode_map", "mixture_mean",
                         "smoothed_map_n10", "m1_star")


def _read_scores_rtilde(out_dir):
    """{name: r_tilde} from the persisted scores.csv (None if absent)."""
    path = Path(out_dir) / "scores.csv"
    if not path.exists():
        return None
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                out[row["name"]] = float(row["r_tilde"])
            except (KeyError, ValueError):
                continue
    return out


def _headline_field_names(fields, rtilde):
    """Ordered headline curve names for the variogram/spectrum. Appends the
    Method 4 sweep row nearest to m1_star in roughness when both are available,
    matching stage_scores' original selection."""
    names = [n for n in _SPECTRUM_BASE_FIELDS if n in fields]
    m4 = {n: (f, kind, param) for n, (f, kind, param) in fields.items()
          if kind == "method4"}
    if m4 and rtilde and "m1_star" in rtilde:
        star_rt = rtilde["m1_star"]
        nearest = min((n for n in m4 if n in rtilde),
                      key=lambda n: abs(rtilde[n] - star_rt), default=None)
        if nearest is not None:
            names.append(nearest)
    return names


def _maybe_budget_field(out_dir):
    """The W3/DEC-R46 faithfulness-budget Joint MAP field, or None.

    Single-sourced from budget_point.npz (scripts/run_budget_point.py) so the
    +48h angular power spectrum shows the SAME field the maps figure adds as its
    faithfulness-budget panel. Spectrum-only: the variogram is retired from the
    report, so the budget curve is not added to it.
    """
    path = Path(out_dir) / "budget_point.npz"
    if not path.exists():
        return None
    with np.load(path) as f:
        return np.asarray(f["field"], dtype=float).reshape(-1)


def _write_spectra(out_dir, latlons, spec_fields):
    """Compute and persist the native O96 angular power spectrum for spec_fields.
    Returns (lmax_resolved, elapsed_s)."""
    t0 = time.perf_counter()
    ell, spectra, lmax_resolved = sampled_spherical_power_spectrum(latlons, spec_fields)
    np.savez(out_dir / "spectra.npz", ell=ell, lmax_resolved=lmax_resolved, **spectra)
    return int(lmax_resolved), time.perf_counter() - t0


def _assemble_spectrum_fields(out_dir, latlons, args):
    """Full spectrum curve set from the persisted artifacts: headline fields +
    ERA5 direction reference (if available) + the W3 budget curve (if solved).
    Used by the stand-alone `spectrum` stage."""
    fields = _collect_fields(out_dir)
    rtilde = _read_scores_rtilde(out_dir)
    spec_fields = {n: fields[n][0] for n in _headline_field_names(fields, rtilde)}
    era5 = _maybe_load_era5_reference(getattr(args, "regime", None), latlons, out_dir, args)
    if era5 is not None:
        spec_fields["era5"] = era5
    budget = _maybe_budget_field(out_dir)
    if budget is not None:
        spec_fields["m1_budget"] = budget
    return spec_fields


def stage_spectrum(data, out_dir, args):
    """Regenerate ONLY spectra.npz from the persisted method fields (no re-scoring,
    no variogram resampling). Refreshes the angular power spectrum's curve set --
    e.g. picking up the W3 faithfulness-budget curve once budget_point.npz exists."""
    latlons = data["latlons"]
    spec_fields = _assemble_spectrum_fields(out_dir, latlons, args)
    lmax_resolved, _ = _write_spectra(out_dir, latlons, spec_fields)
    print(f"[spectrum] wrote spectra.npz (resolved l<= {lmax_resolved}; "
          f"{len(spec_fields)} curves: {', '.join(spec_fields)})")


def stage_scores(data, out_dir, args):
    pi, mu, sigma, latlons = data["pi"], data["mu"], data["sigma"], data["latlons"]
    masks_all = _load_masks(out_dir)
    edges = masks_all["edges"]
    from sampler_research.phase4_eval import STRATUM_ORDER

    masks = {name: masks_all[name] for name in STRATUM_ORDER}
    with np.load(out_dir / "modes.npz") as f:
        mode_values, valid_mask = f["mode_values"], f["valid_mask"]

    regime = getattr(args, "regime", None) or {}
    regime_label = regime.get("regime", "unknown")
    lead_hours = regime.get("lead_hours", "")

    fields = _collect_fields(out_dir)
    t0 = time.perf_counter()
    deltas = {}
    rows = []
    for name, (field, kind, param) in fields.items():
        delta = delta_nll_to_best_mode(field, pi, mu, sigma, mode_values, valid_mask)
        deltas[name] = delta["per_cell"]
        nll = gmm_nll_per_cell(field, pi, mu, sigma)
        row = {"name": name, "kind": kind, "param": param,
               "regime": regime_label, "lead_hours": lead_hours}
        row.update(stratified_scores(field, nll, edges, masks, delta["per_cell"]))
        row["wrap_seam_ratio"] = wrap_seam_ratio(field, latlons, edges)
        rows.append(row)
    np.savez(out_dir / "delta_per_cell.npz", **deltas)
    scores_s = time.perf_counter() - t0

    fieldnames = list(rows[0].keys())
    with open(out_dir / "scores.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def _rt(row, suffix=""):
        if row[f"variance_collapsed{suffix}"]:
            return "collapsed"
        return f"{row[f'r_tilde{suffix}']:.5f}"

    lines = [
        "| name | param | NLL/N | R~ | S_edge | dNLL mean | frac>0.125 | "
        "NLL/N (bimodal) | frac>0.125 (bimodal) | wrap seam |",
        "| " + " | ".join(["---"] * 10) + " |",
    ]
    for row in rows:
        lines.append(
            "| {name} | {param} | {nll:.4f} | {rt} | {se:.5f} | {dm:.4f} | {fr:.4f} | "
            "{nb:.4f} | {fb:.4f} | {ws:.2f} |".format(
                name=row["name"], param=row["param"], nll=row["nll_over_n"],
                rt=_rt(row), se=row["s_edge"], dm=row["dnll_mean"],
                fr=row["dnll_frac_gt_0p125"], nb=row["nll_over_n__bimodal"],
                fb=row["dnll_frac_gt_0p125__bimodal"], ws=row["wrap_seam_ratio"],
            )
        )
    (out_dir / "scores.md").write_text("\n".join(lines) + "\n")

    # Variograms for the headline fields (60k pairs, diagnostics.py convention).
    t0 = time.perf_counter()
    rtilde = {r["name"]: r["r_tilde"] for r in rows}
    headline = _headline_field_names(fields, rtilde)
    vario_fields = {n: fields[n][0] for n in headline if n in fields}

    # Optional ERA5 rung-3 direction reference, added to both the variogram and
    # the spectrum (the same diagnostic two ways). None on any gate failure ->
    # both ship sample-only (cut C4).
    era5 = _maybe_load_era5_reference(regime, latlons, out_dir, args)
    vario_fields_with_era5 = dict(vario_fields)
    spec_fields = dict(vario_fields)
    if era5 is not None:
        vario_fields_with_era5["era5"] = era5
        spec_fields["era5"] = era5

    n_pairs = 10_000 if args.quick else VARIOGRAM_PAIRS
    centres, variograms, _ = sampled_spherical_variogram(
        latlons, vario_fields_with_era5, n_pairs=n_pairs
    )
    np.savez(out_dir / "variograms.npz", centres=centres, **variograms)
    variograms_s = time.perf_counter() - t0

    # The W3/DEC-R46 faithfulness-budget curve joins the spectrum (only) when its
    # solve exists, matching the maps figure's budget panel (single-sourced from
    # budget_point.npz). The spectrum is the report's coherence figure; the
    # variogram is retired, so the curve is not added there.
    budget = _maybe_budget_field(out_dir)
    if budget is not None:
        spec_fields["m1_budget"] = budget

    # Native O96 angular power spectrum (rung-3 bracket diagnostic;
    # spectrum_era5_plan.md). Default engine is the validated pure-numpy SHT.
    lmax_resolved, spectra_s = _write_spectra(out_dir, latlons, spec_fields)
    _update_timings(
        out_dir, scores_s=scores_s, variograms_s=variograms_s, spectra_s=spectra_s,
    )
    print(f"[scores] wrote scores.csv/.md, delta_per_cell.npz, variograms.npz, "
          f"spectra.npz (resolved l<= {lmax_resolved}; {len(rows)} fields, "
          f"era5={'yes' if era5 is not None else 'no'}, "
          f"budget={'yes' if budget is not None else 'no'})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", type=str, default=",".join(DEFAULT_STAGES),
                        help=f"comma-separated subset of {STAGES} (default: "
                             f"{','.join(DEFAULT_STAGES)}; `spectrum` is opt-in)")
    parser.add_argument("--lambda-star", action="store_true",
                        help="apply the S6 lambda* rule + sensitivity (needs anchors+m1)")
    parser.add_argument("--quick", action="store_true", help="smoke mode (reduced settings)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", type=Path, default=DATA_NPZ)
    parser.add_argument("--out-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--graph-cache", type=Path, default=None,
                        help="shared k-NN graph npz reused across leads (same latlons); "
                             "computed and cached on first use")
    parser.add_argument("--no-era5", action="store_true",
                        help="force sample-only spectrum/variogram (skip the ERA5 "
                             "rung-3 direction reference even if data access exists)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = load_real_marginal(args.data)
    print(f"loaded {args.data}: N={data['pi'].shape[0]}, K={data['pi'].shape[1]}")

    # Provenance stamp/verify (prevents silently mixing regimes in one out-dir)
    # and the regime/lead label carried into anchors/scores/lambda_star artifacts.
    args.regime = ensure_provenance(args.out_dir, args.data)
    # Data-derived Method 1 restart jitter (median sigma); AE data ~ 0.148.
    args.restart_scale = float(np.median(data["sigma"]))
    print(f"regime={args.regime}; restart_scale={args.restart_scale:.4f} (median sigma)")

    if args.lambda_star:
        run_lambda_star(data, args.out_dir, args)
        return

    requested = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = set(requested) - set(STAGES)
    if unknown:
        sys.exit(f"unknown stages: {sorted(unknown)}")
    stage_fns = {
        "graph": stage_graph, "anchors": stage_anchors, "modes": stage_modes,
        "m1": stage_m1, "m4": stage_m4, "scores": stage_scores,
        "spectrum": stage_spectrum,
    }
    for name in STAGES:
        if name in requested:
            stage_fns[name](data, args.out_dir, args)


if __name__ == "__main__":
    main()
