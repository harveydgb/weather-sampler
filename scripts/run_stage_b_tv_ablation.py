"""HISTORICAL (archived 11 Jun 2026) — toy TV min-cut certificate; job complete.

This Stage B-TV run is done and its arms are CUT from the real-data path per
phase_4_plan.md "Scope verdicts". The verdict it produced — exact TV collapses
rather than smears below R̃ ≈ 0.5 while the global quadratic mode still smears, so
the smear-vs-smoothness trade-off is structural and the carry-both read stands —
lives in large_notes.md (Phase 2 status); report in notebooks/03. Re-runnable on
the toy only.

Stage B-TV — three-arm TV ablation + exact-MAP certification of Method 1.

Run from the repo root with the local venv:

    .venv/bin/python scripts/run_stage_b_tv_ablation.py

This is the pre-registered §4.1 field-6 / §8 #8 penalty ablation of **Method 1**
(not a new method): Stage B/D showed the quadratic-penalty joint mode smears
~14% of cells >0.125 nats off-mode at its R̃≈0.25 operating point, and the
adversarial review argued that smearing is a property of the quadratic
*objective's mode*, not of the optimiser — a total-variation penalty is
spreading-neutral, so its mode should keep cells on modes. Three arms:

  * **adam-tv** — Method 1's existing Adam machinery with the Huber-smoothed
    TV penalty (`lambda_sweep(penalty="huber")`): does the penalty swap alone
    kill the smearing?
  * **cut-tv** — the exact discretised TV-MAP via Ishikawa min-cut
    (`exact_map_sweep(penalty="tv")`): the true TV frontier. Per lambda it adds
    a G-vs-2G value-grid control, a deterministic unary tie probe, and the
    certificate `ΔJ_TV = J_TV(adam) − J_TV(cut)` flagged against
    `restart floor + lambda*delta/2 (Huber surrogate gap) + quantisation bound`.
  * **cut-quad** — exact discretised quadratic-MAP at the persisted Stage B
    lambdas, Adam-polished into the continuum (Stage D pattern), giving
    `ΔJ = J_lambda(polished cut) − J_lambda(Stage B)` against the Stage B
    restart floor: did Stage B's Adam find the global basin, and does the
    *global* quadratic mode smear?

Both TV arms are scored with the shared `tv_objective` (true TV, not the Huber
surrogate) so the certificate compares like with like. Every selected field is
re-scored with `score_field` (the Stage A scorer) and asserted to agree, and
scored with Method 4's `delta_nll_to_best_mode` against the persisted Stage C
modes (the shared non-smearing reference). Debug arrays are never loaded;
Stage A–D artifacts are read-only inputs.

Outputs (`outputs/runs/stage_b_tv_ablation/`):
  * `stage_b_tv_scores.{csv,md}` — one wide table with an `arm` column
    (nan where a column does not apply to an arm),
  * `phase_1_homoscedastic_tv_ablation.npz` — arm-prefixed fields + controls.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from sampler_research.baselines import score_field
from sampler_research.exact_map import (
    exact_map_solve,
    exact_map_sweep,
    tv_objective,
    value_grid,
)
from sampler_research.graph import graph_laplacian, grid_edges_8
from sampler_research.io import load_sampler_arrays
from sampler_research.method4_mrf import delta_nll_to_best_mode
from sampler_research.regularised_map import (
    _adam_descent,
    lambda_sweep,
    objective,
    objective_gradient,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "outputs" / "data"
RUN_DIR = REPO_ROOT / "outputs" / "runs" / "stage_b_tv_ablation"
STAGE_B_DIR = REPO_ROOT / "outputs" / "runs" / "stage_b_regularised_map"
STAGE_C_DIR = REPO_ROOT / "outputs" / "runs" / "stage_c_method4_mrf"

# Same single-dataset evidence path as Stage A–D.
DATASETS = ("phase_1_homoscedastic",)

# TV-units lambda grid targeting R̃ ≈ 0.15–1.5; the same grid drives adam-tv
# and cut-tv so every certificate row is matched. (TV lambdas are not
# comparable 1:1 with Stage B's quadratic lambdas — different penalty units.)
DEFAULT_LAMBDAS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0)

ARMS = ("adam-tv", "cut-tv", "cut-quad")


def _load_stage_b(name):
    """Persisted Method 1 fields + per-lambda noise floor, or None if absent."""

    path = STAGE_B_DIR / f"{name}_regularised_map.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return {
        "lambdas": data["lambdas"],
        "fields": data["fields"],
        "nll_over_n": data["nll_over_n"],
        "r_tilde": data["r_tilde"],
        "restart_spread": data["restart_spread"],
    }


def _load_stage_c(name):
    """Persisted Stage C mode artifacts (shared non-smearing reference), or None."""

    path = STAGE_C_DIR / f"{name}_method4_mrf.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return {"mode_values": data["mode_values"], "valid_mask": data["valid_mask"]}


def _assert_rescore(field, pi, mu, sigma, nll_over_n, r_tilde, spectral_hf_ratio):
    """Stage A scorer re-score agreement (the cross-stage bracket guarantee)."""

    stage_a = score_field(field, pi, mu, sigma)
    assert np.isclose(stage_a["nll_over_n"], nll_over_n)
    assert np.isclose(stage_a["r_tilde"], r_tilde)
    assert np.isclose(stage_a["spectral_hf_ratio"], spectral_hf_ratio)


def _smear(field, pi, mu, sigma, stage_c):
    """ΔNLL→mode summary columns against the persisted Stage C modes."""

    if stage_c is None:
        return {
            "dnll_to_mode_mean": float("nan"),
            "dnll_to_mode_p95": float("nan"),
            "dnll_to_mode_max": float("nan"),
            "dnll_frac_gt_0p125_v1": float("nan"),
            "dnll_frac_gt_0p5_v1": float("nan"),
        }
    smear = delta_nll_to_best_mode(
        field, pi, mu, sigma, stage_c["mode_values"], stage_c["valid_mask"]
    )
    return {
        "dnll_to_mode_mean": smear["mean"],
        "dnll_to_mode_p95": smear["p95"],
        "dnll_to_mode_max": smear["max"],
        "dnll_frac_gt_0p125_v1": smear["frac_over"][0],
        "dnll_frac_gt_0p5_v1": smear["frac_over"][1],
    }


def _base_row(arm, name, lam, point):
    """Columns shared by all three arms (nan-filled extras added by callers)."""

    return {
        "arm": arm,
        "dataset": name,
        "lambda": lam,
        "nll_over_n": point.nll_over_n,
        "r_tilde": point.r_tilde,
        "variance_collapsed": point.variance_collapsed,
        "spectral_hf_ratio": point.spectral_hf_ratio,
        "spectral_slope": point.spectral_slope,
        "spectral_monotone_fraction": point.spectral_monotone_fraction,
        "spectral_collapsed": point.spectral_collapsed,
        "energy": point.energy,
        "j_tv": float("nan"),
        "restart_spread": float("nan"),
        "restart_field_spread": float("nan"),
        "cut_value": float("nan"),
        "quantisation_gap": float("nan"),
        "quantisation_bound": float("nan"),
        "gcontrol_max_dfield": float("nan"),
        "gcontrol_delta_j_tv": float("nan"),
        "tie_max_dfield": float("nan"),
        "delta_j_tv_adam_minus_cut": float("nan"),
        "certificate_allowance": float("nan"),
        "adam_matches_cut": "",
        "energy_decoded": float("nan"),
        "delta_j_vs_stage_b": float("nan"),
        "restart_floor": float("nan"),
        "beats_stage_b": "",
        "max_abs_dx_vs_stage_b": float("nan"),
    }


def _match_lambda(lam, stage_b):
    """Index of the Stage B row at lambda `lam`, or None if there is no match."""

    if stage_b is None:
        return None
    hits = np.where(np.isclose(stage_b["lambdas"], lam))[0]
    return int(hits[0]) if hits.size else None


CSV_FIELDS = [
    "arm",
    "dataset",
    "lambda",
    "nll_over_n",
    "r_tilde",
    "variance_collapsed",
    "spectral_hf_ratio",
    "spectral_slope",
    "spectral_monotone_fraction",
    "spectral_collapsed",
    "energy",
    "j_tv",
    "restart_spread",
    "restart_field_spread",
    "cut_value",
    "quantisation_gap",
    "quantisation_bound",
    "gcontrol_max_dfield",
    "gcontrol_delta_j_tv",
    "tie_max_dfield",
    "delta_j_tv_adam_minus_cut",
    "certificate_allowance",
    "adam_matches_cut",
    "energy_decoded",
    "delta_j_vs_stage_b",
    "restart_floor",
    "beats_stage_b",
    "max_abs_dx_vs_stage_b",
    "dnll_to_mode_mean",
    "dnll_to_mode_p95",
    "dnll_to_mode_max",
    "dnll_frac_gt_0p125_v1",
    "dnll_frac_gt_0p5_v1",
]


def write_csv(rows, path):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt_float(value, fmt="{:.4f}"):
    if isinstance(value, str):
        return value if value else "—"
    return "nan" if not np.isfinite(value) else fmt.format(value)


def write_markdown(rows, path):
    header = [
        "arm",
        "lambda",
        "NLL/N",
        "R̃",
        "energy (J)",
        "J_TV",
        "ΔJ_TV adam−cut",
        "allowance",
        "adam=cut?",
        "ΔJ vs StageB",
        "floor",
        "beats?",
        "ΔNLL→mode mean",
        ">0.125",
        ">0.5",
        "quant gap",
        "G-ctrl max|Δx|",
        "tie max|Δx|",
    ]
    lines = [
        "*Stage B-TV = Method 1 penalty ablation (Huber/TV swap + exact min-cut "
        "certificates), not a new method. TV lambdas are in TV units (not "
        "comparable 1:1 with Stage B's quadratic lambdas).*",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        lines.append(
            (
                "| {arm} | {lam:g} | {nll:.4f} | {rt} | {energy:.4f} | {jtv} | "
                "{djtv} | {allow} | {match} | {djb} | {floor} | {beats} | "
                "{dmean} | {f125} | {f5} | {qg} | {gc} | {tie} |"
            ).format(
                arm=row["arm"],
                lam=row["lambda"],
                nll=row["nll_over_n"],
                rt=("collapsed" if row["variance_collapsed"] else f"{row['r_tilde']:.4f}"),
                energy=row["energy"],
                jtv=_fmt_float(row["j_tv"]),
                djtv=_fmt_float(row["delta_j_tv_adam_minus_cut"], "{:+.5f}"),
                allow=_fmt_float(row["certificate_allowance"], "{:.5f}"),
                match=row["adam_matches_cut"] if row["adam_matches_cut"] != "" else "—",
                djb=_fmt_float(row["delta_j_vs_stage_b"], "{:+.5f}"),
                floor=_fmt_float(row["restart_floor"], "{:.5f}"),
                beats=row["beats_stage_b"] if row["beats_stage_b"] != "" else "—",
                dmean=_fmt_float(row["dnll_to_mode_mean"]),
                f125=_fmt_float(row["dnll_frac_gt_0p125_v1"], "{:.3f}"),
                f5=_fmt_float(row["dnll_frac_gt_0p5_v1"], "{:.3f}"),
                qg=_fmt_float(row["quantisation_gap"], "{:.2e}"),
                gc=_fmt_float(row["gcontrol_max_dfield"], "{:.4f}"),
                tie=_fmt_float(row["tie_max_dfield"], "{:.2e}"),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def run_adam_tv(name, pi, mu, sigma, stage_c, edges, n_edges, args):
    """Arm 1: Method 1's Adam machinery on the Huber-smoothed TV penalty."""

    points = lambda_sweep(
        pi,
        mu,
        sigma,
        args.lambdas,
        n_restarts=args.n_restarts,
        n_steps=args.n_steps,
        lr=args.lr,
        restart_scale=args.restart_scale,
        seed=args.seed,
        penalty="huber",
        delta=args.delta,
    )
    rows = []
    for p in points:
        _assert_rescore(p.field, pi, mu, sigma, p.nll_over_n, p.r_tilde, p.spectral_hf_ratio)
        row = _base_row("adam-tv", name, p.lam, p)
        row["j_tv"] = tv_objective(p.field, pi, mu, sigma, p.lam, edges=edges, n_edges=n_edges)
        row["restart_spread"] = p.restart_spread
        row["restart_field_spread"] = p.restart_field_spread
        row.update(_smear(p.field, pi, mu, sigma, stage_c))
        rows.append(row)
    return points, rows


def run_cut_tv(name, pi, mu, sigma, stage_c, edges, n_edges, adam_rows, args):
    """Arm 2: exact discretised TV-MAP + G-control, tie probe, certificate."""

    points = exact_map_sweep(
        pi,
        mu,
        sigma,
        args.lambdas,
        penalty="tv",
        n_levels=args.n_levels,
        n_sigma=args.n_sigma,
        edges=edges,
    )
    adam_by_lam = {row["lambda"]: row for row in adam_rows}
    rows = []
    controls = {"gcontrol_max_dfield": [], "gcontrol_delta_j_tv": [], "tie_max_dfield": []}
    for p in points:
        _assert_rescore(p.field, pi, mu, sigma, p.nll_over_n, p.r_tilde, p.spectral_hf_ratio)
        row = _base_row("cut-tv", name, p.lam, p)
        row["j_tv"] = p.energy  # the cut arm's energy *is* J_TV
        row["cut_value"] = p.cut_value
        row["quantisation_gap"] = p.quantisation_gap
        row["quantisation_bound"] = p.quantisation_bound

        if not args.skip_g_control:
            fine = exact_map_solve(
                pi,
                mu,
                sigma,
                p.lam,
                penalty="tv",
                n_levels=2 * args.n_levels - 1,
                n_sigma=args.n_sigma,
                edges=edges,
            )
            row["gcontrol_max_dfield"] = float(np.max(np.abs(fine.field - p.field)))
            row["gcontrol_delta_j_tv"] = fine.energy - p.energy
        if not args.skip_tie_probe:
            dithered = exact_map_solve(
                pi,
                mu,
                sigma,
                p.lam,
                penalty="tv",
                n_levels=args.n_levels,
                n_sigma=args.n_sigma,
                edges=edges,
                unary_dither=1e-9,
            )
            row["tie_max_dfield"] = float(np.max(np.abs(dithered.field - p.field)))
        controls["gcontrol_max_dfield"].append(row["gcontrol_max_dfield"])
        controls["gcontrol_delta_j_tv"].append(row["gcontrol_delta_j_tv"])
        controls["tie_max_dfield"].append(row["tie_max_dfield"])

        # Certificate: did Adam-on-Huber reach the exact TV optimum, up to its
        # own restart noise, the Huber surrogate gap, and integer quantisation?
        adam_row = adam_by_lam.get(p.lam)
        if adam_row is not None:
            delta_j_tv = adam_row["j_tv"] - p.energy
            allowance = (
                adam_row["restart_spread"]
                + p.lam * args.delta / 2.0
                + p.quantisation_bound
            )
            row["delta_j_tv_adam_minus_cut"] = delta_j_tv
            row["certificate_allowance"] = allowance
            row["adam_matches_cut"] = bool(delta_j_tv <= allowance)

        row.update(_smear(p.field, pi, mu, sigma, stage_c))
        rows.append(row)
    return points, rows, controls


def run_cut_quad(name, pi, mu, sigma, stage_b, stage_c, edges, n_edges, laplacian, args):
    """Arm 3 (Step 0): exact quadratic-MAP at Stage B's lambdas + Adam polish."""

    if args.quad_lambdas is not None:
        quad_lambdas = list(args.quad_lambdas)
    elif stage_b is not None:
        quad_lambdas = [float(l) for l in stage_b["lambdas"]]
    else:
        print("warning: no Stage B artifact and no --quad-lambdas; skipping cut-quad")
        return [], [], []

    rows = []
    points = []
    polished_fields = []
    for lam in quad_lambdas:
        res = exact_map_solve(
            pi,
            mu,
            sigma,
            lam,
            penalty="quadratic",
            n_levels=args.n_levels_quad,
            n_sigma=args.n_sigma,
            edges=edges,
        )
        points.append(res)

        # Stage D pattern: deterministic Adam polish of the decoded global
        # discrete optimum into its continuum basin, under the true J_lambda.
        def grad_fn(x, lam=lam):
            return objective_gradient(
                x, pi, mu, sigma, lam, laplacian=laplacian, n_edges=n_edges
            )

        polished = _adam_descent(
            res.field, grad_fn, n_steps=args.polish_steps, lr=args.polish_lr
        )
        polished_fields.append(polished)
        polished_energy = objective(
            polished, pi, mu, sigma, lam, laplacian=laplacian, n_edges=n_edges
        )

        scores = score_field(polished, pi, mu, sigma)
        row = _base_row("cut-quad", name, lam, res)
        # Base row carried the decoded grid field's scores; report the polished
        # field (the arm's deliverable) and keep the decoded energy alongside.
        row.update(
            {
                "nll_over_n": scores["nll_over_n"],
                "r_tilde": scores["r_tilde"],
                "variance_collapsed": scores["variance_collapsed"],
                "spectral_hf_ratio": scores["spectral_hf_ratio"],
                "spectral_slope": scores["spectral_slope"],
                "spectral_monotone_fraction": scores["spectral_monotone_fraction"],
                "spectral_collapsed": scores["spectral_collapsed"],
                "energy": polished_energy,
                "energy_decoded": res.energy,
                "cut_value": res.cut_value,
                "quantisation_gap": res.quantisation_gap,
                "quantisation_bound": res.quantisation_bound,
            }
        )

        idx = _match_lambda(lam, stage_b)
        if idx is not None:
            m1_field = stage_b["fields"][idx]
            m1_energy = objective(
                m1_field, pi, mu, sigma, lam, laplacian=laplacian, n_edges=n_edges
            )
            delta_j = polished_energy - m1_energy
            floor = float(stage_b["restart_spread"][idx])
            row["delta_j_vs_stage_b"] = delta_j
            row["restart_floor"] = floor
            row["beats_stage_b"] = bool(delta_j < -floor)
            row["max_abs_dx_vs_stage_b"] = float(np.max(np.abs(polished - m1_field)))

        row.update(_smear(polished, pi, mu, sigma, stage_c))
        rows.append(row)
    return points, rows, polished_fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="base RNG seed (adam arm only)")
    parser.add_argument(
        "--arms",
        type=lambda s: tuple(s.split(",")),
        default=ARMS,
        help="comma-separated subset of adam-tv,cut-tv,cut-quad",
    )
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=list(DEFAULT_LAMBDAS),
        help="TV-units lambda grid shared by adam-tv and cut-tv (matched certificates)",
    )
    parser.add_argument(
        "--quad-lambdas",
        type=float,
        nargs="+",
        default=None,
        help="cut-quad lambda grid; default = the persisted Stage B lambdas",
    )
    parser.add_argument("--delta", type=float, default=0.05, help="Huber kink half-width")
    parser.add_argument("--n-levels", type=int, default=257, help="TV value-grid size G")
    parser.add_argument("--n-levels-quad", type=int, default=129, help="quadratic value-grid size")
    parser.add_argument("--n-sigma", type=float, default=4.0, help="value-grid margin in sigmas")
    parser.add_argument("--n-restarts", type=int, default=4)
    parser.add_argument(
        "--n-steps",
        type=int,
        default=4000,
        help="Adam steps per restart (Stage B operating budget)",
    )
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--restart-scale", type=float, default=1.0)
    parser.add_argument("--polish-steps", type=int, default=1000, help="cut-quad Adam polish steps")
    parser.add_argument("--polish-lr", type=float, default=0.05)
    parser.add_argument("--skip-g-control", action="store_true")
    parser.add_argument("--skip-tie-probe", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=RUN_DIR)
    args = parser.parse_args()

    unknown = set(args.arms) - set(ARMS)
    if unknown:
        parser.error(f"unknown arms: {sorted(unknown)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in DATASETS:
        path = args.data_dir / f"{name}.npz"
        if not path.exists():
            print(f"skipping {name}: {path} not found")
            continue
        inputs = load_sampler_arrays(path)
        pi, mu, sigma = inputs["pi"], inputs["mu"], inputs["sigma"]
        height, width, _ = mu.shape
        edges = grid_edges_8(height, width)
        n_edges = len(edges)
        laplacian = graph_laplacian(height, width)

        stage_b = _load_stage_b(name)
        stage_c = _load_stage_c(name)
        if stage_b is None:
            print(f"warning: Stage B artifact for {name} missing; ΔJ columns will be nan")
        if stage_c is None:
            print(f"warning: Stage C artifact for {name} missing; ΔNLL→mode columns will be nan")

        npz_arrays = {
            "delta": np.asarray(args.delta, dtype=float),
            "n_levels": np.asarray(args.n_levels, dtype=np.int64),
            "n_levels_quad": np.asarray(args.n_levels_quad, dtype=np.int64),
            "seed": np.asarray(args.seed, dtype=np.int64),
            "value_grid_tv": value_grid(mu, sigma, n_levels=args.n_levels, n_sigma=args.n_sigma)[0],
            "value_grid_quad": value_grid(
                mu, sigma, n_levels=args.n_levels_quad, n_sigma=args.n_sigma
            )[0],
        }

        adam_rows = []
        if "adam-tv" in args.arms:
            adam_points, adam_rows = run_adam_tv(
                name, pi, mu, sigma, stage_c, edges, n_edges, args
            )
            rows.extend(adam_rows)
            npz_arrays.update(
                adam_tv_lambdas=np.asarray([p.lam for p in adam_points], dtype=float),
                adam_tv_fields=np.stack([p.field for p in adam_points]),
                adam_tv_nll_over_n=np.asarray([p.nll_over_n for p in adam_points], dtype=float),
                adam_tv_r_tilde=np.asarray([p.r_tilde for p in adam_points], dtype=float),
                adam_tv_energy=np.asarray([p.energy for p in adam_points], dtype=float),
                adam_tv_j_tv=np.asarray([r["j_tv"] for r in adam_rows], dtype=float),
                adam_tv_restart_spread=np.asarray(
                    [p.restart_spread for p in adam_points], dtype=float
                ),
                adam_tv_restart_field_spread=np.asarray(
                    [p.restart_field_spread for p in adam_points], dtype=float
                ),
            )
            print(f"{name}: adam-tv done ({len(adam_points)} lambda points)")

        if "cut-tv" in args.arms:
            cut_points, cut_rows, controls = run_cut_tv(
                name, pi, mu, sigma, stage_c, edges, n_edges, adam_rows, args
            )
            rows.extend(cut_rows)
            npz_arrays.update(
                cut_tv_lambdas=np.asarray([p.lam for p in cut_points], dtype=float),
                cut_tv_fields=np.stack([p.field for p in cut_points]),
                cut_tv_labels=np.stack([p.labels for p in cut_points]),
                cut_tv_nll_over_n=np.asarray([p.nll_over_n for p in cut_points], dtype=float),
                cut_tv_r_tilde=np.asarray([p.r_tilde for p in cut_points], dtype=float),
                cut_tv_energy=np.asarray([p.energy for p in cut_points], dtype=float),
                cut_tv_cut_value=np.asarray([p.cut_value for p in cut_points], dtype=float),
                cut_tv_quantisation_gap=np.asarray(
                    [p.quantisation_gap for p in cut_points], dtype=float
                ),
                cut_tv_quantisation_bound=np.asarray(
                    [p.quantisation_bound for p in cut_points], dtype=float
                ),
                cut_tv_gcontrol_max_dfield=np.asarray(
                    controls["gcontrol_max_dfield"], dtype=float
                ),
                cut_tv_gcontrol_delta_j_tv=np.asarray(
                    controls["gcontrol_delta_j_tv"], dtype=float
                ),
                cut_tv_tie_max_dfield=np.asarray(controls["tie_max_dfield"], dtype=float),
            )
            print(f"{name}: cut-tv done ({len(cut_points)} lambda points)")

        if "cut-quad" in args.arms:
            quad_points, quad_rows, polished_fields = run_cut_quad(
                name, pi, mu, sigma, stage_b, stage_c, edges, n_edges, laplacian, args
            )
            rows.extend(quad_rows)
            if quad_points:
                npz_arrays.update(
                    cut_quad_lambdas=np.asarray([p.lam for p in quad_points], dtype=float),
                    cut_quad_fields=np.stack([p.field for p in quad_points]),
                    cut_quad_polished_fields=np.stack(polished_fields),
                    cut_quad_cut_value=np.asarray([p.cut_value for p in quad_points], dtype=float),
                    cut_quad_energy_decoded=np.asarray(
                        [p.energy for p in quad_points], dtype=float
                    ),
                    cut_quad_energy_polished=np.asarray(
                        [r["energy"] for r in quad_rows], dtype=float
                    ),
                    cut_quad_delta_j_vs_stage_b=np.asarray(
                        [r["delta_j_vs_stage_b"] for r in quad_rows], dtype=float
                    ),
                    cut_quad_quantisation_gap=np.asarray(
                        [p.quantisation_gap for p in quad_points], dtype=float
                    ),
                    cut_quad_quantisation_bound=np.asarray(
                        [p.quantisation_bound for p in quad_points], dtype=float
                    ),
                )
            print(f"{name}: cut-quad done ({len(quad_points)} lambda points)")

        npz_path = args.out_dir / f"{name}_tv_ablation.npz"
        np.savez(npz_path, **npz_arrays)
        print(f"wrote {npz_path}")

    csv_path = args.out_dir / "stage_b_tv_scores.csv"
    md_path = args.out_dir / "stage_b_tv_scores.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
