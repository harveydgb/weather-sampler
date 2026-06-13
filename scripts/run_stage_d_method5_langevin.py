"""HISTORICAL (archived 11 Jun 2026) — Stage D closed: "nothing moved".

CUT from the real-data path per phase_4_plan.md "Scope verdicts": every |ΔJ| sat
within the Stage B restart band, so Method 1's optimiser is validated and re-running
at scale answers no open question. The verdict lives in large_notes.md (Phase 2
status) / phase_2.md; the energy-delta table is persisted under
outputs/runs/stage_d_method5_langevin/. Re-runnable on the toy only.

Stage D — Method 5 as a Method 1 stochastic-search / robustness ablation.

Run from the repo root with the local (numpy-only) venv:

    .venv/bin/python scripts/run_stage_d_method5_langevin.py

Method 5 is **not** a third sampler family (phase_2_research_plan.md §4.3,
DEFERRED→ensemble). It runs annealed-noise Langevin **search** on Method 1's
*exact* energy `J_lambda` and gradient, then deterministically polishes each
chain's lowest-energy snapshot with Method 1's Adam descent and keeps the
lowest-energy polished field. So it is Method 1 with a harder search procedure,
and the honest question it answers is:

    does stochastic global search find materially lower-`J_lambda` basins than
    Method 1's mode-warm-start + Adam restarts?

Because both methods minimise the *same* objective, their `(NLL/N, R̃)` curves
near-overlap *by construction* — near-overlap is tautology, not corroboration.
The headline is therefore an **energy-delta table**, not a second Pareto curve:
per lambda this reports `ΔJ = J_lambda(Method5_selected) - J_lambda(Method1_selected)`,
with the Method 1 side sourced from the persisted Stage B fields and re-scored
with the *same* shared `objective` at matched lambda. A "materially better basin"
is flagged only when `ΔJ` is clearly below `-restart_spread(lambda)` (the Stage B
per-lambda noise floor) **and** the operating point visibly shifts; any `|ΔJ|`
within the restart band means "no material difference / Method 1 validated."

It also scores every selected field with Method 4's `delta_nll_to_best_mode`
(using the persisted Stage C modes): Method 5 minimises Method 1's *smooth*
objective and can smear into the low-density valleys between modes, so this column
shows whether a harder search smears more or less than plain Method 1.

Outputs (`outputs/runs/stage_d_method5_langevin/`):
  * `stage_d_scores.{csv,md}` — the ablation table,
  * `phase_1_homoscedastic_method5_langevin.npz` — **selected fields + compact
    chain diagnostics only** (one field per lambda; the deliverable is not an
    ensemble).

Every selected field is re-scored with `score_field` (the Stage A scorer) and
asserted to agree, so the numbers sit in the same brackets as Stage A/B/C. Debug
arrays are never loaded.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from sampler_research.baselines import score_field
from sampler_research.graph import grid_edges_8, graph_laplacian
from sampler_research.io import load_sampler_arrays
from sampler_research.method4_mrf import delta_nll_to_best_mode
from sampler_research.method5_langevin import langevin_sweep
from sampler_research.regularised_map import objective

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "outputs" / "data"
RUN_DIR = REPO_ROOT / "outputs" / "runs" / "stage_d_method5_langevin"
STAGE_B_DIR = REPO_ROOT / "outputs" / "runs" / "stage_b_regularised_map"
STAGE_C_DIR = REPO_ROOT / "outputs" / "runs" / "stage_c_method4_mrf"

# Stage D runs on the homoscedastic headline toy only (same evidence path as
# Stage A / B / C).
DATASETS = ("phase_1_homoscedastic",)

# Matches the Stage B lambda grid exactly so every Method 5 row has a matched
# Method 1 field to take ΔJ against (§ plan: "scored by the same function").
DEFAULT_LAMBDAS = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)


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


def _match_lambda(lam, stage_b):
    """Index of the Stage B row at lambda `lam`, or None if there is no match."""

    if stage_b is None:
        return None
    hits = np.where(np.isclose(stage_b["lambdas"], lam))[0]
    return int(hits[0]) if hits.size else None


def _build_rows(name, points, pi, mu, sigma, stage_b, stage_c, laplacian, n_edges):
    """Assemble Stage D score rows: NLL/R̃, ΔJ vs Method 1, smear, material flag."""

    rows = []
    for p in points:
        idx = _match_lambda(p.lam, stage_b)
        if idx is None:
            delta_j = float("nan")
            restart_floor = float("nan")
            max_abs_dx = float("nan")
            rms_dx = float("nan")
            d_nll = float("nan")
            d_rtilde = float("nan")
        else:
            m1_field = stage_b["fields"][idx]
            # Re-score Method 1's field with the *same* objective at this lambda.
            m1_energy = objective(
                m1_field, pi, mu, sigma, p.lam, laplacian=laplacian, n_edges=n_edges
            )
            delta_j = p.energy - m1_energy
            restart_floor = float(stage_b["restart_spread"][idx])
            diff = p.field - m1_field
            max_abs_dx = float(np.max(np.abs(diff)))
            rms_dx = float(np.sqrt(np.mean(diff**2)))
            d_nll = p.nll_over_n - float(stage_b["nll_over_n"][idx])
            d_rtilde = p.r_tilde - float(stage_b["r_tilde"][idx])

        # Non-smearing probe against the persisted Stage C modes.
        if stage_c is None:
            dnll_mean = dnll_p95 = dnll_max = float("nan")
            frac125 = frac5 = float("nan")
        else:
            smear = delta_nll_to_best_mode(
                p.field, pi, mu, sigma, stage_c["mode_values"], stage_c["valid_mask"]
            )
            dnll_mean, dnll_p95, dnll_max = smear["mean"], smear["p95"], smear["max"]
            frac125, frac5 = smear["frac_over"]

        # Material-improvement flag (§ plan promotion rule): ΔJ clearly below the
        # Stage B per-lambda restart floor AND the operating point visibly shifts
        # (NLL/N or R̃ moves beyond that same noise band). Within the band =>
        # "no material difference / Method 1 validated."
        material = bool(
            np.isfinite(delta_j)
            and np.isfinite(restart_floor)
            and delta_j < -restart_floor
            and (abs(d_nll) > restart_floor or abs(d_rtilde) > restart_floor)
        )

        rows.append(
            {
                "dataset": name,
                "lambda": p.lam,
                "nll_over_n": p.nll_over_n,
                "r_tilde": p.r_tilde,
                "variance_collapsed": p.variance_collapsed,
                "spectral_hf_ratio": p.spectral_hf_ratio,
                "spectral_slope": p.spectral_slope,
                "spectral_monotone_fraction": p.spectral_monotone_fraction,
                "spectral_collapsed": p.spectral_collapsed,
                "energy": p.energy,
                "delta_j_vs_method1": delta_j,
                "restart_floor": restart_floor,
                "best_chain": p.best_chain,
                "chain_energy_spread": p.chain_energy_spread,
                "best_chain_step": p.best_chain_step,
                "chain_field_spread": p.chain_field_spread,
                "max_abs_dx_vs_method1": max_abs_dx,
                "rms_dx_vs_method1": rms_dx,
                "dnll_to_mode_mean": dnll_mean,
                "dnll_to_mode_p95": dnll_p95,
                "dnll_to_mode_max": dnll_max,
                "dnll_frac_gt_0p125_v1": frac125,
                "dnll_frac_gt_0p5_v1": frac5,
                "material_improvement": material,
            }
        )
    return rows


CSV_FIELDS = [
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
    "delta_j_vs_method1",
    "restart_floor",
    "best_chain",
    "chain_energy_spread",
    "best_chain_step",
    "chain_field_spread",
    "max_abs_dx_vs_method1",
    "rms_dx_vs_method1",
    "dnll_to_mode_mean",
    "dnll_to_mode_p95",
    "dnll_to_mode_max",
    "dnll_frac_gt_0p125_v1",
    "dnll_frac_gt_0p5_v1",
    "material_improvement",
]


def write_csv(rows, path):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt_float(value):
    return "nan" if not np.isfinite(value) else f"{value:.4f}"


def write_markdown(rows, path):
    header = [
        "dataset",
        "lambda",
        "NLL/N",
        "R̃",
        "energy (J)",
        "ΔJ vs M1",
        "restart_floor",
        "material?",
        "best_chain",
        "chain_energy_spread",
        "best_step",
        "chain_field_spread",
        "max|Δx| vs M1",
        "RMS Δx",
        "ΔNLL→mode mean",
        "p95",
        "max",
    ]
    lines = [
        "*Method 5 = Method 1 energy, harder search (stochastic-search ablation, not a bake-off entrant).*",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        lines.append(
            (
                "| {dataset} | {lam:g} | {nll:.4f} | {rt} | {energy:.4f} | {dj} | "
                "{rf} | {mat} | {bc} | {ces:.3e} | {bs} | {cfs:.4f} | {mdx} | "
                "{rms} | {dmean} | {dp95} | {dmax} |"
            ).format(
                dataset=row["dataset"],
                lam=row["lambda"],
                nll=row["nll_over_n"],
                rt=("collapsed" if row["variance_collapsed"] else f"{row['r_tilde']:.4f}"),
                energy=row["energy"],
                dj=_fmt_float(row["delta_j_vs_method1"]),
                rf=_fmt_float(row["restart_floor"]),
                mat=row["material_improvement"],
                bc=row["best_chain"],
                ces=row["chain_energy_spread"],
                bs=row["best_chain_step"],
                cfs=row["chain_field_spread"],
                mdx=_fmt_float(row["max_abs_dx_vs_method1"]),
                rms=_fmt_float(row["rms_dx_vs_method1"]),
                dmean=_fmt_float(row["dnll_to_mode_mean"]),
                dp95=_fmt_float(row["dnll_to_mode_p95"]),
                dmax=_fmt_float(row["dnll_to_mode_max"]),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="base RNG seed for chains")
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=list(DEFAULT_LAMBDAS),
        help="lambda grid (match Stage B so every row has a matched Method 1 field)",
    )
    parser.add_argument("--n-chains", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=2500, help="Langevin steps per chain")
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--noise-start", type=float, default=0.5)
    parser.add_argument("--noise-end", type=float, default=0.02)
    parser.add_argument("--polish-steps", type=int, default=1000, help="Adam polish steps")
    parser.add_argument("--polish-lr", type=float, default=0.05)
    parser.add_argument("--jitter-scale", type=float, default=1.0)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=RUN_DIR)
    args = parser.parse_args()

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
        laplacian = graph_laplacian(height, width)
        n_edges = len(grid_edges_8(height, width))

        stage_b = _load_stage_b(name)
        stage_c = _load_stage_c(name)
        if stage_b is None:
            print(f"warning: Stage B artifact for {name} missing; ΔJ column will be nan")
        if stage_c is None:
            print(f"warning: Stage C artifact for {name} missing; ΔNLL→mode column will be nan")

        points = langevin_sweep(
            pi,
            mu,
            sigma,
            args.lambdas,
            n_chains=args.n_chains,
            n_steps=args.n_steps,
            step_size=args.step_size,
            noise_start=args.noise_start,
            noise_end=args.noise_end,
            polish_steps=args.polish_steps,
            polish_lr=args.polish_lr,
            jitter_scale=args.jitter_scale,
            seed=args.seed,
        )

        # Re-score with the Stage A scorer for direct bracket comparability;
        # assert it agrees with the sweep's own primary and spectral scores.
        for p in points:
            stage_a = score_field(p.field, pi, mu, sigma)
            assert np.isclose(stage_a["nll_over_n"], p.nll_over_n)
            assert np.isclose(stage_a["r_tilde"], p.r_tilde)
            assert np.isclose(stage_a["spectral_hf_ratio"], p.spectral_hf_ratio)

        rows.extend(
            _build_rows(name, points, pi, mu, sigma, stage_b, stage_c, laplacian, n_edges)
        )

        npz_path = args.out_dir / f"{name}_method5_langevin.npz"
        np.savez(
            npz_path,
            lambdas=np.asarray([p.lam for p in points], dtype=float),
            # One selected field per lambda — the deliverable, NOT the ensemble.
            fields=np.stack([p.field for p in points]),
            nll_over_n=np.asarray([p.nll_over_n for p in points], dtype=float),
            r_tilde=np.asarray([p.r_tilde for p in points], dtype=float),
            energy=np.asarray([p.energy for p in points], dtype=float),
            # Compact chain diagnostics only.
            chain_energy_spread=np.asarray(
                [p.chain_energy_spread for p in points], dtype=float
            ),
            chain_field_spread=np.asarray(
                [p.chain_field_spread for p in points], dtype=float
            ),
            best_chain=np.asarray([p.best_chain for p in points], dtype=np.int64),
            best_chain_step=np.asarray([p.best_chain_step for p in points], dtype=np.int64),
            best_snapshot_energy=np.asarray(
                [p.best_snapshot_energy for p in points], dtype=float
            ),
        )
        print(f"wrote {npz_path} ({len(points)} lambda points)")

    csv_path = args.out_dir / "stage_d_scores.csv"
    md_path = args.out_dir / "stage_d_scores.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
