"""Stage B — regularised MAP (Method 1) lambda-sweep over the Phase 1 toys.

Run from the repo root with the local (numpy-only) venv:

    .venv/bin/python scripts/run_regularised_map.py

For each dataset in ``outputs/data/`` this loads only the sampler-facing arrays
(``pi``, ``mu``, ``sigma``, ``coords``; see ``load_sampler_arrays``), minimises
the regularised-MAP objective

    J_lambda(x) = NLL(x)/N + lambda * (x^T L x)/|E_8|

(closed-form gradient, §4.1.3) with an Adam optimiser warm-started from the
per-cell mode field plus a few random restarts (lowest-energy kept), over a grid
of lambda values. It emits, per dataset:

  * the lambda-sweep table (lambda, NLL/N, R̃, spectral diagnostics,
    restart-spread) as .md / .csv,
  * the chosen field at each lambda stacked into ``<dataset>_regularised_map.npz``.

Every chosen field is *also* re-scored with ``score_field`` (the Stage A scorer)
so the numbers sit in exactly the same brackets as the Stage A baselines (iid,
mode_map, a_star, smoothed_map). Stage B runs on the homoscedastic headline toy
(quadratic slowly-varying means + floored Dirichlet pi).

Debug arrays (``component_fields``, ``perm``) are never loaded.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from sampler_research.baselines import score_field
from sampler_research.io import load_sampler_arrays
from sampler_research.regularised_map import lambda_sweep

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "outputs" / "data"
RUN_DIR = REPO_ROOT / "outputs" / "runs" / "stage_b_regularised_map"

# Stage B runs on the homoscedastic headline toy only (same evidence path as
# Stage A); the heteroscedastic toy lives in notebook 00.
DATASETS = ("phase_1_homoscedastic",)

# lambda grid: 0 (mode-field / NLL-only anchor) then a log-spaced sweep up to
# lambda = 2, tracing the NLL/N-vs-R̃ Pareto curve on the 8x8 toy in the per-edge
# units of J_lambda (§3.1a). The grid stops at 2 on purpose: beyond it the field
# enters a saturated over-smoothing regime (the field goes near-flat, R̃
# differences fall within restart noise, and large lambda needs many more steps
# to converge) -- well past the operating knee (lambda ~ 0.1-0.5), so it is off
# the operating path. Pass --lambdas to extend it.
DEFAULT_LAMBDAS = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)


def _sweep_rows(name, points):
    """Flatten lambda-sweep points into score rows, re-scored with score_field."""

    rows = []
    for p in points:
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
                "restart_spread": p.restart_spread,
                "restart_field_spread": p.restart_field_spread,
                "energy": p.energy,
            }
        )
    return rows


def write_csv(rows, path):
    fieldnames = [
        "dataset",
        "lambda",
        "nll_over_n",
        "r_tilde",
        "variance_collapsed",
        "spectral_hf_ratio",
        "spectral_slope",
        "spectral_monotone_fraction",
        "spectral_collapsed",
        "restart_spread",
        "restart_field_spread",
        "energy",
    ]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
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
        "HF power",
        "slope",
        "mono",
        "var_collapsed",
        "spec_collapsed",
        "restart_spread (J)",
        "restart_field_spread",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        lines.append(
            (
                "| {dataset} | {lam:g} | {nll:.4f} | {rt} | {hf:.4f} | {slope} | "
                "{mono} | {vc} | {sc} | {rs:.3e} | {rfs:.4f} |"
            ).format(
                dataset=row["dataset"],
                lam=row["lambda"],
                nll=row["nll_over_n"],
                rt=("collapsed" if row["variance_collapsed"] else f"{row['r_tilde']:.4f}"),
                hf=row["spectral_hf_ratio"],
                slope=_fmt_float(row["spectral_slope"]),
                mono=_fmt_float(row["spectral_monotone_fraction"]),
                vc=row["variance_collapsed"],
                sc=row["spectral_collapsed"],
                rs=row["restart_spread"],
                rfs=row["restart_field_spread"],
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="base RNG seed for restarts")
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=list(DEFAULT_LAMBDAS),
        help="lambda grid for the Pareto sweep",
    )
    parser.add_argument("--n-restarts", type=int, default=4)
    parser.add_argument(
        "--n-steps",
        type=int,
        default=4000,
        help=(
            "Adam steps per restart. Needs to be large enough for the "
            "strong-smoothing (large-lambda) tail to converge; 600 "
            "under-optimises lambda >= 5."
        ),
    )
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--restart-scale", type=float, default=1.0)
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
        )

        # Re-score with the Stage A scorer for direct bracket comparability;
        # assert it agrees with the sweep's own primary and spectral scores.
        for p in points:
            stage_a = score_field(p.field, pi, mu, sigma)
            assert np.isclose(stage_a["nll_over_n"], p.nll_over_n)
            assert np.isclose(stage_a["r_tilde"], p.r_tilde)
            assert np.isclose(stage_a["spectral_hf_ratio"], p.spectral_hf_ratio)

        rows.extend(_sweep_rows(name, points))

        npz_path = args.out_dir / f"{name}_regularised_map.npz"
        np.savez(
            npz_path,
            lambdas=np.asarray([p.lam for p in points], dtype=float),
            fields=np.stack([p.field for p in points]),
            nll_over_n=np.asarray([p.nll_over_n for p in points], dtype=float),
            r_tilde=np.asarray([p.r_tilde for p in points], dtype=float),
            spectral_hf_ratio=np.asarray([p.spectral_hf_ratio for p in points], dtype=float),
            spectral_slope=np.asarray([p.spectral_slope for p in points], dtype=float),
            spectral_monotone_fraction=np.asarray(
                [p.spectral_monotone_fraction for p in points], dtype=float
            ),
            restart_spread=np.asarray([p.restart_spread for p in points], dtype=float),
        )
        print(f"wrote {npz_path} ({len(points)} lambda points)")

    csv_path = args.out_dir / "stage_b_scores.csv"
    md_path = args.out_dir / "stage_b_scores.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
