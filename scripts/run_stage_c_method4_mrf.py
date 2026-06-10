"""Stage C — mode extraction + value-space MRF (Method 4) beta-sweep.

Run from the repo root with the local (numpy-only) venv:

    .venv/bin/python scripts/run_stage_c_method4_mrf.py

For ``phase_1_homoscedastic`` this loads only the sampler-facing arrays (``pi``,
``mu``, ``sigma``, ``coords``; see ``load_sampler_arrays``), extracts each cell's
GMM modes once, then minimises the value-space MRF energy

    J_beta(a) = (1/N) sum_i -log p_i(m_{i,a_i})
              + beta * (1/|E_8|) sum_(i,j) (m_{i,a_i} - m_{j,a_j})^2

(unary likelihood + value-space pairwise smoothness, §4.2) over a grid of beta.
The grid is *denser* than Method 1's lambda grid because the field is
piecewise-constant in beta — neighbouring beta routinely select the same field.

It emits, per dataset:

  * the beta-sweep table (beta, NLL/N, R̃, spectral diagnostics, restart spread,
    mode-count stats, and the non-smearing ΔNLL-to-best-mode summary) as .md/.csv,
  * ``<dataset>_method4_mrf.npz`` holding the chosen field + assignment at each
    beta **and the reusable mode artifacts** (``mode_values``, ``valid_mask``,
    ``mode_unary``, ``mode_counts``) so downstream stages (Method 1 / smoothed_map
    / the Method 5 ablation) score against the *exact* extracted modes rather than
    re-extracting.

Every chosen field is *also* re-scored with ``score_field`` (the Stage A scorer)
and asserted to agree, so the numbers sit in exactly the same brackets as the
Stage A baselines and the Stage B lambda-sweep. Debug arrays (``component_fields``,
``perm``) are never loaded.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from sampler_research.baselines import score_field
from sampler_research.io import load_sampler_arrays
from sampler_research.method4_mrf import beta_sweep, extract_gmm_modes

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "outputs" / "data"
RUN_DIR = REPO_ROOT / "outputs" / "runs" / "stage_c_method4_mrf"

# Stage C runs on the homoscedastic headline toy only (same evidence path as
# Stage A / Stage B); the heteroscedastic toy lives in notebook 00.
DATASETS = ("phase_1_homoscedastic",)

# beta grid: 0 (unary-best mode field, the smoothness-free anchor) then a denser
# log-spaced sweep than Method 1's. The transition from the jagged unary-best
# field to the saturated smoothest mode assignment happens at small beta and the
# field is piecewise-constant beyond it, so the grid is dense through the knee
# (~0.01-0.2) and then samples the saturated tail to show repeated fields.
DEFAULT_BETAS = (
    0.0,
    0.01,
    0.02,
    0.03,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.3,
    0.5,
    1.0,
    2.0,
)


def _sweep_rows(name, points, mode_counts):
    """Flatten beta-sweep points into score rows (mode-count + ΔNLL summaries)."""

    modes_min = int(mode_counts.min())
    modes_mean = float(mode_counts.mean())
    modes_max = int(mode_counts.max())
    rows = []
    for p in points:
        delta = p.delta_to_mode
        frac = delta["frac_over"]
        rows.append(
            {
                "dataset": name,
                "beta": p.beta,
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
                "modes_min": modes_min,
                "modes_mean": modes_mean,
                "modes_max": modes_max,
                "dnll_to_mode_mean": delta["mean"],
                "dnll_to_mode_p95": delta["p95"],
                "dnll_to_mode_max": delta["max"],
                "dnll_frac_gt_0p125_v1": frac[0],
                "dnll_frac_gt_0p5_v1": frac[1],
            }
        )
    return rows


CSV_FIELDS = [
    "dataset",
    "beta",
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
    "modes_min",
    "modes_mean",
    "modes_max",
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


def _fmt_float(value):
    return "nan" if not np.isfinite(value) else f"{value:.4f}"


def write_markdown(rows, path):
    header = [
        "dataset",
        "beta",
        "NLL/N",
        "R̃",
        "HF power",
        "slope",
        "mono",
        "var_coll",
        "restart_spread (J)",
        "restart_field_spread",
        "modes (min/mean/max)",
        "ΔNLL→mode mean",
        "p95",
        "max",
        "frac>0.125 (v1)",
        "frac>0.5 (v1)",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        lines.append(
            (
                "| {dataset} | {beta:g} | {nll:.4f} | {rt} | {hf:.4f} | {slope} | "
                "{mono} | {vc} | {rs:.3e} | {rfs:.4f} | {mmin}/{mmean:.2f}/{mmax} | "
                "{dmean:.4f} | {dp95:.4f} | {dmax:.4f} | {f125:.3f} | {f5:.3f} |"
            ).format(
                dataset=row["dataset"],
                beta=row["beta"],
                nll=row["nll_over_n"],
                rt=("collapsed" if row["variance_collapsed"] else f"{row['r_tilde']:.4f}"),
                hf=row["spectral_hf_ratio"],
                slope=_fmt_float(row["spectral_slope"]),
                mono=_fmt_float(row["spectral_monotone_fraction"]),
                vc=row["variance_collapsed"],
                rs=row["restart_spread"],
                rfs=row["restart_field_spread"],
                mmin=row["modes_min"],
                mmean=row["modes_mean"],
                mmax=row["modes_max"],
                dmean=row["dnll_to_mode_mean"],
                dp95=row["dnll_to_mode_p95"],
                dmax=row["dnll_to_mode_max"],
                f125=row["dnll_frac_gt_0p125_v1"],
                f5=row["dnll_frac_gt_0p5_v1"],
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="base RNG seed for restarts")
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=list(DEFAULT_BETAS),
        help="beta grid for the value-space MRF sweep (denser than Method 1's)",
    )
    parser.add_argument(
        "--n-restarts",
        type=int,
        default=128,
        help=(
            "ICM restarts per beta. The mode-assignment landscape is non-convex "
            "with a tiny per-cell label set, so few restarts return luck-of-the-"
            "draw local minima; 128 is the point at which the converged sweep is "
            "seed-invariant on the 8x8 toy (4 under-resolves the knee at beta~0.075)."
        ),
    )
    parser.add_argument("--max-sweeps", type=int, default=50, help="ICM sweeps per restart")
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

        # Extract modes once; the whole sweep reuses these exact candidates.
        extraction = extract_gmm_modes(pi, mu, sigma)
        points = beta_sweep(
            pi,
            mu,
            sigma,
            args.betas,
            n_restarts=args.n_restarts,
            seed=args.seed,
            extraction=extraction,
            max_sweeps=args.max_sweeps,
        )

        # Re-score with the Stage A scorer for direct bracket comparability;
        # assert it agrees with the sweep's own primary and spectral scores.
        for p in points:
            stage_a = score_field(p.field, pi, mu, sigma)
            assert np.isclose(stage_a["nll_over_n"], p.nll_over_n)
            assert np.isclose(stage_a["r_tilde"], p.r_tilde)
            assert np.isclose(stage_a["spectral_hf_ratio"], p.spectral_hf_ratio)

        rows.extend(_sweep_rows(name, points, extraction.mode_counts))

        npz_path = args.out_dir / f"{name}_method4_mrf.npz"
        np.savez(
            npz_path,
            betas=np.asarray([p.beta for p in points], dtype=float),
            fields=np.stack([p.field for p in points]),
            assignments=np.stack([p.assignment for p in points]),
            nll_over_n=np.asarray([p.nll_over_n for p in points], dtype=float),
            r_tilde=np.asarray([p.r_tilde for p in points], dtype=float),
            spectral_hf_ratio=np.asarray([p.spectral_hf_ratio for p in points], dtype=float),
            restart_spread=np.asarray([p.restart_spread for p in points], dtype=float),
            dnll_to_mode_mean=np.asarray([p.delta_to_mode["mean"] for p in points], dtype=float),
            # Reusable mode artifacts — the shared non-smearing reference.
            mode_values=extraction.mode_values,
            valid_mask=extraction.valid_mask,
            mode_unary=extraction.mode_unary,
            mode_counts=extraction.mode_counts,
        )
        print(f"wrote {npz_path} ({len(points)} beta points)")

    csv_path = args.out_dir / "stage_c_scores.csv"
    md_path = args.out_dir / "stage_c_scores.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
