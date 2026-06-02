"""Generate Stage A baselines for the Phase 1 toy datasets and score them.

Run from the repo root with the local (numpy-only) venv:

    .venv/bin/python scripts/run_stage_a_baselines.py

For each dataset in ``outputs/data/`` this loads only the sampler-facing arrays
(``pi``, ``mu``, ``sigma``, ``coords``; see ``load_sampler_arrays``), produces
the Stage A baseline fields (§7), and scores every produced field with
``NLL/N``, the raw roughness ``x^T L x``, scale-free roughness ``R̃``,
and secondary power-spectrum roughness diagnostics.

Outputs land in ``outputs/runs/stage_a_baselines/``:
  * ``stage_a_scores.csv`` / ``stage_a_scores.md`` — the score table,
  * ``<dataset>_baselines.npz`` — the generated baseline fields per dataset.

Debug arrays (``truth``, ``component_fields``, ``perm``) are never fed to the
baselines; they are loaded only for the optional ``--diagnostics`` NLL/roughness
of the held-out ground truth, where present.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from sampler_research.baselines import (
    iid_baseline,
    mixture_mean_field,
    mode_field,
    score_field,
    smoothed_map_baseline,
    smoothest_mode_assignment,
    variance_scaled_baseline,
)
from sampler_research.graph import (
    graph_laplacian,
    grid_edges_8,
    roughness_sum,
)
from sampler_research.io import load_npz, load_sampler_arrays

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "outputs" / "data"
RUN_DIR = REPO_ROOT / "outputs" / "runs" / "stage_a_baselines"

DATASETS = (
    "phase_1_field",
    "phase_1_field_heteroscedastic_sigma",
    "phase_1_regime_boundary_pi",
    "phase_1_multimodal",
)


def _score(field, pi, mu, sigma, edges, laplacian):
    """NLL/N, graph roughness, and secondary spectral diagnostics."""

    scores = score_field(field, pi, mu, sigma, edges)
    return {
        "nll_over_n": scores["nll_over_n"],
        "roughness_sum": roughness_sum(field, laplacian),
        "r_tilde": scores["r_tilde"],
        "variance_collapsed": scores["variance_collapsed"],
        "spectral_hf_ratio": scores["spectral_hf_ratio"],
        "spectral_slope": scores["spectral_slope"],
        "spectral_monotone_fraction": scores["spectral_monotone_fraction"],
        "spectral_collapsed": scores["spectral_collapsed"],
    }


def generate_baselines(pi, mu, sigma, alphas, rng):
    """Return an ordered dict of {baseline_name: field} for one dataset."""

    fields = {}

    iid, _ = iid_baseline(pi, mu, sigma, rng=rng)
    fields["iid"] = iid

    mode_f, _ = mode_field(pi, mu, sigma)
    fields["mode_map"] = mode_f

    fields["mixture_mean"] = mixture_mean_field(pi, mu)

    for alpha in alphas:
        sample, _ = variance_scaled_baseline(pi, mu, sigma, alpha=alpha, rng=rng)
        fields[f"variance_scaled_alpha_{alpha:g}"] = sample

    fields["smoothed_map"] = smoothed_map_baseline(pi, mu, sigma)

    astar, _ = smoothest_mode_assignment(pi, mu)
    fields["a_star"] = astar

    return fields


def write_csv(rows, path):
    fieldnames = [
        "dataset",
        "baseline",
        "nll_over_n",
        "roughness_sum",
        "r_tilde",
        "variance_collapsed",
        "spectral_hf_ratio",
        "spectral_slope",
        "spectral_monotone_fraction",
        "spectral_collapsed",
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
        "baseline",
        "NLL/N",
        "x^T L x",
        "R̃",
        "HF power",
        "slope",
        "mono",
        "var_collapsed",
        "spec_collapsed",
    ]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows:
        lines.append(
            (
                "| {dataset} | {baseline} | {nll:.4f} | {rough:.4f} | {rt} | "
                "{hf:.4f} | {slope} | {mono} | {vc} | {sc} |"
            ).format(
                dataset=row["dataset"],
                baseline=row["baseline"],
                nll=row["nll_over_n"],
                rough=row["roughness_sum"],
                rt=("collapsed" if row["variance_collapsed"] else f"{row['r_tilde']:.4f}"),
                hf=row["spectral_hf_ratio"],
                slope=_fmt_float(row["spectral_slope"]),
                mono=_fmt_float(row["spectral_monotone_fraction"]),
                vc=row["variance_collapsed"],
                sc=row["spectral_collapsed"],
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for stochastic baselines")
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.5, 0.25],
        help="variance-scaling factors for the Method 9 baseline",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=RUN_DIR)
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="also score the held-out ground-truth field where present (debug only)",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in DATASETS:
        path = args.data_dir / f"{name}.npz"
        inputs = load_sampler_arrays(path)
        pi, mu, sigma = inputs["pi"], inputs["mu"], inputs["sigma"]
        height, width, _ = mu.shape

        edges = grid_edges_8(height, width)
        laplacian = graph_laplacian(height, width, edges=edges)

        rng = np.random.default_rng(args.seed)
        fields = generate_baselines(pi, mu, sigma, args.alphas, rng)

        for baseline, field in fields.items():
            scores = _score(field, pi, mu, sigma, edges, laplacian)
            rows.append({"dataset": name, "baseline": baseline, **scores})

        if args.diagnostics:
            # Debug-only: score the held-out ground truth. `truth` is never a
            # sampler input; it is loaded here purely as a diagnostic reference.
            full = load_npz(path)
            if "truth" in full:
                scores = _score(full["truth"], pi, mu, sigma, edges, laplacian)
                rows.append({"dataset": name, "baseline": "truth (debug)", **scores})

        npz_path = args.out_dir / f"{name}_baselines.npz"
        np.savez(npz_path, **{k: np.asarray(v) for k, v in fields.items()})
        print(f"wrote {npz_path} ({', '.join(fields)})")

    csv_path = args.out_dir / "stage_a_scores.csv"
    md_path = args.out_dir / "stage_a_scores.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
