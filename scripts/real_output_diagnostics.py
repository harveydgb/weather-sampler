#!/usr/bin/env python3
"""ARCHIVED (11 Jun 2026) — superseded by scripts/audit_real_gmm.py; do not extend.

Targets the wiped e3fz467m checkpoint (`gmm_params_e3fz467m.pt`), whose surviving
facts live in research_notes/phase_4_data_audit.md; the canonical artifact is now
the converted gmm_era5_32ep_v3 .npz, audited (numbers + figures) by
scripts/audit_real_gmm.py with the torch-free research venv. Kept for the trail
only — not deleted per the archival policy.

Run the existing real-checkpoint salt-and-pepper diagnostics."""

import argparse
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent

from sampler_research.diagnostics import sampled_spherical_variogram
from sampler_research.gmm import (
    as_numpy,
    mixture_mean,
    mixture_pdf,
    normal_pdf,
    reference_point_by_median_mean,
    sample_iid_gmm,
)
from sampler_research.io import load_real_checkpoint, real_checkpoint_schema


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=WORKSPACE_ROOT / "model_outputs" / "gmm_params_e3fz467m.pt",
        help="Checkpoint containing `pi`, `mu_2t`, `sigma_2t`, and `latlons`.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-pairs", type=int, default=60_000)
    parser.add_argument(
        "--save-figures",
        action="store_true",
        help="Save the notebook-style diagnostic figures under outputs/figures.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = load_real_checkpoint(args.checkpoint)

    print(f"loaded {args.checkpoint}")
    for key, shape in real_checkpoint_schema(checkpoint):
        print(f"{key:20s} {shape}")

    pi = as_numpy(checkpoint["pi"]).astype(float)
    mu_2t = as_numpy(checkpoint["mu_2t"]).astype(float)
    sigma_2t = as_numpy(checkpoint["sigma_2t"]).astype(float)
    latlons = as_numpy(checkpoint["latlons"]).astype(float)

    rng = np.random.default_rng(args.seed)
    sample_iid, _ = sample_iid_gmm(pi, mu_2t, sigma_2t, rng)
    mean_field = mixture_mean(pi, mu_2t)
    print(f"\nN={pi.shape[0]} spatial points, K={pi.shape[1]} GMM components")
    print(f"IID sample  mean={sample_iid.mean():.3f} std={sample_iid.std():.3f}")
    print(f"Mean field  mean={mean_field.mean():.3f} std={mean_field.std():.3f}")
    print(f"std ratio IID/mean-field={sample_iid.std() / mean_field.std():.2f}")

    i_ref = reference_point_by_median_mean(mean_field)
    print(
        f"Reference point {i_ref}: "
        f"pi={np.round(pi[i_ref], 3)} "
        f"mu={np.round(mu_2t[i_ref], 3)} "
        f"sigma={np.round(sigma_2t[i_ref], 3)}"
    )

    centres, variograms, dist_deg = sampled_spherical_variogram(
        latlons,
        {"IID sample": sample_iid, "Mean field": mean_field},
        n_pairs=args.n_pairs,
        seed=args.seed,
    )
    short = dist_deg < 5.0
    if short.sum() > 10:
        short_iid = variograms["IID sample"][centres < 5.0]
        short_mean = variograms["Mean field"][centres < 5.0]
        ratio = np.nanmean(short_iid) / max(float(np.nanmean(short_mean)), 1e-9)
        print(f"Short-range (<5 deg) semivariance IID/mean-field={ratio:.1f}x")

    if args.save_figures:
        import matplotlib.pyplot as plt

        from sampler_research.plotting import (
            plot_mollweide_fields,
            plot_pooled_distribution,
            plot_single_point_marginal,
            plot_variograms,
        )

        args.figures_dir.mkdir(parents=True, exist_ok=True)
        n_draws = 30_000
        k_s = rng.choice(pi.shape[1], size=n_draws, p=pi[i_ref])
        samples_single = mu_2t[i_ref, k_s] + sigma_2t[i_ref, k_s] * rng.standard_normal(n_draws)
        x = np.linspace(
            mu_2t[i_ref].min() - 3.5 * sigma_2t[i_ref].max(),
            mu_2t[i_ref].max() + 3.5 * sigma_2t[i_ref].max(),
            600,
        )
        component_pdfs = [
            pi[i_ref, k] * normal_pdf(x, mu_2t[i_ref, k], sigma_2t[i_ref, k])
            for k in range(pi.shape[1])
        ]
        total_pdf = mixture_pdf(x, pi[i_ref], mu_2t[i_ref], sigma_2t[i_ref])

        figs = [
            plot_single_point_marginal(
                samples_single,
                x,
                total_pdf,
                component_pdfs,
                pi[i_ref],
                point_index=i_ref,
            )[0],
            plot_pooled_distribution(sample_iid, mean_field)[0],
            plot_mollweide_fields(
                latlons,
                {
                    "Mean field E[GMM] (smooth reference)": mean_field,
                    "IID sample (salt & pepper)": sample_iid,
                },
                suptitle="2m temperature - checkpoint diagnostic",
            )[0],
            plot_variograms(
                centres,
                variograms,
                xlabel="Angular separation (deg)",
                title="Empirical variogram - 2t normalised",
            )[0],
        ]
        names = [
            "real_single_point_marginal.png",
            "real_pooled_distribution.png",
            "real_mollweide_fields.png",
            "real_variogram.png",
        ]
        for fig, name in zip(figs, names):
            out = args.figures_dir / name
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"saved {out}")
        plt.close("all")


if __name__ == "__main__":
    main()
