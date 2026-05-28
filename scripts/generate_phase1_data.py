#!/usr/bin/env python3
"""Generate the Phase 1 and Phase 1.5 toy `.npz` files."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "sampler_research" / "src"))

from sampler_research.io import load_npz
from sampler_research.toy import Phase1ToyConfig, save_phase1_toy


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=(
            "baseline",
            "heteroscedastic",
            "regime_boundary",
            "regime-boundary",
            "both",
            "all",
        ),
        default="all",
        help="Which toy variant to generate. `both` means baseline plus heteroscedastic.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "data",
        help="Directory for generated `.npz` files.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def configs_for_variant(variant, seed):
    variant = variant.replace("-", "_")
    configs = []
    if variant in {"baseline", "both", "all"}:
        configs.append(Phase1ToyConfig(seed=seed, use_heteroscedastic_sigma=False))
    if variant in {"heteroscedastic", "both", "all"}:
        configs.append(Phase1ToyConfig(seed=seed, use_heteroscedastic_sigma=True))
    if variant in {"regime_boundary", "all"}:
        configs.append(Phase1ToyConfig(seed=seed, use_regime_boundary_pi=True))
    return configs


def main():
    args = parse_args()
    for config in configs_for_variant(args.variant, args.seed):
        path = save_phase1_toy(args.output_dir, config)
        data = load_npz(path)
        print(f"wrote {path}")
        print(
            "  "
            + " ".join(
                f"{key}={tuple(data[key].shape)}" for key in ("pi", "mu", "sigma", "coords")
            )
        )
        print(
            f"  variant_name={data['variant_name'].item()} "
            f"pi_mode={data['pi_mode'].item()} sigma_mode={data['sigma_mode'].item()}"
        )


if __name__ == "__main__":
    main()
