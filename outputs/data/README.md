# Synthetic GMM Data

The synthetic toy datasets used in Stage A/B are `.npz` files under `outputs/data/`. They
are **regenerable and not committed** (excluded by `.gitignore`); recreate them with:

    python scripts/generate_toy_data.py

- `phase_1_homoscedastic.npz` — headline toy used in Stage A/B: quadratic
  slowly-varying means, floored independent Dirichlet π, fixed σ = 1.
- `phase_1_heteroscedastic.npz` — same `pi` and `mu`, but per-location
  component-shared σ; currently a σ-only diagnostic variant.

The only dataset committed under `outputs/data/` is `natural_earth/` (the Natural Earth
coastline shapefile used for plotting). Figures live under `outputs/figures/`.

Each `.npz` should contain the sampler-facing arrays:

- `pi`
- `mu`
- `sigma`
- `coords`

It may also contain debug metadata:

- `component_fields`
- `perm`
- `seed`
- `variant_name`
- `constants`
- `pi_mode`
- `sigma_mode`
- `dirichlet_alpha`
- `pi_floor`

Sampler code should read only `pi`, `mu`, `sigma`, and `coords`. The debug metadata is for
diagnostics and reproducibility.
