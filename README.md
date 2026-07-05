# Weather Sampler

Post-hoc sampling of spatially coherent weather fields from a probabilistic foundation model.

## Overview

The WeatherGenerator decoder emits, for every grid location **independently**, a Gaussian-mixture
distribution over the local value. Sampled independently, those mixtures give a spatially
incoherent field. This project builds and evaluates *samplers* that assemble the per-location
mixtures into a single coherent, high-likelihood field that stays faithful to each location's
distribution — without retraining the decoder.

The code is organised by what each part does:

- a small synthetic **toy** model that reproduces the per-location-mixture setting on a
  controlled 8×8 grid;
- **baselines** — independent draw, per-cell MAP, smoothed MAP, mixture mean;
- two samplers — a **regularised MAP** (a smoothness-penalised joint objective) and a
  **mode-selection MRF** (discrete mode assignment via a Markov random field);
- **evaluation** on the real ERA5 2-metre-temperature field on the native O96 grid
  (40,320 locations);
- a **forecast-regime** study of how the mixtures soften with lead time (+6 h … +48 h).

## Layout

- `sampler_research/src/sampler_research/` — the library: toy construction, baselines, the two
  samplers, graph/GMM helpers, the evaluation protocol, diagnostics, and plotting.
- `sampler_research/tests/` — the regression suite.
- `scripts/` — runnable experiment and figure scripts.
- `notebooks/` — result notebooks (shipped with their outputs).
- `outputs/figures/` — key result figures.
- `outputs/data/` — small bundled inputs (Natural Earth coastlines); large generated arrays are
  not committed (see **Data availability**).
- `docs/` — how to reproduce the upstream model.
- `report/` — the dissertation and executive-summary PDFs.

## Install

    python3.11 -m venv .venv
    source .venv/bin/activate
    python -m pip install -U pip
    python -m pip install -r requirements.txt

Python ≥ 3.10. Runtime dependencies are numpy, scipy, matplotlib and pyshp — all pure-wheel, no
system libraries required. `requirements.txt` pins the assessed versions and installs the
package itself.

## Run the tests

    python -m pytest -q

Tests that need large generated arrays skip automatically when those arrays are absent, so the
suite is green on a clean clone.

## Reproduce

The synthetic toy data regenerates from source:

    python scripts/generate_phase1_data.py

The real-data and forecast results depend on the emitted mixture files from the WeatherGenerator
model, which are large and not committed. To regenerate them from scratch, follow
`docs/reproducing_the_model.md` (requires access to the upstream model and its HPC data). Once
the mixture `.npz` files are present under `outputs/data/`, the evaluation, figures and report
numbers regenerate via the `scripts/` runners.

## Data availability

The committed inputs are the synthetic toys (regenerable, above) and the Natural Earth coastline
shapefile used for plotting. The real per-location mixtures (ERA5, hundreds of MB) are produced
by the upstream model and are not redistributed; `docs/reproducing_the_model.md` documents how to
obtain and regenerate them.

## Report

- `report/thesis.pdf` — the dissertation.
- `report/summary.pdf` — the executive summary.

## License

See `LICENSE`.

## Use of AI assistance

Parts of this repository and the accompanying report were prepared with AI assistance; the
report's declaration appendix records this in full.
