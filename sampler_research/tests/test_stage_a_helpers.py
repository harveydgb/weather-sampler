"""Focused sanity tests for the Stage A helpers on the saved toy `.npz` files.

These complement ``test_stage_a_baselines.py`` (which uses the in-memory toy and
covers the |E_8|=210 edge count and the C5 value-vs-label invariance). Here we
drive the baselines from the on-disk sampler arrays and assert the basic
well-formedness contracts: shapes, finiteness, valid component indices,
non-negative roughness, the flat-`(64,)` metric path, and that smoothed-MAP
reduces raw roughness relative to the unsmoothed per-cell mode field.
"""

from pathlib import Path

import numpy as np
import pytest

from sampler_research.baselines import (
    gmm_nll_over_n,
    iid_baseline,
    mixture_mean_field,
    mode_field,
    smoothed_map_baseline,
    smoothest_mode_assignment,
    variance_scaled_baseline,
)
from sampler_research.graph import (
    graph_laplacian,
    grid_edges_8,
    roughness_edge_mean,
    roughness_sum,
)
from sampler_research.io import load_sampler_arrays

DATA_DIR = Path(__file__).resolve().parents[2] / "outputs" / "data"
DATASETS = (
    "phase_1_field",
    "phase_1_field_heteroscedastic_sigma",
    "phase_1_regime_boundary_pi",
)


def _load(name):
    arrays = load_sampler_arrays(DATA_DIR / f"{name}.npz")
    return arrays["pi"], arrays["mu"], arrays["sigma"]


def _baseline_fields(pi, mu, sigma, rng):
    """One representative field per Stage A baseline family."""

    iid, _ = iid_baseline(pi, mu, sigma, rng=rng)
    mode_f, _ = mode_field(pi, mu, sigma)
    scaled, _ = variance_scaled_baseline(pi, mu, sigma, alpha=0.5, rng=rng)
    astar, _ = smoothest_mode_assignment(pi, mu)
    return {
        "iid": iid,
        "mode_map": mode_f,
        "mixture_mean": mixture_mean_field(pi, mu),
        "variance_scaled": scaled,
        "smoothed_map": smoothed_map_baseline(pi, mu, sigma),
        "a_star": astar,
    }


@pytest.fixture(params=DATASETS)
def toy(request):
    name = request.param
    if not (DATA_DIR / f"{name}.npz").exists():
        pytest.skip(f"missing toy data {name}.npz; run scripts/generate_phase1_data.py")
    pi, mu, sigma = _load(name)
    return name, pi, mu, sigma


def test_baseline_fields_are_8x8(toy):
    _, pi, mu, sigma = toy
    rng = np.random.default_rng(0)
    for label, field in _baseline_fields(pi, mu, sigma, rng).items():
        assert field.shape == (8, 8), f"{label} field has shape {field.shape}"
        assert np.all(np.isfinite(field)), f"{label} field has non-finite values"


def test_flat_metric_path_handles_64_vector(toy):
    """Roughness metrics accept the flattened (64,) view and agree with (8,8)."""

    _, pi, mu, _ = toy
    edges = grid_edges_8(8, 8)
    laplacian = graph_laplacian(8, 8)
    field = mixture_mean_field(pi, mu)
    flat = field.reshape(-1)

    assert flat.shape == (64,)
    assert np.isclose(roughness_sum(field, laplacian), roughness_sum(flat, laplacian))
    assert np.isclose(roughness_edge_mean(field, edges), roughness_edge_mean(flat, edges))


def test_iid_finite_and_valid_component_indices(toy):
    _, pi, mu, sigma = toy
    k = mu.shape[-1]
    rng = np.random.default_rng(123)
    sample, index = iid_baseline(pi, mu, sigma, rng=rng)

    assert np.all(np.isfinite(sample))
    assert index.shape == (8, 8)
    assert np.issubdtype(index.dtype, np.integer)
    assert index.min() >= 0 and index.max() < k


def test_nll_over_n_finite_for_all_baselines(toy):
    _, pi, mu, sigma = toy
    rng = np.random.default_rng(0)
    for label, field in _baseline_fields(pi, mu, sigma, rng).items():
        nll = gmm_nll_over_n(field, pi, mu, sigma)
        assert np.isfinite(nll), f"{label} produced non-finite NLL/N"


def test_raw_roughness_is_non_negative(toy):
    _, pi, mu, sigma = toy
    laplacian = graph_laplacian(8, 8)
    rng = np.random.default_rng(0)
    for label, field in _baseline_fields(pi, mu, sigma, rng).items():
        rough = roughness_sum(field, laplacian)
        assert rough >= -1e-9, f"{label} produced negative roughness {rough}"


def test_smoothed_map_reduces_roughness_vs_mode_field(toy):
    """Smoothing the per-cell mode field lowers raw roughness x^T L x."""

    _, pi, mu, sigma = toy
    laplacian = graph_laplacian(8, 8)
    mode_f, _ = mode_field(pi, mu, sigma)
    smoothed = smoothed_map_baseline(pi, mu, sigma)

    assert roughness_sum(smoothed, laplacian) < roughness_sum(mode_f, laplacian)


def test_astar_invariant_to_per_cell_relabeling(toy):
    """`a*` couples on mean *values*, never on a globally meaningful label index.

    Shuffling the component axis per cell (so a given label index no longer
    denotes the same component anywhere) must leave the produced field bit-exact.
    This holds even under uniform-`pi`/tied costs because `a*` breaks every tie
    by component value, not by label index (C5). Run on all three on-disk toys.
    """

    name, pi, mu, _ = toy
    astar, _ = smoothest_mode_assignment(pi, mu)

    rng = np.random.default_rng(7)
    k = pi.shape[-1]
    perm = np.stack([[rng.permutation(k) for _ in range(8)] for _ in range(8)])
    pi_p = np.take_along_axis(pi, perm, axis=-1)
    mu_p = np.take_along_axis(mu, perm, axis=-1)
    astar_relabelled, _ = smoothest_mode_assignment(pi_p, mu_p)

    assert np.allclose(astar, astar_relabelled), f"{name}: a* changed under relabeling"
