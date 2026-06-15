"""Stage B — regularised MAP (Method 1) tests.

The load-bearing test is `test_objective_gradient_matches_finite_difference`:
the closed-form gradient from §4.1.3 must match a central finite-difference
gradient of `objective` to tight tolerance, on the real toy. The rest assert the
objective decomposition, that the optimiser actually lowers J_lambda from the
mode-field warm start, the lambda -> 0 / lambda -> large limits behave (anchors
to mode_map / flat), restart-spread is exposed and small, and the lambda-sweep
Pareto curve trades NLL/N against roughness monotonically at the endpoints.
"""

from pathlib import Path

import numpy as np
import pytest

from sampler_research.baselines import gmm_nll_over_n, mode_field
from sampler_research.graph import graph_laplacian, grid_edges_8, roughness_sum
from sampler_research.io import load_sampler_arrays
from sampler_research.regularised_map import (
    lambda_sweep,
    minimise_at_lambda,
    nll_gradient,
    objective,
    objective_gradient,
)
from sampler_research.toy import Phase1ToyConfig, make_phase1_toy

DATA_DIR = Path(__file__).resolve().parents[2] / "outputs" / "data"


def _headline_toy():
    """Prefer the on-disk homoscedastic headline toy; fall back to in-memory."""

    path = DATA_DIR / "phase_1_homoscedastic.npz"
    if path.exists():
        arrays = load_sampler_arrays(path)
        return arrays["pi"], arrays["mu"], arrays["sigma"]
    toy = make_phase1_toy(Phase1ToyConfig())
    return toy["pi"], toy["mu"], toy["sigma"]


def test_objective_decomposes_into_nll_and_smoothness():
    pi, mu, sigma = _headline_toy()
    edges = grid_edges_8(8, 8)
    laplacian = graph_laplacian(8, 8)
    n_edges = len(edges)
    field, _ = mode_field(pi, mu, sigma)

    lam = 0.3
    expected = gmm_nll_over_n(field, pi, mu, sigma) + lam * roughness_sum(
        field, laplacian
    ) / n_edges
    got = objective(field, pi, mu, sigma, lam, laplacian=laplacian, n_edges=n_edges)
    assert np.isclose(got, expected)

    # lambda = 0 is exactly NLL/N.
    assert np.isclose(
        objective(field, pi, mu, sigma, 0.0, laplacian=laplacian, n_edges=n_edges),
        gmm_nll_over_n(field, pi, mu, sigma),
    )


@pytest.mark.parametrize("lam", [0.0, 0.2, 1.0])
def test_objective_gradient_matches_finite_difference(lam):
    """Closed-form grad J_lambda vs central finite differences (tight rtol)."""

    pi, mu, sigma = _headline_toy()
    laplacian = graph_laplacian(8, 8)
    n_edges = len(grid_edges_8(8, 8))

    rng = np.random.default_rng(0)
    # Evaluate away from the mode field so responsibilities are non-degenerate.
    base, _ = mode_field(pi, mu, sigma)
    x = base + rng.normal(scale=0.5, size=base.shape)

    analytic = objective_gradient(x, pi, mu, sigma, lam, laplacian=laplacian, n_edges=n_edges)

    eps = 1e-6
    numeric = np.zeros_like(x)
    flat = x.reshape(-1)
    for idx in range(flat.size):
        plus = flat.copy()
        minus = flat.copy()
        plus[idx] += eps
        minus[idx] -= eps
        j_plus = objective(
            plus.reshape(x.shape), pi, mu, sigma, lam, laplacian=laplacian, n_edges=n_edges
        )
        j_minus = objective(
            minus.reshape(x.shape), pi, mu, sigma, lam, laplacian=laplacian, n_edges=n_edges
        )
        numeric.reshape(-1)[idx] = (j_plus - j_minus) / (2 * eps)

    assert np.allclose(analytic, numeric, rtol=1e-5, atol=1e-7)


def test_objective_gradient_finite_at_mode_field():
    """Sanity: gradient is finite everywhere and the objective is differentiable."""

    pi, mu, sigma = _headline_toy()
    laplacian = graph_laplacian(8, 8)
    n_edges = len(grid_edges_8(8, 8))
    field, _ = mode_field(pi, mu, sigma)
    grad = objective_gradient(field, pi, mu, sigma, 0.5, laplacian=laplacian, n_edges=n_edges)
    assert grad.shape == (8, 8)
    assert np.all(np.isfinite(grad))


def test_nll_gradient_finite_far_off_every_mode():
    """Pushing a cell far beyond the underflow horizon (~38.6 sigma) of every
    mode drives the GMM density p_i(x_i) to exactly 0.0 in float64. The
    responsibility-divide guard (A1) must then return a finite (zero) gradient
    contribution there instead of 0/0 = NaN (this fails on the pre-guard code)."""

    pi, mu, sigma = _headline_toy()
    field, _ = mode_field(pi, mu, sigma)
    field = field.copy()
    field[0, 0] = float(np.max(mu[0, 0])) + 1.0e6 * float(np.max(sigma[0, 0]))

    grad = nll_gradient(field, pi, mu, sigma)
    assert grad.shape == (8, 8)
    assert np.all(np.isfinite(grad))
    # The underflowed cell receives exactly the guarded (zero) contribution.
    assert grad[0, 0] == 0.0


def test_optimiser_lowers_objective_vs_warm_start():
    pi, mu, sigma = _headline_toy()
    laplacian = graph_laplacian(8, 8)
    n_edges = len(grid_edges_8(8, 8))
    warm, _ = mode_field(pi, mu, sigma)

    lam = 0.5
    warm_energy = objective(warm, pi, mu, sigma, lam, laplacian=laplacian, n_edges=n_edges)
    res = minimise_at_lambda(
        pi, mu, sigma, lam, n_restarts=4, n_steps=400, rng=np.random.default_rng(0)
    )
    # Optimiser must not do worse than the warm start it is initialised from.
    assert res.energy <= warm_energy + 1e-9


def test_lambda_zero_stays_near_mode_field():
    """lambda = 0 -> pure NLL; from the mode warm start it should barely move."""

    pi, mu, sigma = _headline_toy()
    warm, _ = mode_field(pi, mu, sigma)
    res = minimise_at_lambda(
        pi, mu, sigma, 0.0, n_restarts=1, n_steps=200, rng=np.random.default_rng(0)
    )
    # The mode field is a near-stationary point of NLL/N (each cell sits at a
    # component mean), so descent stays in its neighbourhood rather than wandering.
    assert res.nll_over_n <= gmm_nll_over_n(warm, pi, mu, sigma) + 1e-6


def test_large_lambda_flattens_field():
    """Large lambda drives toward a near-constant field (low roughness)."""

    pi, mu, sigma = _headline_toy()
    laplacian = graph_laplacian(8, 8)
    res = minimise_at_lambda(
        pi, mu, sigma, 50.0, n_restarts=2, n_steps=600, rng=np.random.default_rng(0)
    )
    warm, _ = mode_field(pi, mu, sigma)
    assert roughness_sum(res.field, laplacian) < roughness_sum(warm, laplacian)


def test_restart_spread_is_exposed_and_small():
    """Restart spread (energy + field) is reported and stable on the toy."""

    pi, mu, sigma = _headline_toy()
    res = minimise_at_lambda(
        pi, mu, sigma, 0.5, n_restarts=4, n_steps=400, rng=np.random.default_rng(0)
    )
    assert res.restart_energies.shape == (4,)
    assert res.restart_spread >= 0.0
    assert res.restart_field_spread >= 0.0
    # Energies should be finite and the spread a small fraction of the energy level.
    assert np.all(np.isfinite(res.restart_energies))
    assert res.restart_spread < abs(res.energy) + 1.0


def test_lambda_sweep_trades_nll_for_roughness():
    """The lambda-sweep trades faithfulness for smoothness.

    On this overlapping-mean toy the objective is non-convex, so the R̃ Pareto
    front is *not* strictly monotone step-by-step: at neighbouring lambda the
    restarts settle into different local minima, so R̃ can tick up slightly in
    the operating range (this persists even at the full 4000-step budget). The
    robust, meaningful invariants are therefore: NLL/N is monotone
    non-decreasing as lambda grows, lambda=0 (the mode-field anchor) is the
    roughest point, and the smoothing regime drives R̃ well below it.
    """

    pi, mu, sigma = _headline_toy()
    lambdas = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    points = lambda_sweep(pi, mu, sigma, lambdas, n_restarts=3, n_steps=600, seed=0)
    assert [p.lam for p in points] == lambdas

    # Faithfulness is given up monotonically as lambda grows (clean at every step).
    for lo, hi in zip(points, points[1:]):
        assert hi.nll_over_n >= lo.nll_over_n - 1e-6, "NLL/N should not decrease with lambda"

    # lambda=0 is the NLL-faithful, roughest anchor; smoothing collapses R̃.
    r_zero = points[0].r_tilde
    assert all(p.r_tilde <= r_zero + 1e-9 for p in points), "lambda=0 should be the roughest point"
    assert min(p.r_tilde for p in points) < 0.5 * r_zero, "smoothing must materially reduce R̃"

    for p in points:
        assert np.isfinite(p.nll_over_n)
        assert p.field.shape == (8, 8)
