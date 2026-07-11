"""Stage D — Method 5 (annealed-noise Langevin **search**) ablation tests.

Method 5 is a Method 1 stochastic-search / robustness ablation, not a separate
sampler: it minimises Method 1's *exact* `J_lambda` with Method 1's gradient, then
polishes with Method 1's Adam. The load-bearing properties checked here are
therefore (1) it really is an *optimiser* of `J_lambda` — the polished selection
never has higher energy than the chain snapshot it came from, and the zero-noise
limit reduces to deterministic descent landing at a Method-1-equivalent local min;
(2) the `ΔJ` headline against the persisted Stage B fields is computed against the
*same* shared `objective` and is finite; and (3) it can be read against Stage C's
persisted modes via `delta_nll_to_best_mode`. The rest cover reproducibility,
one-field-per-lambda output schema, and the Stage-B roughness-vs-lambda nuance.
"""

from pathlib import Path

import numpy as np

from sampler_research.baselines import mode_field, score_field
from sampler_research.graph import grid_edges_8, graph_laplacian
from sampler_research.io import load_sampler_arrays
from sampler_research.method4_mrf import delta_nll_to_best_mode, extract_gmm_modes
from sampler_research.method5_langevin import (
    langevin_at_lambda,
    langevin_chain,
    langevin_sweep,
)
from sampler_research.regularised_map import objective, objective_gradient
from sampler_research.toy import Phase1ToyConfig, make_phase1_toy

DATA_DIR = Path(__file__).resolve().parents[2] / "outputs" / "data"

# Light search settings: keep the suite fast while still exercising the chain +
# polish + selection machinery (the production grid lives in the Stage D script).
FAST = dict(n_chains=4, n_steps=200, polish_steps=200)


def _headline_toy():
    """Prefer the on-disk homoscedastic headline toy; fall back to in-memory."""

    path = DATA_DIR / "phase_1_homoscedastic.npz"
    if path.exists():
        arrays = load_sampler_arrays(path)
        return arrays["pi"], arrays["mu"], arrays["sigma"]
    toy = make_phase1_toy(Phase1ToyConfig())
    return toy["pi"], toy["mu"], toy["sigma"]


# --- reproducibility --------------------------------------------------------


def test_same_seed_gives_identical_fields_and_diagnostics():
    pi, mu, sigma = _headline_toy()
    lambdas = [0.0, 0.1, 0.5]

    a = langevin_sweep(pi, mu, sigma, lambdas, seed=0, **FAST)
    b = langevin_sweep(pi, mu, sigma, lambdas, seed=0, **FAST)
    assert [p.lam for p in a] == lambdas
    for pa, pb in zip(a, b):
        assert np.array_equal(pa.field, pb.field)
        assert pa.energy == pb.energy
        assert pa.best_chain == pb.best_chain
        assert pa.best_chain_step == pb.best_chain_step
        assert pa.chain_energy_spread == pb.chain_energy_spread


def test_different_seeds_stay_finite_and_one_field_per_lambda():
    pi, mu, sigma = _headline_toy()
    lambdas = [0.0, 0.1, 0.5, 1.0]

    points = langevin_sweep(pi, mu, sigma, lambdas, seed=7, **FAST)
    assert len(points) == len(lambdas)
    for p in points:
        assert p.field.shape == (8, 8)
        assert np.all(np.isfinite(p.field))
        assert np.isfinite(p.energy)
        assert np.isfinite(p.nll_over_n)
        assert p.chain_energy_spread >= 0.0
        assert p.chain_field_spread >= 0.0
        # The selected field must score finite under the Stage A scorer too.
        scores = score_field(p.field, pi, mu, sigma)
        assert np.isfinite(scores["nll_over_n"])


# --- zero-noise reduces to deterministic descent ----------------------------


def test_zero_noise_reduces_to_deterministic_descent():
    """noise_start=noise_end=0 (+ no jitter) => deterministic GD+polish on J_lambda.

    With no injected noise and no warm-start jitter the chain is a plain
    (RNG-independent) gradient descent on `J_lambda`; followed by the Adam polish
    it lands at a Method-1 local minimum, so the selected energy must not exceed
    the mode-field warm start (the lambda anchor Method 1 starts from).
    """

    pi, mu, sigma = _headline_toy()
    lam = 0.2
    laplacian = graph_laplacian(8, 8)
    n_edges = len(grid_edges_8(8, 8))
    warm, _ = mode_field(pi, mu, sigma)
    warm_energy = objective(warm, pi, mu, sigma, lam, laplacian=laplacian, n_edges=n_edges)

    res_a = langevin_at_lambda(
        pi, mu, sigma, lam, noise_start=0.0, noise_end=0.0, jitter_scale=0.0,
        rng=np.random.default_rng(0), **FAST,
    )
    res_b = langevin_at_lambda(
        pi, mu, sigma, lam, noise_start=0.0, noise_end=0.0, jitter_scale=0.0,
        rng=np.random.default_rng(123), **FAST,
    )
    # No noise and no jitter => the chains are deterministic regardless of RNG.
    assert np.array_equal(res_a.field, res_b.field)
    assert res_a.energy <= warm_energy + 1e-9


# --- polish never worsens the snapshot it came from -------------------------


def test_selected_polished_energy_at_most_best_snapshot_energy():
    """The deterministic polish is a descent step: selected J <= snapshot J."""

    pi, mu, sigma = _headline_toy()
    for lam in (0.0, 0.1, 0.5):
        res = langevin_at_lambda(
            pi, mu, sigma, lam, rng=np.random.default_rng(0), **FAST
        )
        assert res.energy <= res.best_snapshot_energy + 1e-9
        # The selected chain is the lowest-energy polished chain.
        assert res.energy <= res.chain_energies.min() + 1e-12


# --- chain is a true optimiser of J_lambda (gradient matches objective) -----


def test_chain_snapshot_lowers_energy_below_warm_start():
    """A single search chain returns a snapshot no worse than its warm start."""

    pi, mu, sigma = _headline_toy()
    lam = 0.2
    laplacian = graph_laplacian(8, 8)
    n_edges = len(grid_edges_8(8, 8))
    warm, _ = mode_field(pi, mu, sigma)

    def grad_fn(x):
        return objective_gradient(x, pi, mu, sigma, lam, laplacian=laplacian, n_edges=n_edges)

    def energy_fn(x):
        return objective(x, pi, mu, sigma, lam, laplacian=laplacian, n_edges=n_edges)

    field, energy, step = langevin_chain(
        warm, grad_fn, energy_fn, n_steps=200, step_size=0.05,
        noise_start=0.5, noise_end=0.02, rng=np.random.default_rng(0),
    )
    assert energy <= energy_fn(warm) + 1e-9
    assert 0 <= step <= 200
    assert np.isclose(energy_fn(field), energy)


# --- ΔJ vs Method 1 against the shared objective ----------------------------


def test_delta_j_against_shared_objective_is_finite_and_correct():
    """ΔJ uses the *same* `objective` to score both methods' selected fields."""

    pi, mu, sigma = _headline_toy()
    lam = 0.2
    laplacian = graph_laplacian(8, 8)
    n_edges = len(grid_edges_8(8, 8))

    res = langevin_at_lambda(pi, mu, sigma, lam, rng=np.random.default_rng(0), **FAST)
    # Stand-in "Method 1" field: the mode-field warm start, scored by the shared
    # objective. The point is the bookkeeping, not which field is better.
    m1_field, _ = mode_field(pi, mu, sigma)
    m1_energy = objective(m1_field, pi, mu, sigma, lam, laplacian=laplacian, n_edges=n_edges)
    delta_j = res.energy - m1_energy
    assert np.isfinite(delta_j)
    # `res.energy` is itself J_lambda under the same objective.
    assert np.isclose(
        res.energy,
        objective(res.field, pi, mu, sigma, lam, laplacian=laplacian, n_edges=n_edges),
    )


# --- non-smearing probe against Stage C modes -------------------------------


def test_can_score_against_extracted_modes():
    """Method 5's selected field reads against Method 4's modes (drift >= 0)."""

    pi, mu, sigma = _headline_toy()
    ext = extract_gmm_modes(pi, mu, sigma)
    res = langevin_at_lambda(pi, mu, sigma, 0.2, rng=np.random.default_rng(0), **FAST)

    smear = delta_nll_to_best_mode(
        res.field, pi, mu, sigma, ext.mode_values, ext.valid_mask
    )
    assert smear["max"] >= -1e-9
    for key in ("mean", "p95", "max"):
        assert np.isfinite(smear[key])


# --- roughness-vs-lambda nuance (mirrors Stage B) ---------------------------


def test_roughness_drops_with_lambda_without_strict_monotonicity():
    """lambda=0 is roughest and min R̃ < 0.5 * R̃(lambda=0); not strict step-wise.

    Mirrors test_stage_b_regularised_map.py: strong smoothing must visibly lower
    R̃ relative to the unregularised field, but we do not assert monotonicity at
    every step (restart/chain noise makes neighbouring lambdas non-monotone).
    """

    pi, mu, sigma = _headline_toy()
    lambdas = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
    points = langevin_sweep(pi, mu, sigma, lambdas, seed=0, **FAST)
    r_tildes = np.array([p.r_tilde for p in points])

    assert np.argmax(r_tildes) == 0  # lambda=0 is the roughest
    assert r_tildes.min() < 0.5 * r_tildes[0]


# --- output schema is one field per lambda, not an ensemble -----------------


def test_sweep_output_is_one_field_per_lambda():
    pi, mu, sigma = _headline_toy()
    lambdas = [0.0, 0.1, 0.5]
    points = langevin_sweep(pi, mu, sigma, lambdas, seed=0, **FAST)
    assert len(points) == len(lambdas)
    for p in points:
        # A single [H, W] field, never a stacked [C, H, W] ensemble.
        assert p.field.ndim == 2
        assert p.field.shape == (8, 8)
