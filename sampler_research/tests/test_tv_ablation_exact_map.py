"""Stage B-TV ablation tests — Huber penalty + exact Ishikawa min-cut MAP.

The authority here is `test_exact_map_matches_brute_force`: on tiny grids the
Ishikawa min-cut energy must equal the enumerated minimum over *all* labelings
(energies compared, so the check is tie-robust), for both penalties, several
value-grid sizes and seeds. The rest: the Huber closed-form gradient against
central finite differences (away from the |d| = delta kink), the lambda = 0
cut collapsing to the per-cell unary argmax on the real toy, the quantisation
gap staying within its certified bound, and — load-bearing for Method 1 — the
default quadratic path of `regularised_map` staying bit-identical with and
without the new `penalty` keyword, including a re-score of the persisted
Stage B artifact.
"""

from pathlib import Path

import numpy as np
import pytest

from sampler_research import exact_map
from sampler_research.baselines import gmm_nll_over_n, mode_field
from sampler_research.graph import graph_laplacian, grid_edges_8, scale_free_roughness
from sampler_research.io import load_sampler_arrays
from sampler_research.regularised_map import (
    minimise_at_lambda,
    objective,
    objective_gradient,
)
from sampler_research.toy import Phase1ToyConfig, make_phase1_toy

DATA_DIR = Path(__file__).resolve().parents[2] / "outputs" / "data"
STAGE_B_NPZ = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "runs"
    / "stage_b_regularised_map"
    / "phase_1_homoscedastic_regularised_map.npz"
)


def _headline_toy():
    """Prefer the on-disk homoscedastic headline toy; fall back to in-memory."""

    path = DATA_DIR / "phase_1_homoscedastic.npz"
    if path.exists():
        arrays = load_sampler_arrays(path)
        return arrays["pi"], arrays["mu"], arrays["sigma"]
    toy = make_phase1_toy(Phase1ToyConfig())
    return toy["pi"], toy["mu"], toy["sigma"]


def _tiny_gmm(seed, height=2, width=3, k=2):
    """Small random GMM field for the brute-force enumeration tests."""

    rng = np.random.default_rng(seed)
    mu = rng.normal(0.0, 1.0, size=(height, width, k))
    sigma = rng.uniform(0.5, 1.5, size=(height, width, k))
    raw = rng.uniform(0.2, 1.0, size=(height, width, k))
    pi = raw / raw.sum(axis=-1, keepdims=True)
    return pi, mu, sigma


# --- (a) Huber gradient vs central finite differences -----------------------


@pytest.mark.parametrize("lam", [0.0, 0.2, 1.0])
@pytest.mark.parametrize("delta", [0.05, 0.5])
def test_huber_objective_gradient_matches_finite_difference(lam, delta):
    """Closed-form Huber grad vs central FD, away from the |d| = delta kink."""

    pi, mu, sigma = _headline_toy()
    edges = grid_edges_8(8, 8)
    base, _ = mode_field(pi, mu, sigma)

    x = None
    for seed in range(50):
        cand = base + np.random.default_rng(seed).normal(scale=0.5, size=base.shape)
        flat = cand.reshape(-1)
        d = flat[edges[:, 0]] - flat[edges[:, 1]]
        # FD steps of 1e-6 must not cross the kink: keep every |d| at least
        # 1e-4 away from delta.
        if np.min(np.abs(np.abs(d) - delta)) > 1e-4:
            x = cand
            break
    assert x is not None, "no kink-free evaluation point found"

    analytic = objective_gradient(
        x, pi, mu, sigma, lam, penalty="huber", delta=delta, edges=edges
    )

    eps = 1e-6
    numeric = np.zeros_like(x)
    flat = x.reshape(-1)
    for idx in range(flat.size):
        plus = flat.copy()
        minus = flat.copy()
        plus[idx] += eps
        minus[idx] -= eps
        j_plus = objective(
            plus.reshape(x.shape), pi, mu, sigma, lam,
            penalty="huber", delta=delta, edges=edges,
        )
        j_minus = objective(
            minus.reshape(x.shape), pi, mu, sigma, lam,
            penalty="huber", delta=delta, edges=edges,
        )
        numeric.reshape(-1)[idx] = (j_plus - j_minus) / (2 * eps)

    assert np.allclose(analytic, numeric, rtol=1e-5, atol=1e-7)


# --- (b) Exactness vs brute force — the authority ----------------------------


@pytest.mark.parametrize("penalty", ["tv", "quadratic"])
@pytest.mark.parametrize("n_levels", [5, 7])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_exact_map_matches_brute_force(penalty, n_levels, seed):
    """Cut energy == enumerated minimum over all G^6 labelings (tie-robust)."""

    height, width = 2, 3
    pi, mu, sigma = _tiny_gmm(seed, height=height, width=width)
    n = height * width
    edges = grid_edges_8(height, width)
    n_edges = len(edges)

    values, _ = exact_map.value_grid(mu, sigma, n_levels=n_levels, n_sigma=4.0)
    unary = exact_map.unary_table(values, pi, mu, sigma)  # [N, G] raw nats

    # Every labeling, vectorised: [G^N, N] grid indices.
    labelings = np.stack(
        np.meshgrid(*([np.arange(n_levels)] * n), indexing="ij"), axis=-1
    ).reshape(-1, n)
    unary_term = unary[np.arange(n)[None, :], labelings].sum(axis=1) / n
    diffs = values[labelings[:, edges[:, 0]]] - values[labelings[:, edges[:, 1]]]
    pen_term = (np.abs(diffs) if penalty == "tv" else diffs**2).sum(axis=1) / n_edges

    for lam in (0.0, 0.3, 1.0):
        enum_min = float(np.min(unary_term + lam * pen_term))
        res = exact_map.exact_map_solve(
            pi, mu, sigma, lam, penalty=penalty, n_levels=n_levels
        )
        # The decoded field is itself an enumerated labeling, so its float
        # energy can only sit above the enumerated minimum; the cut certifies
        # it is also (within integer quantisation) not above it.
        assert res.energy >= enum_min - 1e-9
        assert res.energy <= enum_min + res.quantisation_bound + 1e-9


# --- (c) lambda = 0 reduces to the per-cell unary argmax ---------------------


@pytest.mark.parametrize("penalty", ["tv", "quadratic"])
def test_lambda_zero_cut_is_unary_argmax(penalty):
    """lambda = 0 decodes each cell's unary argmax (tie-robust at int resolution).

    Two grid points symmetric about a density peak can have unaries equal
    within the integer capacity resolution (`n/scale_u = 2 * quantisation
    bound` raw nats); there the cut legitimately tie-breaks to the smaller
    value, so cells are held to per-cell optimality within that resolution and
    to exact argmax agreement wherever the runner-up gap exceeds it.
    """

    pi, mu, sigma = _headline_toy()
    res = exact_map.exact_map_solve(pi, mu, sigma, 0.0, penalty=penalty)
    values, _ = exact_map.value_grid(mu, sigma, n_levels=res.n_levels)
    unary = exact_map.unary_table(values, pi, mu, sigma)

    rows = np.arange(unary.shape[0])
    labels = res.labels.reshape(-1)
    tol = 2.0 * res.quantisation_bound
    gap = unary[rows, labels] - unary.min(axis=1)
    assert np.all(gap <= tol)

    runner_up = np.partition(unary, 1, axis=1)[:, 1] - unary.min(axis=1)
    clear = runner_up > tol
    assert clear.sum() >= unary.shape[0] // 2  # the check must have teeth
    assert np.array_equal(labels[clear], np.argmin(unary, axis=1)[clear])


# --- (d) quantisation gap within its certified bound on the real toy ---------


@pytest.mark.parametrize("penalty,lam", [("tv", 0.2), ("tv", 2.0), ("quadratic", 0.2)])
def test_quantisation_gap_within_bound_on_real_toy(penalty, lam):
    pi, mu, sigma = _headline_toy()
    res = exact_map.exact_map_solve(pi, mu, sigma, lam, penalty=penalty)
    assert res.quantisation_gap <= res.quantisation_bound
    # The bound itself is tiny: half an integer ulp per cut data arc.
    assert res.quantisation_bound < 1e-3


# --- (e) Method 1 regression: default quadratic path is bit-identical --------


def test_objective_and_gradient_bit_identical_with_quadratic_kwarg():
    pi, mu, sigma = _headline_toy()
    laplacian = graph_laplacian(8, 8)
    n_edges = len(grid_edges_8(8, 8))
    base, _ = mode_field(pi, mu, sigma)
    x = base + np.random.default_rng(0).normal(scale=0.5, size=base.shape)

    for lam in (0.0, 0.2, 1.0):
        legacy = objective(x, pi, mu, sigma, lam, laplacian=laplacian, n_edges=n_edges)
        explicit = objective(
            x, pi, mu, sigma, lam, laplacian=laplacian, n_edges=n_edges,
            penalty="quadratic",
        )
        assert legacy == explicit  # bitwise, not isclose

        legacy_grad = objective_gradient(
            x, pi, mu, sigma, lam, laplacian=laplacian, n_edges=n_edges
        )
        explicit_grad = objective_gradient(
            x, pi, mu, sigma, lam, laplacian=laplacian, n_edges=n_edges,
            penalty="quadratic",
        )
        assert np.array_equal(legacy_grad, explicit_grad)


def test_minimise_at_lambda_bit_identical_with_quadratic_kwarg():
    pi, mu, sigma = _headline_toy()
    legacy = minimise_at_lambda(
        pi, mu, sigma, 0.5, n_restarts=2, n_steps=60, rng=np.random.default_rng(0)
    )
    explicit = minimise_at_lambda(
        pi, mu, sigma, 0.5, n_restarts=2, n_steps=60, rng=np.random.default_rng(0),
        penalty="quadratic",
    )
    assert np.array_equal(legacy.field, explicit.field)
    assert legacy.energy == explicit.energy
    assert np.array_equal(legacy.restart_energies, explicit.restart_energies)


def test_unknown_penalty_raises():
    pi, mu, sigma = _headline_toy()
    field, _ = mode_field(pi, mu, sigma)
    with pytest.raises(ValueError):
        objective(field, pi, mu, sigma, 0.1, penalty="cauchy")
    with pytest.raises(ValueError):
        objective_gradient(field, pi, mu, sigma, 0.1, penalty="cauchy")
    with pytest.raises(ValueError):
        exact_map.exact_map_solve(pi, mu, sigma, 0.1, penalty="huber")


def test_persisted_stage_b_npz_still_rescores():
    """The persisted Stage B fields must re-score to their stored columns."""

    if not STAGE_B_NPZ.exists():
        pytest.skip("persisted Stage B artifact not present")
    pi, mu, sigma = _headline_toy()
    edges = grid_edges_8(8, 8)
    data = np.load(STAGE_B_NPZ)
    for fld, nll, r_tilde in zip(data["fields"], data["nll_over_n"], data["r_tilde"]):
        assert np.isclose(gmm_nll_over_n(fld, pi, mu, sigma), nll, rtol=0, atol=1e-10)
        got_r, _ = scale_free_roughness(fld, edges)
        assert np.isclose(got_r, r_tilde, rtol=0, atol=1e-10)
