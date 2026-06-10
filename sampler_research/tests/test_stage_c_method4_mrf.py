"""Stage C — mode extraction + value-space MRF (Method 4) tests.

The load-bearing correctness property is
`test_unary_prevents_low_density_compromise`: without the unary term the
value-space MRF would pick a value-compatible but low-density mode, and the
`-log p_i(mode)` unary is what stops that (§4.2 step 3). The rest cover mode
finding (single / separated / overlapping GMMs and the `<=K` bound), the ragged
padding/masking (no `nan`/`inf` leaking into `argmin`), C5 relabel invariance of
both the modes and the final field, the `beta=0` unary-best limit, that the
shared ICM core reproduces `a*` bit-for-bit when fed component means with no
unary, seeded-restart reproducibility + finite scores, and (optional) that ICM
never raises `J_beta` and the extracted modes are true density maxima.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sampler_research.baselines import (
    _argmin_by_value,
    _value_space_icm,
    gmm_nll_per_cell,
    mode_field,
    score_field,
    smoothest_mode_assignment,
)
from sampler_research.gmm import normal_pdf
from sampler_research.graph import grid_edges_8
from sampler_research.io import load_sampler_arrays
from sampler_research.method4_mrf import (
    DRIFT_THRESHOLDS_V1,
    beta_sweep,
    delta_nll_to_best_mode,
    extract_gmm_modes,
    solve_value_mrf,
    value_mrf_energy,
    _unary_best_assignment,
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


def _single_cell(pi, mu, sigma):
    """Wrap one cell's (pi, mu, sigma) lists into [1, 1, K] toy arrays."""

    pi = np.asarray(pi, dtype=float).reshape(1, 1, -1)
    mu = np.asarray(mu, dtype=float).reshape(1, 1, -1)
    sigma = np.asarray(sigma, dtype=float).reshape(1, 1, -1)
    return pi, mu, sigma


# --- mode finding -----------------------------------------------------------


def test_single_gaussian_has_one_mode_at_the_mean():
    pi, mu, sigma = _single_cell([1.0], [2.5], [1.0])
    ext = extract_gmm_modes(pi, mu, sigma)
    assert ext.mode_counts[0] == 1
    assert np.isclose(ext.mode_values[0, 0], 2.5)


def test_separated_components_give_two_modes():
    pi, mu, sigma = _single_cell([0.5, 0.5], [-5.0, 5.0], [1.0, 1.0])
    ext = extract_gmm_modes(pi, mu, sigma)
    assert ext.mode_counts[0] == 2
    assert np.allclose(np.sort(ext.mode_values[0, :2]), [-5.0, 5.0], atol=1e-3)


def test_overlapping_components_collapse_to_fewer_modes_than_components():
    """Overlapping K=2 -> a single GMM peak (not a component mean): != a*.

    `a*` would keep both component means {-0.3, 0.3} as candidates; mode
    extraction returns the one true density peak at 0, proving the methods couple
    on different candidate sets.
    """

    pi, mu, sigma = _single_cell([0.5, 0.5], [-0.3, 0.3], [1.0, 1.0])
    ext = extract_gmm_modes(pi, mu, sigma)
    assert ext.mode_counts[0] == 1
    assert np.isclose(ext.mode_values[0, 0], 0.0, atol=1e-6)
    # The mode is a genuine peak, not either component mean (the a* candidates).
    assert not np.isclose(ext.mode_values[0, 0], 0.3)
    assert not np.isclose(ext.mode_values[0, 0], -0.3)


def test_mode_count_respects_k_bound_on_the_toy():
    """Equal-sigma 1D mean-shift yields <= K modes per cell (Kmax = K)."""

    pi, mu, sigma = _headline_toy()
    k = mu.shape[-1]
    ext = extract_gmm_modes(pi, mu, sigma)
    assert ext.mode_values.shape[1] == k
    assert ext.mode_counts.min() >= 1
    assert ext.mode_counts.max() <= k


# --- ragged padding / masking ----------------------------------------------


def test_ragged_padding_and_masking_never_leak_nan_or_inf():
    pi, mu, sigma = _headline_toy()
    ext = extract_gmm_modes(pi, mu, sigma)
    n, kmax = ext.mode_values.shape

    # Padded *values* are finite sentinels (never nan), so the pairwise term is
    # always finite arithmetic; only the *unary* carries the +inf exclusion.
    assert np.all(np.isfinite(ext.mode_values))
    assert np.all(np.isfinite(ext.mode_unary[ext.valid_mask]))
    assert np.all(np.isinf(ext.mode_unary[~ext.valid_mask]))

    # Valid mask matches the per-cell counts and the first count slots.
    for i in range(n):
        c = ext.mode_counts[i]
        assert ext.valid_mask[i, :c].all()
        assert not ext.valid_mask[i, c:].any()

    res = solve_value_mrf(pi, mu, sigma, ext, 0.2, rng=np.random.default_rng(0))
    assert np.all(np.isfinite(res.field))
    assert np.isfinite(res.energy)
    # Chosen index is always a valid (real) mode, never a padded slot.
    assert np.all(res.assignment.reshape(n) < ext.mode_counts)


# --- C5 relabel invariance --------------------------------------------------


def test_relabeling_components_leaves_modes_and_field_unchanged():
    """Per-cell component relabeling changes neither the modes nor the field (C5)."""

    pi, mu, sigma = _headline_toy()
    ext = extract_gmm_modes(pi, mu, sigma)
    field = solve_value_mrf(pi, mu, sigma, ext, 0.2, rng=np.random.default_rng(3)).field

    rng = np.random.default_rng(7)
    k = pi.shape[-1]
    perm = np.stack([[rng.permutation(k) for _ in range(8)] for _ in range(8)])
    pi_p = np.take_along_axis(pi, perm, axis=-1)
    mu_p = np.take_along_axis(mu, perm, axis=-1)
    sigma_p = np.take_along_axis(sigma, perm, axis=-1)

    ext_p = extract_gmm_modes(pi_p, mu_p, sigma_p)
    field_p = solve_value_mrf(pi_p, mu_p, sigma_p, ext_p, 0.2, rng=np.random.default_rng(3)).field

    # Sorted modes per cell are identical (mean-shift is symmetric in k).
    assert np.allclose(ext.mode_values, ext_p.mode_values)
    assert np.array_equal(ext.mode_counts, ext_p.mode_counts)
    assert np.allclose(field, field_p)


# --- beta = 0 limit ---------------------------------------------------------


def test_beta_zero_selects_the_unary_best_mode_field():
    pi, mu, sigma = _headline_toy()
    ext = extract_gmm_modes(pi, mu, sigma)
    n = ext.mode_values.shape[0]

    res = solve_value_mrf(pi, mu, sigma, ext, 0.0, n_restarts=4, rng=np.random.default_rng(1))

    warm_best = _unary_best_assignment(ext)
    expected = ext.mode_values[np.arange(n), warm_best].reshape(ext.height, ext.width)
    assert np.allclose(res.field, expected)
    # beta = 0 sits on the highest-density mode of every cell, so its drift to the
    # best mode is exactly zero.
    delta = delta_nll_to_best_mode(res.field, pi, mu, sigma, ext.mode_values, ext.valid_mask)
    assert delta["max"] < 1e-9


# --- the load-bearing unary property ---------------------------------------


def test_unary_prevents_low_density_compromise():
    """The unary stops the MRF taking a value-compatible but low-density mode.

    A 1x3 chain: the two ends sit at a single mode at 0; the middle cell has two
    modes, a low-density one near its neighbours (value ~0) and the high-density
    mode far away (value ~3). With the unary the middle keeps the high-density
    mode; with the unary zeroed the pairwise term alone drags it onto the
    low-density compromise.
    """

    pi = np.array([[[0.5, 0.5], [0.2, 0.8], [0.5, 0.5]]], dtype=float)
    mu = np.array([[[0.0, 0.0], [0.0, 3.0], [0.0, 0.0]]], dtype=float)
    sigma = np.ones((1, 3, 2), dtype=float)

    ext = extract_gmm_modes(pi, mu, sigma)
    # End cells have a single mode at 0; the middle cell is genuinely bimodal.
    assert ext.mode_counts[0] == 1 and ext.mode_counts[2] == 1
    assert ext.mode_counts[1] == 2

    beta = 0.02
    with_unary = solve_value_mrf(pi, mu, sigma, ext, beta, rng=np.random.default_rng(0)).field
    ext_no_unary = replace(ext, mode_unary=np.where(ext.valid_mask, 0.0, np.inf))
    without_unary = solve_value_mrf(
        pi, mu, sigma, ext_no_unary, beta, rng=np.random.default_rng(0)
    ).field

    # With the unary the middle cell keeps its high-density mode (~3); without it
    # the MRF compromises onto the low-density neighbour-compatible mode (~0).
    assert with_unary[0, 1] > 2.0
    assert without_unary[0, 1] < 1.0


# --- shared ICM core reproduces a* -----------------------------------------


def test_value_space_icm_reproduces_astar_on_component_means():
    """Component means + zero unary + unit pairwise weight == the a* baseline."""

    pi, mu, _ = _headline_toy()
    height, width, k = mu.shape
    n = height * width
    edges = grid_edges_8(height, width)

    mu_flat = mu.reshape(n, k)
    pi_flat = pi.reshape(n, k)
    valid_mask = np.ones((n, k), dtype=bool)
    unary = np.zeros((n, k))
    # a*'s warm start: highest pi, ties broken by smallest mean value (C5).
    warm = np.array([_argmin_by_value(-pi_flat[i], mu_flat[i]) for i in range(n)])

    assignment = _value_space_icm(
        mu_flat, valid_mask, unary, edges, warm, pairwise_weight=1.0
    )
    field = mu_flat[np.arange(n), assignment].reshape(height, width)

    astar_field, _ = smoothest_mode_assignment(pi, mu)
    assert np.array_equal(field, astar_field)


# --- restart reproducibility + finite scores --------------------------------


def test_seeded_restarts_reproducible_and_scores_finite():
    pi, mu, sigma = _headline_toy()
    betas = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]

    points_a = beta_sweep(pi, mu, sigma, betas, n_restarts=4, seed=0)
    points_b = beta_sweep(pi, mu, sigma, betas, n_restarts=4, seed=0)
    assert [p.beta for p in points_a] == betas

    for pa, pb in zip(points_a, points_b):
        assert np.array_equal(pa.field, pb.field)
        assert np.array_equal(pa.assignment, pb.assignment)

    for p in points_a:
        assert np.isfinite(p.nll_over_n)
        assert np.isfinite(p.energy)
        assert p.restart_spread >= 0.0
        assert p.restart_field_spread >= 0.0
        assert p.field.shape == (8, 8)
        # The Stage C field must score finite under the Stage A scorer too.
        scores = score_field(p.field, pi, mu, sigma)
        assert np.isfinite(scores["nll_over_n"])
        for key in ("mean", "p95", "max"):
            assert np.isfinite(p.delta_to_mode[key])
        assert p.delta_to_mode["thresholds"] == DRIFT_THRESHOLDS_V1


def test_solve_exposes_restart_spread():
    pi, mu, sigma = _headline_toy()
    ext = extract_gmm_modes(pi, mu, sigma)
    res = solve_value_mrf(pi, mu, sigma, ext, 0.5, n_restarts=4, rng=np.random.default_rng(0))
    assert res.restart_energies.shape == (4,)
    assert np.all(np.isfinite(res.restart_energies))
    assert res.restart_spread >= 0.0
    assert res.restart_field_spread >= 0.0


# --- the non-smearing diagnostic is a cross-method reference ----------------


def test_delta_to_mode_is_nonnegative_and_method4_beats_mode_map():
    """ΔNLL-to-best-mode >= 0 always, and Method 4 drifts less than mode_map.

    The metric is the drift of a field below the global GMM peak among the
    extracted modes, so it is non-negative for *any* field. Method 4 sits on
    modes (small drift); the per-cell mode field (component means) already drifts
    further off the true peaks.
    """

    pi, mu, sigma = _headline_toy()
    ext = extract_gmm_modes(pi, mu, sigma)
    res = solve_value_mrf(pi, mu, sigma, ext, 0.2, rng=np.random.default_rng(0))

    method4 = delta_nll_to_best_mode(res.field, pi, mu, sigma, ext.mode_values, ext.valid_mask)
    mode_f, _ = mode_field(pi, mu, sigma)
    modemap = delta_nll_to_best_mode(mode_f, pi, mu, sigma, ext.mode_values, ext.valid_mask)

    assert method4["max"] >= -1e-9
    assert modemap["max"] >= -1e-9
    assert method4["mean"] <= modemap["mean"] + 1e-9


# --- optional: monotonicity + true maxima -----------------------------------


def test_icm_does_not_increase_energy_from_warm_start():
    """ICM never raises J_beta above its unary-best warm start."""

    pi, mu, sigma = _headline_toy()
    edges = grid_edges_8(8, 8)
    ext = extract_gmm_modes(pi, mu, sigma)

    warm_best = _unary_best_assignment(ext)
    for beta in (0.05, 0.2, 1.0):
        warm_energy = value_mrf_energy(warm_best, ext, edges, beta)
        res = solve_value_mrf(pi, mu, sigma, ext, beta, rng=np.random.default_rng(0))
        assert res.energy <= warm_energy + 1e-9


def test_extracted_modes_are_true_density_maxima():
    """At each extracted mode the GMM density gradient vanishes (stationary)."""

    pi, mu, sigma = _headline_toy()
    height, width, k = mu.shape
    n = height * width
    ext = extract_gmm_modes(pi, mu, sigma)

    pi_f = pi.reshape(n, k)
    mu_f = mu.reshape(n, k)
    sigma_f = sigma.reshape(n, k)
    for i in range(n):
        for j in range(ext.mode_counts[i]):
            v = ext.mode_values[i, j]
            comp = pi_f[i] * normal_pdf(v, mu_f[i], sigma_f[i])
            grad_p = np.sum(comp * (mu_f[i] - v) / sigma_f[i] ** 2)
            assert abs(grad_p) < 1e-6
