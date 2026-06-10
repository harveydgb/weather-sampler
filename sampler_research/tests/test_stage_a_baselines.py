import numpy as np

from sampler_research.graph import (
    graph_laplacian,
    grid_edges_8,
    laplacian_blur,
    roughness_edge_mean,
    roughness_sum,
    scale_free_roughness,
)
from sampler_research.baselines import (
    gmm_nll_over_n,
    mixture_mean_field,
    mode_field,
    score_field,
    smoothed_map_baseline,
    smoothest_mode_assignment,
    variance_scaled_baseline,
)
from sampler_research.toy import Phase1ToyConfig, make_phase1_toy


def _headline_toy():
    toy = make_phase1_toy(Phase1ToyConfig())
    return toy["pi"], toy["mu"], toy["sigma"]


def test_e8_edge_count_and_laplacian_structure() -> None:
    edges = grid_edges_8(8, 8)
    assert len(edges) == 210
    assert np.all(edges[:, 0] < edges[:, 1])

    laplacian = graph_laplacian(8, 8)
    assert laplacian.shape == (64, 64)
    assert np.allclose(laplacian, laplacian.T)
    assert np.allclose(laplacian.sum(axis=1), 0.0)


def test_roughness_definitions_agree() -> None:
    edges = grid_edges_8(8, 8)
    laplacian = graph_laplacian(8, 8)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8, 8))

    edge_sum = np.sum((x.reshape(-1)[edges[:, 0]] - x.reshape(-1)[edges[:, 1]]) ** 2)
    assert np.isclose(roughness_sum(x, laplacian), edge_sum)
    assert np.isclose(roughness_edge_mean(x, edges), edge_sum / len(edges))


def test_constant_field_is_flagged_variance_collapsed() -> None:
    edges = grid_edges_8(8, 8)
    laplacian = graph_laplacian(8, 8)
    const = np.full((8, 8), 3.0)

    assert np.isclose(roughness_sum(const, laplacian), 0.0)
    value, collapsed = scale_free_roughness(const, edges)
    assert collapsed
    assert np.isfinite(value)


def test_laplacian_blur_is_mean_preserving() -> None:
    laplacian = graph_laplacian(8, 8)
    rng = np.random.default_rng(1)
    x = rng.normal(size=(8, 8))
    blurred = laplacian_blur(x, laplacian, step=0.1, n_iters=20)

    assert np.isclose(blurred.mean(), x.mean())
    assert roughness_sum(blurred, laplacian) < roughness_sum(x, laplacian)


def test_gmm_nll_lower_at_mode_than_off_mode() -> None:
    pi, mu, sigma = _headline_toy()
    field, _ = mode_field(pi, mu, sigma)
    off = field + 5.0
    assert gmm_nll_over_n(field, pi, mu, sigma) < gmm_nll_over_n(off, pi, mu, sigma)


def test_iid_is_rougher_than_coherent_baselines() -> None:
    pi, mu, sigma = _headline_toy()
    edges = grid_edges_8(8, 8)
    rng = np.random.default_rng(0)

    iid, _ = variance_scaled_baseline(pi, mu, sigma, alpha=1.0, rng=rng)
    astar, _ = smoothest_mode_assignment(pi, mu)
    smoothed = smoothed_map_baseline(pi, mu, sigma)

    r_iid = score_field(iid, pi, mu, sigma, edges)["r_tilde"]
    assert score_field(astar, pi, mu, sigma, edges)["r_tilde"] < r_iid
    assert score_field(smoothed, pi, mu, sigma, edges)["r_tilde"] < r_iid


def test_astar_is_at_most_as_rough_as_per_cell_mode() -> None:
    pi, mu, sigma = _headline_toy()
    edges = grid_edges_8(8, 8)
    mode_f, _ = mode_field(pi, mu, sigma)
    astar, _ = smoothest_mode_assignment(pi, mu)

    assert score_field(astar, pi, mu, sigma, edges)["r_tilde"] <= score_field(
        mode_f, pi, mu, sigma, edges
    )["r_tilde"] + 1e-9


def test_astar_couples_on_values_not_label_index() -> None:
    pi, mu, _ = _headline_toy()
    astar, _ = smoothest_mode_assignment(pi, mu)

    rng = np.random.default_rng(7)
    perm = np.stack([[rng.permutation(pi.shape[-1]) for _ in range(8)] for _ in range(8)])
    pi_p = np.take_along_axis(pi, perm, axis=-1)
    mu_p = np.take_along_axis(mu, perm, axis=-1)
    astar_relabelled, _ = smoothest_mode_assignment(pi_p, mu_p)

    assert np.allclose(astar, astar_relabelled)


def test_variance_scaled_rejects_invalid_alpha() -> None:
    pi, mu, sigma = _headline_toy()
    for bad_alpha in (0.0, -0.5, 1.5):
        try:
            variance_scaled_baseline(pi, mu, sigma, alpha=bad_alpha)
        except ValueError:
            continue
        raise AssertionError(f"alpha={bad_alpha} should have raised")


def test_mixture_mean_field_matches_helper() -> None:
    pi, mu, _ = _headline_toy()
    field = mixture_mean_field(pi, mu)
    assert field.shape == (8, 8)
    assert np.allclose(field, np.sum(pi * mu, axis=-1))
