import numpy as np

from sampler_research.diagnostics import empirical_variogram, spherical_pair_dist_deg
from sampler_research.gmm import mixture_mean, mixture_pdf, normal_pdf, sample_iid_gmm


def test_mixture_mean_and_pdf_shapes() -> None:
    pi = np.array([[0.25, 0.75], [0.5, 0.5]])
    mu = np.array([[0.0, 2.0], [1.0, 3.0]])
    sigma = np.ones_like(mu)
    x = np.array([0.0, 1.0])

    assert np.allclose(mixture_mean(pi, mu), np.array([1.5, 2.0]))
    assert normal_pdf(x, x, np.ones_like(x)).shape == (2,)
    assert mixture_pdf(x, pi, mu, sigma).shape == (2,)
    assert np.all(mixture_pdf(x, pi, mu, sigma) > 0)


def test_sample_iid_gmm_returns_field_and_components() -> None:
    pi = np.full((3, 2, 4), 0.25)
    mu = np.zeros((3, 2, 4))
    sigma = np.ones((3, 2, 4))

    sample, components = sample_iid_gmm(pi, mu, sigma, np.random.default_rng(0))

    assert sample.shape == (3, 2)
    assert components.shape == (3, 2)
    assert components.min() >= 0
    assert components.max() < 4


def test_empirical_variogram_constant_field_is_zero() -> None:
    coords = np.stack(np.meshgrid(np.arange(2), np.arange(2), indexing="ij"), axis=-1)
    centres, variograms = empirical_variogram(coords, {"constant": np.ones((2, 2))}, n_bins=3)

    assert centres.shape == (3,)
    assert np.nanmax(variograms["constant"]) == 0.0


def test_spherical_pair_distance_zero_for_same_points() -> None:
    dist = spherical_pair_dist_deg(
        np.array([10.0, -20.0]),
        np.array([30.0, 40.0]),
        np.array([10.0, -20.0]),
        np.array([30.0, 40.0]),
    )

    assert np.allclose(dist, 0.0)
