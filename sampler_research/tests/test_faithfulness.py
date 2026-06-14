"""Faithfulness numerics: gmm_cdf, analytic vs MC CRPS, PIT uniformity, dCRPS~0.

These guard the S5.5 do-no-harm code, whose discipline is subtle: the iid
ensemble checks (delta CRPS ~ 0, PIT ~ Uniform) are do-no-harm baselines, while a
deterministic field's PIT is only a marginal-position diagnostic.
"""

import numpy as np
import pytest

from sampler_research import faithfulness as fth
from sampler_research.gmm import gmm_cdf, mixture_pdf, sample_iid_gmm


def _random_gmm(n, k, rng):
    pi = rng.dirichlet(np.ones(k), size=n)
    mu = rng.normal(0.0, 2.0, size=(n, k))
    sigma = rng.uniform(0.3, 1.5, size=(n, k))
    return pi, mu, sigma


def test_gmm_cdf_in_unit_interval_and_monotone():
    rng = np.random.default_rng(1)
    pi, mu, sigma = _random_gmm(200, 3, rng)
    xs = np.linspace(-8, 8, 50)
    prev = None
    for x in xs:
        c = gmm_cdf(np.full(200, x), pi, mu, sigma)
        assert np.all(c >= -1e-12) and np.all(c <= 1.0 + 1e-12)
        if prev is not None:
            assert np.all(c >= prev - 1e-12)  # non-decreasing in x
        prev = c
    # tails -> 0 and 1
    assert np.all(gmm_cdf(np.full(200, -50.0), pi, mu, sigma) < 1e-6)
    assert np.all(gmm_cdf(np.full(200, 50.0), pi, mu, sigma) > 1 - 1e-6)


def test_gmm_cdf_derivative_is_pdf():
    rng = np.random.default_rng(2)
    pi, mu, sigma = _random_gmm(64, 4, rng)
    x = rng.normal(0, 1, size=64)
    h = 1e-4
    fd = (gmm_cdf(x + h, pi, mu, sigma) - gmm_cdf(x - h, pi, mu, sigma)) / (2 * h)
    pdf = mixture_pdf(x, pi, mu, sigma)
    assert np.allclose(fd, pdf, atol=1e-5)


def test_e_abs_normal_matches_montecarlo():
    rng = np.random.default_rng(3)
    for m, s in [(0.0, 1.0), (2.0, 0.5), (-1.5, 2.0)]:
        draws = rng.normal(m, s, size=400_000)
        assert fth._e_abs_normal(np.array(m), np.array(s)) == pytest.approx(
            np.mean(np.abs(draws)), abs=5e-3
        )


def test_analytic_crps_matches_monte_carlo_mean():
    rng = np.random.default_rng(4)
    pi, mu, sigma = _random_gmm(1500, 3, rng)
    y = rng.normal(0, 2, size=1500)
    ana = fth.gmm_crps_analytic(y, pi, mu, sigma)
    samples = fth.gmm_ensemble(pi, mu, sigma, 6000, rng)
    mc = fth.ensemble_crps(samples, y)
    assert np.mean(ana) == pytest.approx(np.mean(mc), abs=1e-2)
    assert np.all(ana >= 0.0)


def test_ensemble_crps_fair_matches_bruteforce():
    rng = np.random.default_rng(5)
    m, n = 12, 7
    samples = rng.normal(size=(m, n))
    y = rng.normal(size=n)
    fast = fth.ensemble_crps(samples, y)
    # brute force fair estimator
    mean_abs = np.mean(np.abs(samples - y[None, :]), axis=0)
    pair = np.zeros(n)
    for a in range(m):
        for b in range(m):
            pair += np.abs(samples[a] - samples[b])
    brute = mean_abs - pair / (2 * m * (m - 1))
    assert np.allclose(fast, brute)


def test_pit_of_own_draw_is_uniform():
    rng = np.random.default_rng(6)
    pi, mu, sigma = _random_gmm(20000, 3, rng)
    draw, _ = sample_iid_gmm(pi, mu, sigma, rng)
    pit = fth.pit_values(draw, pi, mu, sigma)
    assert fth.ks_uniform(pit) < 0.03  # ~ Uniform(0,1)
    cov = fth.quantile_coverage(pit)
    assert cov[0.5] == pytest.approx(0.5, abs=0.03)
    assert cov[0.9] == pytest.approx(0.9, abs=0.02)


def test_delta_crps_iid_near_zero():
    rng = np.random.default_rng(7)
    pi, mu, sigma = _random_gmm(4000, 3, rng)
    out = fth.delta_crps_iid(pi, mu, sigma, n_members=100, rng=rng)
    assert abs(out["delta_crps"]) < 0.02  # do-no-harm
    assert out["analytic_crps"] > 0


def test_ks_uniform_known_values():
    # All PIT at 0.5 -> KS = 0.5; perfectly spread -> small
    assert fth.ks_uniform(np.full(1000, 0.5)) == pytest.approx(0.5, abs=1e-3)
    assert fth.ks_uniform(np.linspace(0, 1, 1001)) < 1e-2
