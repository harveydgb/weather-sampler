"""Marginal faithfulness and do-no-harm checks vs the emitted GMMs (report D6/S5.5).

The project's core principle: a coherent joint sampler must keep its per-location
marginals faithful to the head's emitted GMMs -- smoothness is a selection
criterion over fields that already agree with the marginals, never added skill.
This module supplies the numerics for the S5.5 build, none of which existed in
the tree before (only `gmm.gmm_log_pdf` and `gmm.gmm_cdf`).

Two notions, kept rigorously separate (report non-claim #3 -- no ensemble
calibration claim):

* **Marginal position (deterministic fields, e.g. Method 1 @ lambda*).** For a
  single produced field, `pit_values` returns where each cell's selected value
  sits in its own marginal CDF. This is a *marginal-position / coverage*
  diagnostic -- it shows the field stays inside GMM mass and drifts toward the
  tails as lambda pushes off-mode. It is NOT an ensemble-calibration statement.

* **Do-no-harm (the stochastic iid baseline only).** `delta_crps_iid` compares
  the CRPS of an iid sampler ensemble (draws from the emitted GMM) against the
  analytic GMM CRPS. Because the iid sampler draws from the emitted marginal,
  this is ~0 by construction -- the do-no-harm baseline, not a model claim. PIT
  of iid draws from their own GMM is exactly Uniform(0,1) (`pit_values` of a draw).

The mixture CRPS is the Grimit et al. (2006) closed form; `ensemble_crps` uses
the fair (bias-corrected) estimator so finite-ensemble CRPS is unbiased for the
analytic value, making `delta_crps_iid ~ 0` hold at moderate ensemble sizes.
"""

import numpy as np
from scipy.special import ndtr

from sampler_research.gmm import gmm_cdf, sample_iid_gmm

# Central-interval coverage levels reported by default (marginal-position diag).
COVERAGE_LEVELS = (0.5, 0.9)


def pit_values(field, pi, mu, sigma):
    """PIT values `F_i(x_i)` of a produced field under each cell's GMM `[N]`.

    For an iid draw from the same GMM these are exactly Uniform(0, 1) (do-no-harm
    identity); for a deterministic field they are a marginal-position diagnostic.
    """

    return gmm_cdf(np.asarray(field, dtype=float), pi, mu, sigma)


def quantile_coverage(pit, levels=COVERAGE_LEVELS):
    """Central-interval coverage of PIT values, `{level: fraction in band}`.

    For nominal level `q`, the fraction of cells with PIT in the central band
    `[(1-q)/2, (1+q)/2]`. An ideal *stochastic* sampler matches `q`; for a
    deterministic field this is the fraction whose selected value lies within the
    central `q`-mass of its marginal (marginal position, not calibration).
    """

    pit = np.asarray(pit, dtype=float)
    out = {}
    for q in levels:
        lo, hi = (1.0 - q) / 2.0, (1.0 + q) / 2.0
        out[q] = float(np.mean((pit >= lo) & (pit <= hi)))
    return out


def ks_uniform(pit):
    """Kolmogorov-Smirnov distance of PIT values from Uniform(0, 1).

    `sup_x |F_emp(x) - x|`. ~0 for a faithful stochastic sampler; reported for
    the iid baseline as the do-no-harm check, and for deterministic fields only
    as a descriptive marginal-spread number (never as a calibration test).
    """

    pit = np.sort(np.asarray(pit, dtype=float))
    n = pit.size
    if n == 0:
        return float("nan")
    upper = np.arange(1, n + 1) / n
    lower = np.arange(0, n) / n
    return float(np.max(np.maximum(upper - pit, pit - lower)))


def mean_abs_pit_centre(pit):
    """Mean `|PIT - 0.5|` -- a scalar off-centre summary (0.25 for Uniform)."""

    return float(np.mean(np.abs(np.asarray(pit, dtype=float) - 0.5)))


def _e_abs_normal(m, s):
    """`E|X|` for `X ~ N(m, s^2)` (folded-normal mean), vectorised, `s > 0`.

    `E|X| = m (2 Phi(m/s) - 1) + s sqrt(2/pi) exp(-m^2 / (2 s^2))`.
    """

    m = np.asarray(m, dtype=float)
    s = np.asarray(s, dtype=float)
    r = m / s
    return m * (2.0 * ndtr(r) - 1.0) + s * np.sqrt(2.0 / np.pi) * np.exp(-0.5 * r * r)


def gmm_crps_analytic(y, pi, mu, sigma):
    """Closed-form CRPS of each cell's Gaussian-mixture predictive vs `y` `[N]`.

    Grimit et al. (2006):
        CRPS = sum_k pi_k E|mu_k - y|
               - 0.5 sum_{k,l} pi_k pi_l E|mu_k - mu_l|,
    with the two expectations folded-normal means at variances `sigma_k^2` and
    `sigma_k^2 + sigma_l^2` respectively. `y` is `[N]`; `pi`/`mu`/`sigma` `[N, K]`.
    """

    y = np.asarray(y, dtype=float)
    pi = np.asarray(pi, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    term1 = np.sum(pi * _e_abs_normal(mu - y[..., None], sigma), axis=-1)
    m_kl = mu[..., :, None] - mu[..., None, :]
    s_kl = np.sqrt(sigma[..., :, None] ** 2 + sigma[..., None, :] ** 2)
    w_kl = pi[..., :, None] * pi[..., None, :]
    term2 = 0.5 * np.sum(w_kl * _e_abs_normal(m_kl, s_kl), axis=(-2, -1))
    return term1 - term2


def ensemble_crps(samples, y):
    """Fair (bias-corrected) empirical CRPS of an ensemble vs `y`, per cell `[N]`.

    `samples` is `[M, N]` (M ensemble members), `y` is `[N]`. Uses the standard
    fair estimator
        CRPS = (1/M) sum_m |x_m - y|
               - 1 / (M (M-1)) * sum_{m<m'} |x_m - x_{m'}|,
    evaluated via the sorted-order Gini identity (O(M log M) per cell, no
    `[M, M, N]` tensor). Unbiased for the analytic CRPS when `samples ~ F`, so
    `delta_crps_iid` is ~0 even at moderate `M`. Requires `M >= 2`.
    """

    samples = np.asarray(samples, dtype=float)
    y = np.asarray(y, dtype=float)
    m = samples.shape[0]
    if m < 2:
        raise ValueError("ensemble_crps needs at least 2 members")
    mean_abs = np.mean(np.abs(samples - y[None, :]), axis=0)
    xs = np.sort(samples, axis=0)
    i = np.arange(1, m + 1)[:, None]
    gini = np.sum((2 * i - m - 1) * xs, axis=0)  # = sum_{m<m'} (x_(m') - x_(m))
    return mean_abs - gini / (m * (m - 1))


def gmm_ensemble(pi, mu, sigma, n_members, rng=None):
    """Stack `n_members` independent iid GMM draws into an `[M, N]` ensemble."""

    rng = rng or np.random.default_rng()
    draws = [sample_iid_gmm(pi, mu, sigma, rng)[0] for _ in range(int(n_members))]
    return np.stack(draws, axis=0)


def delta_crps_iid(pi, mu, sigma, n_members=50, rng=None):
    """Do-no-harm: mean(ensemble CRPS) - mean(analytic CRPS) for the iid baseline.

    Draws one pseudo-observation per cell from the emitted GMM, builds an
    independent `n_members`-member iid ensemble, and returns
    `{"ensemble_crps", "analytic_crps", "delta_crps"}` (means over cells). For the
    iid sampler this is ~0 by construction -- the stochastic do-no-harm baseline.
    It is meaningful ONLY for a stochastic ensemble; never call it on a single
    deterministic field (that would conflate marginal position with calibration).
    """

    rng = rng or np.random.default_rng(0)
    pi = np.asarray(pi, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    y, _ = sample_iid_gmm(pi, mu, sigma, rng)  # pseudo-obs ~ F, independent of ens
    samples = gmm_ensemble(pi, mu, sigma, n_members, rng)
    ens = float(np.mean(ensemble_crps(samples, y)))
    ana = float(np.mean(gmm_crps_analytic(y, pi, mu, sigma)))
    return {"ensemble_crps": ens, "analytic_crps": ana, "delta_crps": ens - ana,
            "n_members": int(n_members)}


def field_faithfulness(field, pi, mu, sigma, levels=COVERAGE_LEVELS):
    """Marginal-position summary for one produced (typically deterministic) field.

    Returns `{"pit_ks", "mean_abs_pit_centre", "cov{q}"...}`. Labelled
    marginal-position, not ensemble calibration (report non-claim #3).
    """

    pit = pit_values(field, pi, mu, sigma)
    cov = quantile_coverage(pit, levels)
    out = {"pit_ks": ks_uniform(pit), "mean_abs_pit_centre": mean_abs_pit_centre(pit)}
    for q, frac in cov.items():
        out[f"cov{int(round(q * 100))}"] = frac
    return out
