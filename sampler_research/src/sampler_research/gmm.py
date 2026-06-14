"""Small GMM helper functions used by the existing notebooks."""

import numpy as np
from scipy.special import ndtr


def as_numpy(value):
    """Convert NumPy arrays or torch-like tensors to NumPy arrays."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def normal_pdf(x, mu, sigma):
    """Evaluate a univariate normal density."""

    x = np.asarray(x)
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def mixture_pdf(x, pi, mu, sigma):
    """Evaluate a univariate GMM density at `x`."""

    return np.sum(np.asarray(pi) * normal_pdf(np.asarray(x)[..., None], mu, sigma), axis=-1)


def gmm_log_pdf(x, pi, mu, sigma):
    """Log GMM density `log p(x)` via a max-shifted logsumexp (phase_4_plan §1).

    Shape-agnostic: the mixture axis is the final axis of `pi`/`mu`/`sigma` and
    `x` carries every leading axis (`[N]` or `[H, W]`). Equals
    `log(mixture_pdf(...))` wherever the linear-space density does not
    underflow, and stays finite far off-mode (e.g. a 40-sigma probe) where the
    float64 linear density is a true zero. Dead components (`pi == 0`)
    contribute `-inf` log terms and drop out of the logsumexp.
    """

    x = np.asarray(x, dtype=float)
    pi = np.asarray(pi, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    z = (x[..., None] - mu) / sigma
    with np.errstate(divide="ignore"):
        log_comp = np.log(pi) - np.log(sigma) - 0.5 * z * z - 0.5 * np.log(2.0 * np.pi)
    shift = np.max(log_comp, axis=-1, keepdims=True)
    shift = np.where(np.isfinite(shift), shift, 0.0)
    with np.errstate(divide="ignore"):
        return shift[..., 0] + np.log(np.sum(np.exp(log_comp - shift), axis=-1))


def gmm_cdf(x, pi, mu, sigma):
    """GMM cumulative distribution `F(x) = sum_k pi_k Phi((x - mu_k)/sigma_k)`.

    Shape-agnostic in the same convention as `gmm_log_pdf`: the mixture axis is
    the final axis of `pi`/`mu`/`sigma`, and `x` carries every leading axis
    (`[N]` or `[H, W]`). Returns values in `[0, 1]`, monotone non-decreasing in
    `x`; its `x`-derivative is `mixture_pdf` (used for the finite-difference
    test). The building block for PIT values and quantile coverage
    (`faithfulness.pit_values`). Dead components (`pi == 0`) contribute nothing.
    """

    x = np.asarray(x, dtype=float)
    pi = np.asarray(pi, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    z = (x[..., None] - mu) / sigma
    return np.sum(pi * ndtr(z), axis=-1)


def mixture_mean(pi, mu):
    """Return the per-location mixture mean."""

    return np.sum(np.asarray(pi) * np.asarray(mu), axis=-1)


def sample_iid_gmm(
    pi,
    mu,
    sigma,
    rng=None,
):
    """Draw one independent sample from each local GMM.

    Returns `(sample, component_index)`. The mixture axis is assumed to be the
    final axis, so the function works for both `[N, K]` and `[H, W, K]` arrays.
    """

    rng = rng or np.random.default_rng()
    pi = np.asarray(pi, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    if pi.shape != mu.shape or pi.shape != sigma.shape:
        raise ValueError("pi, mu, and sigma must have matching shapes")
    if pi.ndim < 1:
        raise ValueError("pi must have at least one dimension")
    if not np.allclose(pi.sum(axis=-1), 1.0):
        raise ValueError("mixture weights must sum to 1 along the final axis")

    # Vectorised inverse-CDF component draw (one uniform per location); replaces
    # the per-row `rng.choice` loop, which was the only O(N)-python part.
    flat_pi = pi.reshape(-1, pi.shape[-1])
    cdf = np.cumsum(flat_pi, axis=-1)
    u = rng.random(flat_pi.shape[0])
    flat_components = np.minimum(
        np.sum(u[:, None] > cdf, axis=-1), pi.shape[-1] - 1
    ).astype(np.int64)
    component_index = flat_components.reshape(pi.shape[:-1])

    flat_mu = mu.reshape(-1, mu.shape[-1])
    flat_sigma = sigma.reshape(-1, sigma.shape[-1])
    flat_location = np.arange(flat_components.size)
    chosen_mu = flat_mu[flat_location, flat_components]
    chosen_sigma = flat_sigma[flat_location, flat_components]
    sample = rng.normal(chosen_mu, chosen_sigma).reshape(pi.shape[:-1])

    return sample, component_index


def reference_point_by_median_mean(mean_field):
    """Return the flat index closest to the median mixture mean."""

    flat = np.asarray(mean_field).reshape(-1)
    return int(np.argmin(np.abs(flat - np.median(flat))))
