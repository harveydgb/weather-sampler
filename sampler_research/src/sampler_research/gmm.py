"""Small GMM helper functions used by the existing notebooks."""

import numpy as np


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

    flat_pi = pi.reshape(-1, pi.shape[-1])
    flat_components = np.array([rng.choice(pi.shape[-1], p=row) for row in flat_pi])
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
