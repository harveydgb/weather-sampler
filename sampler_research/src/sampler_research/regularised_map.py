"""Method 1 — regularised MAP / Lagrangian (Phase 2 Stage B).

Implements the *swept-lambda regularised* form decided in
phase_2_research_plan.md §3.3 / §4.1 (the strict constrained/KKT form and the
Langevin layer are explicitly out of scope — §4.1.7). The objective, in the
per-cell / per-edge units of §3.1a, is

    J_lambda(x) = NLL(x)/N + lambda * (x^T L x) / |E_8|

with the closed-form gradient of §4.1.3:

    d NLL / d x_i = - sum_k r_ik * (mu_ik - x_i) / sigma_ik^2
    r_ik = pi_ik N(x_i; mu_ik, sigma_ik^2) / p_i(x_i)         (GMM responsibilities)
    d S   / d x   = 2 L x

Both terms carry their J_lambda normalisers (1/N on the NLL term, lambda/|E_8|
on the smoothness term), so the gradient returned here is exactly grad J_lambda
and matches a finite-difference gradient of `objective` (see the test suite).

This module reuses the Stage A machinery rather than reinventing it: the per-cell
GMM density (`mixture_pdf` / `gmm_nll_over_n`), the `E_8` graph + Laplacian and
roughness metrics (`sampler_research.graph`), and `mode_field` as the warm start
(the lambda -> 0 anchor, §4.1.4). It does *not* touch the optimiser-free Stage A
baselines or load any debug arrays.
"""

from dataclasses import dataclass, field

import numpy as np

from sampler_research.baselines import gmm_nll_over_n, mode_field
from sampler_research.gmm import mixture_pdf, normal_pdf
from sampler_research.graph import (
    graph_laplacian,
    grid_edges_8,
    roughness_sum,
    scale_free_roughness,
)


def nll_gradient(field_grid, pi, mu, sigma):
    """Closed-form gradient of the *raw* summed NLL, d(-sum_i log p_i)/d x_i.

    Shapes: `field_grid` is `[H, W]`; `pi`, `mu`, `sigma` are `[H, W, K]`.
    Returns a `[H, W]` array. This is the gradient of `NLL = -sum_i log p_i`
    (no 1/N factor); `objective_gradient` applies the J_lambda normaliser.
    """

    field_grid = np.asarray(field_grid, dtype=float)
    pi = np.asarray(pi, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # Component densities and responsibilities r_ik = pi_ik N_ik / p_i.
    comp = pi * normal_pdf(field_grid[..., None], mu, sigma)  # [H, W, K]
    density = np.sum(comp, axis=-1)  # p_i(x_i), == mixture_pdf(...)
    resp = comp / density[..., None]  # [H, W, K]

    # d(-log p_i)/dx_i = - sum_k r_ik (mu_ik - x_i)/sigma_ik^2.
    per_component = resp * (mu - field_grid[..., None]) / sigma**2
    return -np.sum(per_component, axis=-1)


def objective(field_grid, pi, mu, sigma, lam, laplacian=None, n_edges=None):
    """J_lambda(x) = NLL(x)/N + lambda * (x^T L x)/|E_8| (§3.1a / §4.1.3).

    `laplacian` and `n_edges` are accepted so a sweep/optimiser can build them
    once; both default to the `E_8` grid for the field's shape.
    """

    field_grid = np.asarray(field_grid, dtype=float)
    height, width = field_grid.shape
    if laplacian is None:
        laplacian = graph_laplacian(height, width)
    if n_edges is None:
        n_edges = len(grid_edges_8(height, width))

    nll_over_n = gmm_nll_over_n(field_grid, pi, mu, sigma)
    smooth = roughness_sum(field_grid, laplacian) / n_edges
    return nll_over_n + lam * smooth


def objective_gradient(field_grid, pi, mu, sigma, lam, laplacian=None, n_edges=None):
    """Gradient of `objective` w.r.t. the field, shape `[H, W]`.

    grad J_lambda = (1/N) * grad NLL + (lambda/|E_8|) * 2 L x. The NLL part is the
    closed form in `nll_gradient`; the smoothness part is the edge-mean-normalised
    `2 L x` so it matches the `(x^T L x)/|E_8|` term in `objective`.
    """

    field_grid = np.asarray(field_grid, dtype=float)
    height, width = field_grid.shape
    n = height * width
    if laplacian is None:
        laplacian = graph_laplacian(height, width)
    if n_edges is None:
        n_edges = len(grid_edges_8(height, width))

    grad_nll = nll_gradient(field_grid, pi, mu, sigma) / n
    smooth_grad_flat = 2.0 * (laplacian @ field_grid.reshape(-1)) / n_edges
    grad_smooth = smooth_grad_flat.reshape(height, width)
    return grad_nll + lam * grad_smooth


@dataclass
class OptimResult:
    """Outcome of one warm-started, multi-restart minimisation at a fixed lambda."""

    field: np.ndarray  # lowest-energy field found, [H, W]
    energy: float  # J_lambda of `field`
    nll_over_n: float  # NLL/N of `field`
    r_tilde: float  # scale-free roughness of `field`
    variance_collapsed: bool
    restart_energies: np.ndarray  # J_lambda reached by each restart
    restart_spread: float  # max - min over restart energies (energy units)
    restart_field_spread: float  # max over cells of (max - min) restart field value
    n_restarts: int


def _adam_descent(x0, grad_fn, n_steps, lr, betas=(0.9, 0.999), eps=1e-8):
    """Minimise via Adam from `x0`. `grad_fn(x)->grad` has x's shape. Returns x."""

    x = x0.astype(float).copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2 = betas
    for t in range(1, n_steps + 1):
        g = grad_fn(x)
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * g * g
        m_hat = m / (1.0 - b1**t)
        v_hat = v / (1.0 - b2**t)
        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
    return x


def minimise_at_lambda(
    pi,
    mu,
    sigma,
    lam,
    *,
    n_restarts=4,
    n_steps=400,
    lr=0.05,
    restart_scale=1.0,
    rng=None,
    laplacian=None,
    n_edges=None,
    edges=None,
):
    """Minimise J_lambda with an Adam optimiser, warm-started from `mode_field`.

    Restart 0 is the bare mode-field warm start (§4.1.4); the remaining restarts
    add zero-mean Gaussian jitter (std `restart_scale`) to the mode field so the
    non-convex landscape is probed from a few nearby basins. The lowest-energy
    field is kept; `restart_spread` (energy range) and `restart_field_spread`
    (worst per-cell value range across restarts) are exposed as stability outputs.
    """

    pi = np.asarray(pi, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    height, width, _ = mu.shape

    if laplacian is None:
        laplacian = graph_laplacian(height, width)
    if edges is None:
        edges = grid_edges_8(height, width)
    if n_edges is None:
        n_edges = len(edges)
    rng = rng or np.random.default_rng()

    warm, _ = mode_field(pi, mu, sigma)

    def grad_fn(x):
        return objective_gradient(x, pi, mu, sigma, lam, laplacian=laplacian, n_edges=n_edges)

    fields = []
    energies = []
    for r in range(n_restarts):
        x0 = warm.copy() if r == 0 else warm + rng.normal(scale=restart_scale, size=warm.shape)
        x_opt = _adam_descent(x0, grad_fn, n_steps=n_steps, lr=lr)
        fields.append(x_opt)
        energies.append(
            objective(x_opt, pi, mu, sigma, lam, laplacian=laplacian, n_edges=n_edges)
        )

    fields = np.stack(fields)  # [R, H, W]
    energies = np.asarray(energies)
    best = int(np.argmin(energies))
    best_field = fields[best]

    r_tilde, collapsed = scale_free_roughness(best_field, edges)
    # Worst-case disagreement between restart fields, per cell, then max over cells.
    field_spread = float(np.max(fields.max(axis=0) - fields.min(axis=0)))

    return OptimResult(
        field=best_field,
        energy=float(energies[best]),
        nll_over_n=gmm_nll_over_n(best_field, pi, mu, sigma),
        r_tilde=r_tilde,
        variance_collapsed=collapsed,
        restart_energies=energies,
        restart_spread=float(energies.max() - energies.min()),
        restart_field_spread=field_spread,
        n_restarts=n_restarts,
    )


@dataclass
class LambdaSweepPoint:
    """One row of the lambda-sweep Pareto curve."""

    lam: float
    nll_over_n: float
    r_tilde: float
    variance_collapsed: bool
    restart_spread: float
    restart_field_spread: float
    energy: float
    field: np.ndarray = field(repr=False)


def lambda_sweep(
    pi,
    mu,
    sigma,
    lambdas,
    *,
    n_restarts=4,
    n_steps=400,
    lr=0.05,
    restart_scale=1.0,
    seed=0,
):
    """Produce the Pareto curve (NLL/N vs scale-free R̃) over a lambda grid.

    Each lambda is solved by `minimise_at_lambda` from its own freshly-seeded RNG
    (seed offset by the lambda index) so the sweep is reproducible and order
    independent. Returns a list of `LambdaSweepPoint`, one per lambda.
    """

    pi = np.asarray(pi, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    height, width, _ = mu.shape

    laplacian = graph_laplacian(height, width)
    edges = grid_edges_8(height, width)
    n_edges = len(edges)

    points = []
    for idx, lam in enumerate(lambdas):
        rng = np.random.default_rng(seed + idx)
        res = minimise_at_lambda(
            pi,
            mu,
            sigma,
            lam,
            n_restarts=n_restarts,
            n_steps=n_steps,
            lr=lr,
            restart_scale=restart_scale,
            rng=rng,
            laplacian=laplacian,
            n_edges=n_edges,
            edges=edges,
        )
        points.append(
            LambdaSweepPoint(
                lam=float(lam),
                nll_over_n=res.nll_over_n,
                r_tilde=res.r_tilde,
                variance_collapsed=res.variance_collapsed,
                restart_spread=res.restart_spread,
                restart_field_spread=res.restart_field_spread,
                energy=res.energy,
                field=res.field,
            )
        )
    return points
