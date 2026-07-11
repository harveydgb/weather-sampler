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

Stage B-TV ablation (§4.1 field 6 / §8 #8): `objective`, `objective_gradient`,
`minimise_at_lambda`, and `lambda_sweep` accept a keyword-only
`penalty="quadratic" | "huber"`. The default `"quadratic"` runs the literal
existing code path (bit-identical to the pre-ablation behaviour); `"huber"`
swaps the per-edge quadratic for a Huber-smoothed total-variation penalty

    J_lambda(x) = NLL(x)/N + lambda * (1/|E_8|) sum_(i,j) rho_delta(x_i - x_j)
    rho_delta(d) = d^2/(2*delta) for |d| <= delta, |d| - delta/2 otherwise

so the same Adam machinery answers "does the penalty swap alone kill the
smearing?". The Huber edge-mean under-estimates the true TV edge-mean by at
most delta/2 per edge, so `J_huber <= J_TV <= J_huber + lambda*delta/2`; that
surrogate gap is part of the exact-solver certificate in `exact_map`.
"""

from dataclasses import dataclass, field

import numpy as np

from sampler_research.baselines import gmm_nll_over_n, mode_field
from sampler_research.gmm import normal_pdf
from sampler_research.graph import (
    graph_laplacian,
    grid_edges_8,
    roughness_sum,
    scale_free_roughness,
)
from sampler_research.spectral import spectral_roughness


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
    # Guard the responsibility divide: a far-off-mode underflow (density -> 0)
    # would give 0/0 = NaN. np.where returns exactly comp/density wherever
    # density > 0 (same float bits), so results do not move; only the
    # pathological underflow region changes NaN -> finite (zero) gradient.
    with np.errstate(invalid="ignore", divide="ignore"):
        resp = np.where(density[..., None] > 0.0, comp / density[..., None], 0.0)  # [H, W, K]

    # d(-log p_i)/dx_i = - sum_k r_ik (mu_ik - x_i)/sigma_ik^2.
    per_component = resp * (mu - field_grid[..., None]) / sigma**2
    return -np.sum(per_component, axis=-1)


def huber_rho(d, delta):
    """Huber-smoothed absolute value `rho_delta(d)` (the per-edge TV surrogate).

    Quadratic `d^2/(2*delta)` inside `|d| <= delta`, linear `|d| - delta/2`
    outside, so `0 <= |d| - rho_delta(d) <= delta/2` everywhere — the surrogate
    gap quoted in the module docstring.
    """

    d = np.asarray(d, dtype=float)
    abs_d = np.abs(d)
    return np.where(abs_d <= delta, d * d / (2.0 * delta), abs_d - 0.5 * delta)


def huber_penalty_edge_mean(field_grid, edges, delta):
    """Edge-averaged Huber-TV penalty `(1/|E|) sum_(i,j) rho_delta(x_i - x_j)`."""

    flat = np.asarray(field_grid, dtype=float).reshape(-1)
    edges = np.asarray(edges, dtype=np.int64)
    diffs = flat[edges[:, 0]] - flat[edges[:, 1]]
    return float(np.mean(huber_rho(diffs, delta)))


def huber_penalty_gradient(field_grid, edges, delta):
    """Gradient of `huber_penalty_edge_mean` w.r.t. the field, shape `[H, W]`.

    Per edge the influence function is `rho'(d) = clip(d/delta, -1, 1)`; it is
    scattered back onto the two endpoint cells (`+` on `i`, `-` on `j`) via
    `np.add.at` on flat row-major indices, then edge-mean normalised.
    """

    field_grid = np.asarray(field_grid, dtype=float)
    edges = np.asarray(edges, dtype=np.int64)
    flat = field_grid.reshape(-1)
    diffs = flat[edges[:, 0]] - flat[edges[:, 1]]
    psi = np.clip(diffs / delta, -1.0, 1.0)
    grad = np.zeros_like(flat)
    np.add.at(grad, edges[:, 0], psi)
    np.add.at(grad, edges[:, 1], -psi)
    return (grad / len(edges)).reshape(field_grid.shape)


def _default_grid_edges(field_grid):
    """Toy `E_8` edge default; flat `[N]` fields must pass `edges` explicitly."""

    if field_grid.ndim != 2:
        raise ValueError("flat fields require explicit edges (no E_8 default)")
    return grid_edges_8(*field_grid.shape)


def _default_grid_laplacian(field_grid):
    """Toy dense-`E_8` Laplacian default; flat fields must pass a sparse one."""

    if field_grid.ndim != 2:
        raise ValueError("flat fields require an explicit (sparse) laplacian")
    return graph_laplacian(*field_grid.shape)


def objective(
    field_grid,
    pi,
    mu,
    sigma,
    lam,
    laplacian=None,
    n_edges=None,
    *,
    penalty="quadratic",
    delta=0.05,
    edges=None,
):
    """J_lambda(x) = NLL(x)/N + lambda * (x^T L x)/|E_8| (§3.1a / §4.1.3).

    `laplacian` and `n_edges` are accepted so a sweep/optimiser can build them
    once; both default to the `E_8` grid for the field's shape. With
    `penalty="huber"` the smoothness term becomes the Huber-TV edge mean of
    `huber_penalty_edge_mean` (with `edges` defaulted like `laplacian`), giving
    `J_huber <= J_TV <= J_huber + lambda*delta/2`. The default
    `penalty="quadratic"` is the literal pre-ablation code path.
    """

    field_grid = np.asarray(field_grid, dtype=float)
    if penalty == "huber":
        if edges is None:
            edges = _default_grid_edges(field_grid)
        nll_over_n = gmm_nll_over_n(field_grid, pi, mu, sigma)
        return nll_over_n + lam * huber_penalty_edge_mean(field_grid, edges, delta)
    if penalty != "quadratic":
        raise ValueError(f"unknown penalty {penalty!r}; expected 'quadratic' or 'huber'")
    if laplacian is None:
        laplacian = _default_grid_laplacian(field_grid)
    if n_edges is None:
        n_edges = len(_default_grid_edges(field_grid))

    nll_over_n = gmm_nll_over_n(field_grid, pi, mu, sigma)
    smooth = roughness_sum(field_grid, laplacian) / n_edges
    return nll_over_n + lam * smooth


def objective_gradient(
    field_grid,
    pi,
    mu,
    sigma,
    lam,
    laplacian=None,
    n_edges=None,
    *,
    penalty="quadratic",
    delta=0.05,
    edges=None,
):
    """Gradient of `objective` w.r.t. the field, shape `[H, W]`.

    grad J_lambda = (1/N) * grad NLL + (lambda/|E_8|) * 2 L x. The NLL part is the
    closed form in `nll_gradient`; the smoothness part is the edge-mean-normalised
    `2 L x` so it matches the `(x^T L x)/|E_8|` term in `objective`. With
    `penalty="huber"` the smoothness part is `huber_penalty_gradient` instead.
    """

    field_grid = np.asarray(field_grid, dtype=float)
    n = field_grid.size
    if penalty == "huber":
        if edges is None:
            edges = _default_grid_edges(field_grid)
        grad_nll = nll_gradient(field_grid, pi, mu, sigma) / n
        return grad_nll + lam * huber_penalty_gradient(field_grid, edges, delta)
    if penalty != "quadratic":
        raise ValueError(f"unknown penalty {penalty!r}; expected 'quadratic' or 'huber'")
    if laplacian is None:
        laplacian = _default_grid_laplacian(field_grid)
    if n_edges is None:
        n_edges = len(_default_grid_edges(field_grid))

    grad_nll = nll_gradient(field_grid, pi, mu, sigma) / n
    smooth_grad_flat = 2.0 * (laplacian @ field_grid.reshape(-1)) / n_edges
    grad_smooth = smooth_grad_flat.reshape(field_grid.shape)
    return grad_nll + lam * grad_smooth


@dataclass
class OptimResult:
    """Outcome of one warm-started, multi-restart minimisation at a fixed lambda."""

    field: np.ndarray  # lowest-energy field found, [H, W] or flat [N]
    energy: float  # J_lambda of `field`
    nll_over_n: float  # NLL/N of `field`
    r_tilde: float  # scale-free roughness of `field`
    variance_collapsed: bool
    spectral_hf_ratio: float  # secondary high-frequency power diagnostic; None for flat fields
    spectral_slope: float  # log-log radial-spectrum slope (shape check); None for flat fields
    spectral_monotone_fraction: float  # fraction of decreasing adjacent bins; None for flat fields
    spectral_collapsed: bool  # None for flat fields
    restart_energies: np.ndarray  # J_lambda reached by each restart
    restart_spread: float  # max - min over restart energies (energy units)
    restart_field_spread: float  # max over cells of (max - min) restart field value
    n_restarts: int
    restart_nll_over_n: np.ndarray = None  # per-restart NLL/N (robustness row)


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
    penalty="quadratic",
    delta=0.05,
):
    """Minimise J_lambda with an Adam optimiser, warm-started from `mode_field`.

    Restart 0 is the bare mode-field warm start (§4.1.4); the remaining restarts
    add zero-mean Gaussian jitter (std `restart_scale`) to the mode field so the
    non-convex landscape is probed from a few nearby basins. The lowest-energy
    field is kept; `restart_spread` (energy range) and `restart_field_spread`
    (worst per-cell value range across restarts) are exposed as stability outputs.
    `penalty`/`delta` select the smoothness term (see `objective`); the default
    `"quadratic"` is the literal pre-ablation code path.

    Toy `[H, W, K]` parameters keep the `E_8` defaults (bit-identical path);
    flat `[N, K]` parameters require explicit `edges` and a sparse `laplacian`,
    and the spectral fields of the result are `None` (planar-FFT diagnostics
    are gated to 2D lattices). `restart_scale` stays 1.0 for the toy; use
    ~median sigma = 0.15 on the real data (the toy value would jump ~7 sigma).
    """

    pi = np.asarray(pi, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    if mu.ndim == 3:
        height, width, _ = mu.shape
        if laplacian is None:
            laplacian = graph_laplacian(height, width)
        if edges is None:
            edges = grid_edges_8(height, width)
    elif mu.ndim == 2:
        if laplacian is None or edges is None:
            raise ValueError(
                "flat [N, K] inputs require explicit edges and a (sparse) laplacian"
            )
    else:
        raise ValueError("mu must have shape [H, W, K] or [N, K]")
    if n_edges is None:
        n_edges = len(edges)
    rng = np.random.default_rng() if rng is None else rng

    warm, _ = mode_field(pi, mu, sigma)

    def grad_fn(x):
        return objective_gradient(
            x,
            pi,
            mu,
            sigma,
            lam,
            laplacian=laplacian,
            n_edges=n_edges,
            penalty=penalty,
            delta=delta,
            edges=edges,
        )

    fields = []
    energies = []
    for r in range(n_restarts):
        x0 = warm.copy() if r == 0 else warm + rng.normal(scale=restart_scale, size=warm.shape)
        x_opt = _adam_descent(x0, grad_fn, n_steps=n_steps, lr=lr)
        fields.append(x_opt)
        energies.append(
            objective(
                x_opt,
                pi,
                mu,
                sigma,
                lam,
                laplacian=laplacian,
                n_edges=n_edges,
                penalty=penalty,
                delta=delta,
                edges=edges,
            )
        )

    fields = np.stack(fields)  # [R, H, W] or [R, N]
    energies = np.asarray(energies)
    best = int(np.argmin(energies))
    best_field = fields[best]

    r_tilde, collapsed = scale_free_roughness(best_field, edges)
    if best_field.ndim == 2:
        spectral = spectral_roughness(best_field)
    else:
        spectral = {
            "spectral_hf_ratio": None,
            "spectral_slope": None,
            "spectral_monotone_fraction": None,
            "spectral_collapsed": None,
        }
    # Worst-case disagreement between restart fields, per cell, then max over cells.
    field_spread = float(np.max(fields.max(axis=0) - fields.min(axis=0)))
    restart_nll = np.asarray([gmm_nll_over_n(f, pi, mu, sigma) for f in fields])

    return OptimResult(
        field=best_field,
        energy=float(energies[best]),
        nll_over_n=gmm_nll_over_n(best_field, pi, mu, sigma),
        r_tilde=r_tilde,
        variance_collapsed=collapsed,
        spectral_hf_ratio=spectral["spectral_hf_ratio"],
        spectral_slope=spectral["spectral_slope"],
        spectral_monotone_fraction=spectral["spectral_monotone_fraction"],
        spectral_collapsed=spectral["spectral_collapsed"],
        restart_energies=energies,
        restart_spread=float(energies.max() - energies.min()),
        restart_field_spread=field_spread,
        n_restarts=n_restarts,
        restart_nll_over_n=restart_nll,
    )


@dataclass
class LambdaSweepPoint:
    """One row of the lambda-sweep Pareto curve."""

    lam: float
    nll_over_n: float
    r_tilde: float
    variance_collapsed: bool
    spectral_hf_ratio: float
    spectral_slope: float
    spectral_monotone_fraction: float
    spectral_collapsed: bool
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
    penalty="quadratic",
    delta=0.05,
    edges=None,
    laplacian=None,
):
    """Produce the Pareto curve (NLL/N vs scale-free R̃) over a lambda grid.

    Each lambda is solved by `minimise_at_lambda` from its own freshly-seeded RNG
    (seed offset by the lambda index) so the sweep is reproducible and order
    independent. Returns a list of `LambdaSweepPoint`, one per lambda.
    `penalty`/`delta` select the smoothness term (see `objective`); each point's
    `energy` is then J under that penalty. Toy `[H, W, K]` inputs keep the `E_8`
    defaults; flat `[N, K]` inputs require explicit `edges` + sparse `laplacian`.
    """

    pi = np.asarray(pi, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    if mu.ndim == 3:
        height, width, _ = mu.shape
        if laplacian is None:
            laplacian = graph_laplacian(height, width)
        if edges is None:
            edges = grid_edges_8(height, width)
    elif edges is None or laplacian is None:
        raise ValueError("flat [N, K] inputs require explicit edges and a (sparse) laplacian")
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
            penalty=penalty,
            delta=delta,
        )
        points.append(
            LambdaSweepPoint(
                lam=float(lam),
                nll_over_n=res.nll_over_n,
                r_tilde=res.r_tilde,
                variance_collapsed=res.variance_collapsed,
                spectral_hf_ratio=res.spectral_hf_ratio,
                spectral_slope=res.spectral_slope,
                spectral_monotone_fraction=res.spectral_monotone_fraction,
                spectral_collapsed=res.spectral_collapsed,
                restart_spread=res.restart_spread,
                restart_field_spread=res.restart_field_spread,
                energy=res.energy,
                field=res.field,
            )
        )
    return points
