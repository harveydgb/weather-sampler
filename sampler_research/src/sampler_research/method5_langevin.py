"""Method 5 — annealed-noise Langevin **stochastic search** (Phase 2 Stage D).

This is *not* a third sampler family. Method 5 in phase_2_research_plan.md §4.3 is
tagged INFRASTRUCTURE · DEFERRED→ensemble: it is the engine for Method 1, not a
standalone entrant. It runs noisy gradient steps on **Method 1's exact energy**

    J_lambda(x) = NLL(x)/N + lambda * (x^T L x)/|E_8|

with **Method 1's exact gradient** (`objective` / `objective_gradient` from
`regularised_map`), then polishes each chain's lowest-energy snapshot to a local
minimum with the same Adam descent Method 1 uses and keeps the lowest-energy
polished field. A stochastic search that returns the argmin of `J_lambda` is an
*optimiser of `J_lambda`* — i.e. Method 1 with a different search procedure.

So this module is built as a scoped **robustness ablation** answering one honest
question: does stochastic global search find materially lower-`J_lambda` basins
than Method 1's mode-warm-start + Adam restarts? The reporting (in the Stage D
script) is an *energy-delta* table against the persisted Stage B fields, not a
second independent Pareto curve — Method 1 and Method 5 minimise the *same*
objective, so near-overlap on the `(NLL/N, R̃)` plane is tautology, not
corroboration.

**Honesty note on the "Langevin" name.** The update

    x_next = x - 0.5*step*grad_J(x) + sqrt(step)*noise_t*eta

decouples a *fixed* `step` from a *separately annealed* `noise_t` (geometric from
`noise_start` to `noise_end`). This is **not** principled annealed Langevin —
Song & Ermon (2019) tie the step to sigma^2, and this chain samples no
well-defined distribution. That is acceptable *because it is used as a stochastic
optimiser, not a sampler*; the chain only ever feeds its lowest-`J_lambda`
snapshot into a deterministic polish. Do not read distributional meaning into it.
Langevin-as-*sampler* stays deferred to the ensemble extension (§4.3 field 10).

Reuses, rather than re-deriving, Method 1's machinery: `objective`,
`objective_gradient`, `_adam_descent`, the warm start `mode_field`, the `E_8`
graph + roughness, and `spectral_roughness` for scoring.
"""

from dataclasses import dataclass, field

import numpy as np

from sampler_research.baselines import gmm_nll_over_n, mode_field
from sampler_research.graph import (
    graph_laplacian,
    grid_edges_8,
    scale_free_roughness,
)
from sampler_research.regularised_map import _adam_descent, objective, objective_gradient
from sampler_research.spectral import spectral_roughness


def _noise_schedule(noise_start, noise_end, n_steps):
    """Geometric anneal `noise_start -> noise_end` over `n_steps`.

    Falls back to a linear schedule when either endpoint is non-positive (e.g. the
    zero-noise `noise_start = noise_end = 0` reduction to deterministic descent),
    where a geometric ratio is undefined.
    """

    if n_steps <= 1:
        return np.full(max(n_steps, 0), float(noise_end))
    if noise_start > 0.0 and noise_end > 0.0:
        ratio = noise_end / noise_start
        return noise_start * ratio ** (np.arange(n_steps) / (n_steps - 1))
    return np.linspace(noise_start, noise_end, n_steps)


def langevin_chain(
    x0,
    grad_fn,
    energy_fn,
    *,
    n_steps,
    step_size,
    noise_start,
    noise_end,
    rng,
):
    """Run one annealed-noise Langevin **search** chain; track its best snapshot.

    Iterates `x <- x - 0.5*step*grad_J(x) + sqrt(step)*noise_t*eta` with `noise_t`
    on the geometric `_noise_schedule`, and keeps the lowest-`J_lambda` field seen
    (including the warm start `x0` at step 0). The chain is a stochastic optimiser,
    not a sampler (see the module docstring); only its argmin snapshot is used.

    A fixed-step chain on the stiff large-lambda Laplacian term can diverge
    (§ plan stability caution). If an iterate goes non-finite the chain stops early
    and returns the best finite snapshot so far, so the diagnostics stay finite.

    Returns `(best_field, best_energy, best_step)`.
    """

    x = np.asarray(x0, dtype=float).copy()
    noise_levels = _noise_schedule(noise_start, noise_end, n_steps)
    sqrt_step = np.sqrt(step_size)

    best_field = x.copy()
    best_energy = float(energy_fn(x))
    best_step = 0
    for t in range(n_steps):
        grad = grad_fn(x)
        eta = rng.standard_normal(x.shape)
        x = x - 0.5 * step_size * grad + sqrt_step * noise_levels[t] * eta
        if not np.all(np.isfinite(x)):
            break  # diverged: keep the best finite snapshot seen so far
        energy = float(energy_fn(x))
        if energy < best_energy:
            best_energy = energy
            best_field = x.copy()
            best_step = t + 1
    return best_field, best_energy, best_step


@dataclass
class LangevinResult:
    """Outcome of one multi-chain Langevin search + deterministic polish at a lambda."""

    field: np.ndarray  # lowest-energy polished field found, [H, W]
    energy: float  # J_lambda of `field` (the selected field)
    nll_over_n: float  # NLL/N of `field`
    r_tilde: float  # scale-free roughness of `field`
    variance_collapsed: bool
    spectral_hf_ratio: float
    spectral_slope: float
    spectral_monotone_fraction: float
    spectral_collapsed: bool
    chain_energies: np.ndarray  # polished J_lambda reached by each chain
    chain_energy_spread: float  # max - min over polished chain energies (J units)
    chain_field_spread: float  # max over cells of (max - min) polished chain value
    best_chain: int  # index of the selected (lowest-energy) chain
    best_chain_step: int  # step at which the selected chain's best snapshot occurred
    best_snapshot_energy: float  # unpolished J_lambda of the selected chain's snapshot
    n_chains: int


def langevin_at_lambda(
    pi,
    mu,
    sigma,
    lam,
    *,
    n_chains=8,
    n_steps=2500,
    step_size=0.05,
    noise_start=0.5,
    noise_end=0.02,
    polish_steps=1000,
    polish_lr=0.05,
    jitter_scale=1.0,
    rng=None,
    laplacian=None,
    n_edges=None,
    edges=None,
):
    """Minimise `J_lambda` by multi-chain Langevin search + deterministic polish.

    Chain 0 is the bare `mode_field` warm start (the lambda -> 0 anchor, mirroring
    Method 1's restart 0); chains 1..`n_chains`-1 add zero-mean Gaussian jitter
    (std `jitter_scale`) to it. Each chain tracks its lowest-`J_lambda` snapshot,
    which is then **deterministically polished** with `_adam_descent` (the same
    optimiser Method 1 uses), and the lowest-energy polished field is kept as the
    single selected field for this lambda.

    Chain-energy spread and worst per-cell field spread are exposed as stability
    outputs, mirroring Method 1's `OptimResult`.
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

    def energy_fn(x):
        return objective(x, pi, mu, sigma, lam, laplacian=laplacian, n_edges=n_edges)

    polished_fields = []
    polished_energies = []
    snapshot_energies = []
    best_steps = []
    for c in range(n_chains):
        x0 = warm.copy() if c == 0 else warm + rng.normal(scale=jitter_scale, size=warm.shape)
        snapshot, snapshot_energy, best_step = langevin_chain(
            x0,
            grad_fn,
            energy_fn,
            n_steps=n_steps,
            step_size=step_size,
            noise_start=noise_start,
            noise_end=noise_end,
            rng=rng,
        )
        polished = _adam_descent(snapshot, grad_fn, n_steps=polish_steps, lr=polish_lr)
        polished_fields.append(polished)
        polished_energies.append(float(energy_fn(polished)))
        snapshot_energies.append(snapshot_energy)
        best_steps.append(best_step)

    polished_fields = np.stack(polished_fields)  # [C, H, W]
    polished_energies = np.asarray(polished_energies)
    best = int(np.argmin(polished_energies))
    best_field = polished_fields[best]

    r_tilde, collapsed = scale_free_roughness(best_field, edges)
    spectral = spectral_roughness(best_field)
    field_spread = float(np.max(polished_fields.max(axis=0) - polished_fields.min(axis=0)))

    return LangevinResult(
        field=best_field,
        energy=float(polished_energies[best]),
        nll_over_n=gmm_nll_over_n(best_field, pi, mu, sigma),
        r_tilde=r_tilde,
        variance_collapsed=collapsed,
        spectral_hf_ratio=spectral["spectral_hf_ratio"],
        spectral_slope=spectral["spectral_slope"],
        spectral_monotone_fraction=spectral["spectral_monotone_fraction"],
        spectral_collapsed=spectral["spectral_collapsed"],
        chain_energies=polished_energies,
        chain_energy_spread=float(polished_energies.max() - polished_energies.min()),
        chain_field_spread=field_spread,
        best_chain=best,
        best_chain_step=int(best_steps[best]),
        best_snapshot_energy=float(snapshot_energies[best]),
        n_chains=n_chains,
    )


@dataclass
class LangevinSweepPoint:
    """One row of the Stage D Langevin-search sweep (mirrors `LambdaSweepPoint`)."""

    lam: float
    nll_over_n: float
    r_tilde: float
    variance_collapsed: bool
    spectral_hf_ratio: float
    spectral_slope: float
    spectral_monotone_fraction: float
    spectral_collapsed: bool
    chain_energy_spread: float
    chain_field_spread: float
    best_chain: int
    best_chain_step: int
    best_snapshot_energy: float
    energy: float
    field: np.ndarray = field(repr=False)


def langevin_sweep(
    pi,
    mu,
    sigma,
    lambdas,
    *,
    n_chains=8,
    n_steps=2500,
    step_size=0.05,
    noise_start=0.5,
    noise_end=0.02,
    polish_steps=1000,
    polish_lr=0.05,
    jitter_scale=1.0,
    seed=0,
):
    """Run the Langevin-search ablation over a lambda grid (one field per lambda).

    Each lambda is solved by `langevin_at_lambda` from its own freshly-seeded RNG
    (`default_rng(seed + idx)`, mirroring `lambda_sweep`) so the sweep is
    reproducible and order independent. Returns a list of `LangevinSweepPoint`,
    one selected field per lambda — **not** an ensemble (the deliverable is one
    field per lambda; the chains are search infrastructure only).
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
        res = langevin_at_lambda(
            pi,
            mu,
            sigma,
            lam,
            n_chains=n_chains,
            n_steps=n_steps,
            step_size=step_size,
            noise_start=noise_start,
            noise_end=noise_end,
            polish_steps=polish_steps,
            polish_lr=polish_lr,
            jitter_scale=jitter_scale,
            rng=rng,
            laplacian=laplacian,
            n_edges=n_edges,
            edges=edges,
        )
        points.append(
            LangevinSweepPoint(
                lam=float(lam),
                nll_over_n=res.nll_over_n,
                r_tilde=res.r_tilde,
                variance_collapsed=res.variance_collapsed,
                spectral_hf_ratio=res.spectral_hf_ratio,
                spectral_slope=res.spectral_slope,
                spectral_monotone_fraction=res.spectral_monotone_fraction,
                spectral_collapsed=res.spectral_collapsed,
                chain_energy_spread=res.chain_energy_spread,
                chain_field_spread=res.chain_field_spread,
                best_chain=res.best_chain,
                best_chain_step=res.best_chain_step,
                best_snapshot_energy=res.best_snapshot_energy,
                energy=res.energy,
                field=res.field,
            )
        )
    return points
