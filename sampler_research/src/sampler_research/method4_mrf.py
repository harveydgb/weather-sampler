"""Method 4 — mode extraction + value-space MRF (Phase 2 Stage C).

Implements the explicitly mode-conditioned single-field route of
phase_2_research_plan.md §4.2: extract each location's GMM density peaks
(modes), then pick one spatially coherent mode per cell by minimising a unary
likelihood + value-space pairwise smoothness energy over the (<=K) modes. The
pairwise term couples on mode *values*, never on the label index (C5), and the
unary keeps each chosen mode high-likelihood (without it the MRF could pick a
value-compatible but low-density compromise — §4.2 step 3).

The objective is the discrete value-space analogue of Method 1's `J_lambda`,
in the same per-cell / per-edge units (§3.1a):

    J_beta(a) = (1/N) sum_i u_i(a_i)
              + beta * (1/|E_8|) sum_{(i,j) in E_8} (m_{i,a_i} - m_{j,a_j})^2

with `u_i(m) = -log p_i(m)` the per-cell unary and `beta` the discrete
smoothness/coupling knob swept in place of `lambda`. Because the field is
piecewise-constant in `beta` (it can only ever equal one of the finitely many
extracted modes), the `beta`-sweep is denser than Method 1's and reports
repeated identical fields rather than a smooth Pareto front.

This module reuses the Stage A / Stage B machinery rather than reinventing it:
the per-cell GMM density (`mixture_pdf` / `normal_pdf` / `gmm_nll_*`), the `E_8`
graph + roughness metrics (`sampler_research.graph`), the secondary spectral
diagnostics (`sampler_research.spectral`), and — crucially — the shared
value-space ICM core `_value_space_icm` that also drives the `a*` baseline. The
warm-start *policy* differs per caller (a* = highest-pi; Method 4 restart 0 =
unary-best), so it is passed in rather than hard-coded.

The mean-shift mode finder iterates the *heteroscedastic* stationary point

    v <- (sum_k r_k(v) mu_{ik} / sigma_{ik}^2) / (sum_k r_k(v) / sigma_{ik}^2)

which is the exact zero of `p'(v)` for per-component sigma (phase_4_plan §3).
When sigma is component-shared within a cell it reduces algebraically to the
shared-sigma fixed point `v <- sum_k r_k(v) mu_{ik}` (Carreira-Perpinan 2000)
used previously, so both Phase 1 toys reproduce bit-near. For per-component
sigma the equal-covariance at-most-K mode bound (Carreira-Perpinan & Williams
2003) no longer strictly applies — >K modes are a theoretical 1D possibility —
but K per-mean starts with `Kmax = K` padding are kept: the diagnostic needs
good candidates, not exhaustive enumeration.
"""

from dataclasses import dataclass
from dataclasses import field as _field

import numpy as np

from sampler_research.baselines import _value_space_icm, _argmin_by_value, gmm_nll_over_n
from sampler_research.gmm import gmm_log_pdf, mixture_pdf, normal_pdf
from sampler_research.graph import grid_edges_8, roughness_edge_mean, scale_free_roughness
from sampler_research.spectral import spectral_roughness

# Working drift thresholds for the non-smearing diagnostic, in NLL nats.
# DERIVED (v1) from the sigma-unit epsilon reasoning at large_notes.md:260:
# near a mode `ΔNLL ≈ ½ z²`, so a `0.5σ` drift ⇒ ε ≈ 0.125 and a `1σ` drift ⇒
# ε ≈ 0.5. There they are *illustrative examples* for choosing ε, NOT locked
# Phase-3 thresholds (they are not in phase_3.md). Kept parameterised and
# labelled "v1" until/unless a phase spec promotes them.
DRIFT_THRESHOLDS_V1 = (0.125, 0.5)


@dataclass
class ModeExtraction:
    """Padded ragged GMM modes for one toy field (the reusable Stage C artifact).

    `mode_values[i, :mode_counts[i]]` are the sorted extracted modes of cell `i`;
    the trailing padded slots repeat the first (smallest) mode as a **finite**
    sentinel so they never inject `nan` into the pairwise arithmetic. `valid_mask`
    marks the real slots and `mode_unary` pads invalid slots with `+inf` so the
    ICM never selects them.
    """

    mode_values: np.ndarray  # [N, Kmax], padded with a finite sentinel
    valid_mask: np.ndarray  # [N, Kmax] bool
    mode_unary: np.ndarray  # [N, Kmax], -log p_i(mode); +inf on padded slots
    mode_counts: np.ndarray  # [N] int, number of real modes per cell
    height: int = None  # toy 2D grid shape; None for flat [N, K] extractions
    width: int = None

    @property
    def field_shape(self):
        """Shape a per-cell field should take: `(H, W)` for the toy, `(N,)` flat."""

        if self.height is None:
            return (self.mode_values.shape[0],)
        return (self.height, self.width)


def extract_gmm_modes(
    pi,
    mu,
    sigma,
    *,
    max_iter=200,
    tol=1e-10,
    merge_tol=1e-4,
    density_floor=1e-300,
):
    """Extract each cell's GMM modes by 1D mean-shift; return padded candidates.

    For every cell the heteroscedastic fixed point
    `v <- (sum_k r_k(v) mu_k / sigma_k^2) / (sum_k r_k(v) / sigma_k^2)`
    (responsibilities `r_k(v) = pi_k N(v; mu_k, sigma_k^2) / p(v)`) is iterated
    from each component mean to convergence, duplicate fixed points are merged
    within `merge_tol`, survivors are sorted by value and scored with the unary
    `-log p_i(mode)` (floored by `density_floor` for finite logs). Returns a
    `ModeExtraction` whose ragged per-cell mode sets are padded to `Kmax = K`
    (see the module docstring for the >K caveat): values padded with a finite
    sentinel, unary padded with `+inf`. Accepts `[H, W, K]` (toy) or flat
    `[N, K]` parameters; the latter leaves `height`/`width` as `None`.
    """

    pi = np.asarray(pi, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if mu.ndim == 3:
        height, width, k = mu.shape
        n = height * width
    elif mu.ndim == 2:
        height = width = None
        n, k = mu.shape
    else:
        raise ValueError("mu must have shape [H, W, K] or [N, K]")

    pi_f = pi.reshape(n, k)
    mu_f = mu.reshape(n, k)
    sigma_f = sigma.reshape(n, k)
    inv_var = 1.0 / sigma_f**2  # [N, K]

    # Vectorised 1D Gaussian mean-shift, one trajectory per component mean.
    # v has shape [N, S] with S = K starts; responsibilities are [N, S, K].
    v = mu_f.copy()
    for _ in range(max_iter):
        comp = pi_f[:, None, :] * normal_pdf(
            v[:, :, None], mu_f[:, None, :], sigma_f[:, None, :]
        )  # [N, S, K]
        density = np.sum(comp, axis=-1)  # [N, S]; > 0 (point lies within the mu range)
        resp = comp / density[..., None]  # [N, S, K]
        # Heteroscedastic stationary point: precision-weighted responsibility mean.
        numer = np.sum(resp * (mu_f * inv_var)[:, None, :], axis=-1)  # [N, S]
        denom = np.sum(resp * inv_var[:, None, :], axis=-1)  # [N, S]
        v_new = numer / denom
        delta = np.max(np.abs(v_new - v))
        v = v_new
        if delta < tol:
            break

    Kmax = k
    mode_values = np.empty((n, Kmax), dtype=float)
    valid_mask = np.zeros((n, Kmax), dtype=bool)
    mode_unary = np.full((n, Kmax), np.inf, dtype=float)
    mode_counts = np.empty(n, dtype=np.int64)

    for i in range(n):
        fixed_points = np.sort(v[i])
        # Merge duplicate fixed points (same mode reached from several starts).
        modes = [fixed_points[0]]
        for val in fixed_points[1:]:
            if val - modes[-1] > merge_tol:
                modes.append(val)
        modes = np.asarray(modes, dtype=float)
        count = modes.size
        mode_counts[i] = count

        density = mixture_pdf(modes, pi_f[i], mu_f[i], sigma_f[i])
        unary = -np.log(np.maximum(density, density_floor))

        mode_values[i, :count] = modes
        mode_values[i, count:] = modes[0]  # finite sentinel, never selected
        valid_mask[i, :count] = True
        mode_unary[i, :count] = unary

    return ModeExtraction(
        mode_values=mode_values,
        valid_mask=valid_mask,
        mode_unary=mode_unary,
        mode_counts=mode_counts,
        height=height,
        width=width,
    )


def value_mrf_energy(assignment, extraction, edges, beta):
    """Normalised value-space MRF energy `J_beta` of a mode assignment.

    `J_beta(a) = mean_i u_i(a_i) + beta * mean_edges (m_{i,a_i} - m_{j,a_j})^2`,
    in the same per-cell / per-edge units as Method 1's `J_lambda` (§3.1a).
    `assignment` is a flat `[N]` or `[H, W]` index into the extracted modes.
    """

    n = extraction.mode_values.shape[0]
    a = np.asarray(assignment, dtype=np.int64).reshape(n)
    rows = np.arange(n)
    field = extraction.mode_values[rows, a]  # flat; roughness_edge_mean flattens anyway
    unary_mean = float(np.mean(extraction.mode_unary[rows, a]))
    pairwise_mean = roughness_edge_mean(field, edges)
    return unary_mean + beta * pairwise_mean


@dataclass
class ValueMRFResult:
    """Outcome of one multi-restart value-space MRF solve at a fixed beta."""

    field: np.ndarray  # lowest-energy field found, [H, W] or flat [N]
    assignment: np.ndarray  # chosen mode index per cell, [H, W] or flat [N]
    energy: float  # J_beta of `field`
    nll_over_n: float  # NLL/N of `field`
    r_tilde: float  # scale-free roughness of `field`
    variance_collapsed: bool
    spectral_hf_ratio: float  # None for flat fields
    spectral_slope: float  # None for flat fields
    spectral_monotone_fraction: float  # None for flat fields
    spectral_collapsed: bool  # None for flat fields
    restart_energies: np.ndarray  # J_beta reached by each restart
    restart_spread: float  # max - min over restart energies (energy units)
    restart_field_spread: float  # max over cells of (max - min) restart field value
    n_restarts: int
    restart_n_sweeps: np.ndarray = None  # ICM sweeps executed per restart


def _unary_best_assignment(extraction):
    """Per-cell lowest-unary (highest-density) mode, ties broken by value (C5)."""

    n = extraction.mode_values.shape[0]
    return np.array(
        [
            _argmin_by_value(extraction.mode_unary[i], extraction.mode_values[i])
            for i in range(n)
        ],
        dtype=np.int64,
    )


def solve_value_mrf(
    pi,
    mu,
    sigma,
    extraction,
    beta,
    *,
    n_restarts=4,
    edges=None,
    n_edges=None,
    rng=None,
    max_sweeps=50,
):
    """Minimise `J_beta` over mode assignments by warm-started multi-restart ICM.

    Restart 0 is the unary-best (highest-density) mode field; restarts 1..R are
    seeded uniform-random *valid* mode assignments. Each restart runs the shared
    `_value_space_icm` core with `pairwise_weight = beta * N / |E_8|` (so the
    integer-weighted sweep minimises the normalised `J_beta`). The lowest-energy
    field is kept; restart energy spread and worst per-cell field spread are
    exposed as stability outputs, mirroring Method 1's `OptimResult`.
    """

    pi = np.asarray(pi, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = extraction.mode_values.shape[0]

    if edges is None:
        if extraction.height is None:
            raise ValueError("flat extractions require explicit edges")
        edges = grid_edges_8(extraction.height, extraction.width)
    edges = np.asarray(edges, dtype=np.int64)
    if n_edges is None:
        n_edges = len(edges)
    rng = rng or np.random.default_rng()

    pairwise_weight = beta * n / n_edges
    mode_values = extraction.mode_values
    valid_mask = extraction.valid_mask
    mode_unary = extraction.mode_unary
    mode_counts = extraction.mode_counts
    rows = np.arange(n)

    warm_best = _unary_best_assignment(extraction)

    field_shape = extraction.field_shape

    fields = []
    assignments = []
    energies = []
    sweep_counts = []
    for r in range(n_restarts):
        if r == 0:
            warm = warm_best
        else:
            warm = np.array([rng.integers(0, mode_counts[i]) for i in range(n)], dtype=np.int64)
        a, n_sweeps = _value_space_icm(
            mode_values,
            valid_mask,
            mode_unary,
            edges,
            warm,
            pairwise_weight=pairwise_weight,
            max_sweeps=max_sweeps,
            return_n_sweeps=True,
        )
        assignments.append(a)
        fields.append(mode_values[rows, a].reshape(field_shape))
        energies.append(value_mrf_energy(a, extraction, edges, beta))
        sweep_counts.append(n_sweeps)

    fields = np.stack(fields)  # [R, H, W] or [R, N]
    energies = np.asarray(energies)
    best = int(np.argmin(energies))
    best_field = fields[best]
    best_assignment = assignments[best].reshape(field_shape)

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
    field_spread = float(np.max(fields.max(axis=0) - fields.min(axis=0)))

    return ValueMRFResult(
        field=best_field,
        assignment=best_assignment,
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
        restart_n_sweeps=np.asarray(sweep_counts, dtype=np.int64),
    )


def delta_nll_to_best_mode(
    field,
    pi,
    mu,
    sigma,
    mode_values,
    valid_mask,
    *,
    thresholds=DRIFT_THRESHOLDS_V1,
):
    """Per-cell drift of a field off its best extracted mode (non-smearing probe).

    `ΔNLL_i = -log p_i(x_i) - min_{valid k} (-log p_i(mode_{ik}))`: how much
    likelihood the realised value `x_i` gives up relative to the highest-density
    extracted mode (the global peak of `p_i`, so `ΔNLL_i >= 0`). It is the
    property the bare `(NLL/N, R̃)` plane cannot show: Method 1 reaches low `R̃`
    by moving values continuously into the low-density valleys *between* modes,
    whereas a mode-restricted field stays on the peaks. This is a standalone,
    **cross-method** helper — it scores any field against any persisted modes, so
    Method 1, `smoothed_map`, and the Method 5 ablation can all be read against
    Stage C's reference modes.

    Shape-agnostic (`[H, W]` or flat `[N]` fields); densities are routed through
    the log-space `gmm_log_pdf`, so far-off-mode values cannot underflow.
    Returns mean / p95 / max over cells plus the fraction of cells exceeding each
    drift threshold; `per_cell` keeps the field's shape. The default thresholds
    (0.125, 0.5) are **v1 working** values, not locked Phase-3 thresholds — see
    `DRIFT_THRESHOLDS_V1`.
    """

    field = np.asarray(field, dtype=float)
    pi = np.asarray(pi, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    k = mu.shape[-1]
    n = mu.size // k

    mode_values = np.asarray(mode_values, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    value_nll = -gmm_log_pdf(field, pi, mu, sigma).reshape(n)

    pi_f = pi.reshape(n, 1, k)
    mu_f = mu.reshape(n, 1, k)
    sigma_f = sigma.reshape(n, 1, k)
    mode_nll = -gmm_log_pdf(mode_values, pi_f, mu_f, sigma_f)  # [N, Kmax]
    mode_nll = np.where(valid_mask, mode_nll, np.inf)
    best_mode_nll = np.min(mode_nll, axis=1)  # [N]

    delta = value_nll - best_mode_nll
    frac_over = tuple(float(np.mean(delta > t)) for t in thresholds)
    return {
        "mean": float(np.mean(delta)),
        "p95": float(np.percentile(delta, 95)),
        "max": float(np.max(delta)),
        "thresholds": tuple(float(t) for t in thresholds),
        "frac_over": frac_over,
        "per_cell": delta.reshape(field.shape),
    }


@dataclass
class BetaSweepPoint:
    """One row of the beta-sweep curve (mirrors Method 1's `LambdaSweepPoint`)."""

    beta: float
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
    delta_to_mode: dict = _field(repr=False)
    field: np.ndarray = _field(repr=False)
    assignment: np.ndarray = _field(repr=False)
    restart_n_sweeps: np.ndarray = None  # ICM sweeps executed per restart


def beta_sweep(
    pi,
    mu,
    sigma,
    betas,
    *,
    n_restarts=4,
    seed=0,
    extraction=None,
    max_sweeps=50,
    thresholds=DRIFT_THRESHOLDS_V1,
    edges=None,
):
    """Produce the `NLL/N` vs `R̃` curve over a beta grid (mode artifacts reused).

    The modes are extracted once (`extraction`, reused across the whole sweep —
    every beta selects among the *same* modes) and each beta is solved by
    `solve_value_mrf` from its own freshly-seeded RNG (`seed + idx`, mirroring
    `lambda_sweep`) so the sweep is reproducible and order independent. Each
    point also carries its `delta_nll_to_best_mode` summary. Returns a list of
    `BetaSweepPoint`. Use a denser grid than Method 1: the field is
    piecewise-constant in beta, so neighbouring betas can repeat the same field.
    Toy `[H, W, K]` inputs keep the `E_8` default; flat `[N, K]` inputs require
    explicit `edges`.
    """

    pi = np.asarray(pi, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    if edges is None:
        if mu.ndim != 3:
            raise ValueError("flat [N, K] inputs require explicit edges")
        height, width, _ = mu.shape
        edges = grid_edges_8(height, width)
    n_edges = len(edges)
    if extraction is None:
        extraction = extract_gmm_modes(pi, mu, sigma)

    points = []
    for idx, beta in enumerate(betas):
        rng = np.random.default_rng(seed + idx)
        res = solve_value_mrf(
            pi,
            mu,
            sigma,
            extraction,
            beta,
            n_restarts=n_restarts,
            edges=edges,
            n_edges=n_edges,
            rng=rng,
            max_sweeps=max_sweeps,
        )
        delta = delta_nll_to_best_mode(
            res.field,
            pi,
            mu,
            sigma,
            extraction.mode_values,
            extraction.valid_mask,
            thresholds=thresholds,
        )
        points.append(
            BetaSweepPoint(
                beta=float(beta),
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
                delta_to_mode=delta,
                field=res.field,
                assignment=res.assignment,
                restart_n_sweeps=res.restart_n_sweeps,
            )
        )
    return points
