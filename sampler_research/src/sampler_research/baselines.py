"""Stage A baseline fields and scores.

These are the yardsticks from phase_2_research_plan.md §7 Stage A: iid /
salt-and-pepper, per-location likelihood-only mode/MAP, mixture mean,
variance-scaled GMM (Method 9), the smoothed-MAP critical baseline, and the
smoothest high-likelihood mode-assignment field `a*`. They are scored with
`NLL/N`, scale-free roughness `R̃`, and secondary power-spectrum diagnostics.

This module deliberately does *not* implement regularised MAP (Method 1) or
mode extraction + MRF (Method 4); `a*` couples on component-mean *values*, never
on the per-cell label index (constraint C5).
"""

import numpy as np

from sampler_research.gmm import gmm_log_pdf, mixture_mean, normal_pdf, sample_iid_gmm
from sampler_research.graph import (
    graph_laplacian,
    grid_edges_8,
    laplacian_blur,
    scale_free_roughness,
)
from sampler_research.spectral import spectral_roughness


def gmm_nll_per_cell(field, pi, mu, sigma):
    """Per-location GMM negative log-likelihood `-log p_i(x_i)`.

    Shape-agnostic: `field` is `[H, W]` or `[N]` with the mixture axis last on
    `pi`/`mu`/`sigma`. Computed in log space (`-gmm_log_pdf`) so smeared fields
    at separated-mode cells cannot underflow; equals the previous linear-space
    value to 1e-10 on sane toy fields (phase_4_plan §1).
    """

    return -gmm_log_pdf(field, pi, mu, sigma)


def gmm_nll_over_n(field, pi, mu, sigma):
    """Mean per-location GMM NLL, `NLL/N` (§3.4)."""

    return float(np.mean(gmm_nll_per_cell(field, pi, mu, sigma)))


def iid_baseline(pi, mu, sigma, rng=None):
    """Independent / salt-and-pepper sampler: one draw per cell (Stage A anchor)."""

    sample, component_index = sample_iid_gmm(pi, mu, sigma, rng)
    return sample, component_index


def mixture_mean_field(pi, mu):
    """Per-location mixture-mean diagnostic field (over-smooth / low-likelihood)."""

    return mixture_mean(pi, mu)


def mode_field(pi, mu, sigma):
    """Per-location likelihood-only mode / MAP field.

    Returns `(field, mode_index)`. For each cell the value is the component mean
    `mu_ik` whose mixture density `pi_ik N(mu_ik; ...)` is highest, i.e. the most
    likely component-centre under that cell's GMM. With shared within-cell sigma
    this is the highest-`pi` component mean; it is the high-faithfulness roughness
    anchor.
    """

    pi = np.asarray(pi, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    component_density = pi * normal_pdf(mu, mu, sigma)
    mode_index = np.argmax(component_density, axis=-1)
    field = np.take_along_axis(mu, mode_index[..., None], axis=-1)[..., 0]
    return field, mode_index


def variance_scaled_baseline(pi, mu, sigma, alpha, rng=None):
    """Variance-scaled GMM samples (Method 9): draw with `sigma -> alpha * sigma`.

    No spatial coupling: still salt-and-pepper, only the within-mode noise shrinks.
    """

    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must satisfy 0 < alpha <= 1")
    sample, component_index = sample_iid_gmm(pi, mu, np.asarray(sigma, dtype=float) * alpha, rng)
    return sample, component_index


def smoothed_map_baseline(pi, mu, sigma, step=0.1, n_iters=10, laplacian=None):
    """Smoothed-MAP critical baseline: per-cell mode field blurred over the graph.

    Takes the likelihood-only mode field and applies a graph-Laplacian blur.
    2D `[H, W, K]` input keeps the toy default (dense `E_8` Laplacian built
    here); flat `[N, K]` input must pass the (sparse) `laplacian` operator.
    This is the baseline a real coupling method must beat on the orthogonal
    label-coherence axis.
    """

    field, _ = mode_field(pi, mu, sigma)
    if laplacian is None:
        if field.ndim != 2:
            raise ValueError("flat [N, K] inputs require an explicit (sparse) laplacian")
        height, width = field.shape
        laplacian = graph_laplacian(height, width)
    return laplacian_blur(field, laplacian, step=step, n_iters=n_iters)


def _argmin_by_value(cost, values):
    """Index minimising `cost`, ties broken by smallest `values` (label-free, C5).

    Shared by the smoothest mode-assignment baseline (`a*`) and the Method 4
    value-space MRF: every selection breaks ties by candidate *value*, never by
    the raw label index, so the result is invariant to per-cell component
    relabeling even when the weights or the local cost are tied.
    """

    best = np.flatnonzero(np.isclose(cost, cost.min()))
    return int(best[np.argmin(values[best])])


def _value_space_icm(
    candidate_values,
    valid_mask,
    unary,
    edges,
    warm_start,
    *,
    pairwise_weight=1.0,
    max_sweeps=50,
    return_n_sweeps=False,
):
    """Row-major ICM over per-cell value candidates (shared `a*` / Method 4 core).

    Minimise, by iterated conditional modes starting from `warm_start`,

        sum_i unary_i(a_i)
          + pairwise_weight * sum_{(i,j) in E} (v_{i,a_i} - v_{j,a_j})^2

    where `v = candidate_values` are per-cell candidate *values* (`[N, Kmax]`,
    ragged cells padded to a finite sentinel) and the pairwise term couples on
    those values, never on the label index (C5). Each sweep sets every cell's
    candidate to the one minimising its local (unary + pairwise-to-current-
    neighbours) cost, ties broken by smallest candidate value. Padded slots are
    excluded via `valid_mask` (their local cost is forced to `+inf`), so they are
    never selected as long as each cell has at least one valid candidate.
    Returns the final assignment `[N]`; with `return_n_sweeps=True` (opt-in so
    existing callers keep the bare return) it returns
    `(assignment, n_sweeps_executed)` instead.

    The default `pairwise_weight=1.0` with a zero `unary` and an all-true
    `valid_mask` reproduces the `a*` baseline bit-for-bit; Method 4 passes
    `pairwise_weight = beta * N / |E|` so the same sweep minimises its normalised
    energy `J_beta` (see `method4_mrf.solve_value_mrf`).
    """

    candidate_values = np.asarray(candidate_values, dtype=float)
    unary = np.asarray(unary, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    edges = np.asarray(edges, dtype=np.int64)
    n = candidate_values.shape[0]

    neighbours = [[] for _ in range(n)]
    for i, j in edges:
        neighbours[i].append(j)
        neighbours[j].append(i)

    assignment = np.asarray(warm_start, dtype=np.int64).copy()
    n_sweeps = 0
    for _ in range(max_sweeps):
        n_sweeps += 1
        changed = False
        for i in range(n):
            if not neighbours[i]:
                continue
            neighbour_vals = np.array(
                [candidate_values[m, assignment[m]] for m in neighbours[i]]
            )
            pairwise = np.sum(
                (candidate_values[i][:, None] - neighbour_vals[None, :]) ** 2, axis=-1
            )
            cost = np.where(valid_mask[i], unary[i] + pairwise_weight * pairwise, np.inf)
            best = _argmin_by_value(cost, candidate_values[i])
            if best != assignment[i]:
                assignment[i] = best
                changed = True
        if not changed:
            break
    if return_n_sweeps:
        return assignment, n_sweeps
    return assignment


def smoothest_mode_assignment(pi, mu, edges=None, use_pi_unary=False, max_sweeps=50):
    """Smoothest high-likelihood mode-assignment field `a*`.

    Choose one local component per cell to minimise the value-space smoothness
    cost `sum_(i,j in E) w_ij (mu_{i,k_i} - mu_{j,k_j})^2`, optionally with a
    `-log pi` unary penalty. Coupling is on component-mean *values*, never on the
    raw label index (C5).

    Solved by iterated conditional modes: warm-start from the highest-`pi`
    component, then repeatedly set each cell's component to the one minimising
    its local cost given the current neighbours. The sweep itself is the shared
    `_value_space_icm` core (also used by Method 4), called here with every
    component a valid candidate, unit pairwise weight, and a zero (or `-log pi`)
    unary. Returns `(field, assignment)`.

    Both selections break ties by component *value* (smallest `mu`), never by the
    raw label index. This keeps `a*` invariant to per-cell component relabeling
    (C5) even when `pi` or the cost is tied, e.g. under uniform-`pi` inputs.
    """

    pi = np.asarray(pi, dtype=float)
    mu = np.asarray(mu, dtype=float)
    height, width, k = mu.shape
    n = height * width

    if edges is None:
        edges = grid_edges_8(height, width)
    edges = np.asarray(edges, dtype=np.int64)

    mu_flat = mu.reshape(n, k)
    unary = -np.log(pi.reshape(n, k)) if use_pi_unary else np.zeros((n, k))
    valid_mask = np.ones((n, k), dtype=bool)

    # Warm-start: highest pi, ties broken by smallest mean value (not label index).
    pi_flat = pi.reshape(n, k)
    warm_start = np.array([_argmin_by_value(-pi_flat[i], mu_flat[i]) for i in range(n)])

    assignment = _value_space_icm(
        mu_flat, valid_mask, unary, edges, warm_start, pairwise_weight=1.0, max_sweeps=max_sweeps
    )
    field = mu_flat[np.arange(n), assignment].reshape(height, width)
    return field, assignment.reshape(height, width)


def score_field(field, pi, mu, sigma, edges=None):
    """Return Stage A scores for a produced field.

    The primary scores are `NLL/N` and scale-free `R̃` (§3.1a). Power-spectrum
    scores are secondary necessary-condition diagnostics for high-frequency
    roughness, not standalone skill metrics. Toy `[H, W, K]` parameters keep
    the 1D-to-2D reshape and `E_8` default; true flat `[N, K]` parameters
    require `edges` and report the planar-FFT spectral keys as NaN placeholders
    (the real-grid coherence diagnostic is the sampled spherical variogram).
    """

    field_grid = np.asarray(field, dtype=float)
    params_2d = np.asarray(pi).ndim == 3
    if params_2d:
        if field_grid.ndim == 1:
            field_grid = field_grid.reshape(np.asarray(pi).shape[:2])
        if edges is None:
            height, width = field_grid.shape
            edges = grid_edges_8(height, width)
        spectral = spectral_roughness(field_grid)
    else:
        if edges is None:
            raise ValueError("flat [N, K] inputs require explicit edges")
        spectral = {
            "spectral_hf_ratio": np.nan,
            "spectral_slope": np.nan,
            "spectral_monotone_fraction": np.nan,
            "spectral_collapsed": False,
        }

    nll = gmm_nll_over_n(field_grid, pi, mu, sigma)
    r_tilde, collapsed = scale_free_roughness(field_grid, edges)
    return {
        "nll_over_n": nll,
        "r_tilde": r_tilde,
        "variance_collapsed": collapsed,
        **spectral,
    }
