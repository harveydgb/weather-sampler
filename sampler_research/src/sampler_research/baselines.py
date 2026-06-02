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

from sampler_research.gmm import mixture_mean, mixture_pdf, normal_pdf, sample_iid_gmm
from sampler_research.graph import (
    graph_laplacian,
    grid_edges_8,
    laplacian_blur,
    scale_free_roughness,
)
from sampler_research.spectral import spectral_roughness


def gmm_nll_per_cell(field, pi, mu, sigma):
    """Per-location GMM negative log-likelihood `-log p_i(x_i)`.

    `field` has shape `[H, W]`; `pi`, `mu`, `sigma` have shape `[H, W, K]`.
    """

    density = mixture_pdf(field, pi, mu, sigma)
    return -np.log(density)


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


def smoothed_map_baseline(pi, mu, sigma, step=0.1, n_iters=10):
    """Smoothed-MAP critical baseline: per-cell mode field blurred over `E_8`.

    Takes the likelihood-only mode field and applies a graph-Laplacian blur on
    the default 8-neighbour grid. This is the baseline a real coupling method
    must beat on the orthogonal label-coherence axis.
    """

    field, _ = mode_field(pi, mu, sigma)
    height, width = field.shape
    laplacian = graph_laplacian(height, width)
    return laplacian_blur(field, laplacian, step=step, n_iters=n_iters)


def smoothest_mode_assignment(pi, mu, edges=None, use_pi_unary=False, max_sweeps=50):
    """Smoothest high-likelihood mode-assignment field `a*`.

    Choose one local component per cell to minimise the value-space smoothness
    cost `sum_(i,j in E) w_ij (mu_{i,k_i} - mu_{j,k_j})^2`, optionally with a
    `-log pi` unary penalty. Coupling is on component-mean *values*, never on the
    raw label index (C5).

    Solved by iterated conditional modes: warm-start from the highest-`pi`
    component, then repeatedly set each cell's component to the one minimising
    its local cost given the current neighbours. Returns `(field, assignment)`.

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

    neighbours = [[] for _ in range(n)]
    for i, j in edges:
        neighbours[i].append(j)
        neighbours[j].append(i)

    def _argmin_by_value(cost, values):
        """Index minimising `cost`, ties broken by smallest `values` (label-free)."""
        best = np.flatnonzero(np.isclose(cost, cost.min()))
        return int(best[np.argmin(values[best])])

    # Warm-start: highest pi, ties broken by smallest mean value (not label index).
    pi_flat = pi.reshape(n, k)
    assignment = np.array(
        [_argmin_by_value(-pi_flat[i], mu_flat[i]) for i in range(n)]
    )

    for _ in range(max_sweeps):
        changed = False
        for i in range(n):
            if not neighbours[i]:
                continue
            neighbour_vals = np.array([mu_flat[m, assignment[m]] for m in neighbours[i]])
            pairwise = np.sum((mu_flat[i][:, None] - neighbour_vals[None, :]) ** 2, axis=-1)
            best = _argmin_by_value(unary[i] + pairwise, mu_flat[i])
            if best != assignment[i]:
                assignment[i] = best
                changed = True
        if not changed:
            break

    field = mu_flat[np.arange(n), assignment].reshape(height, width)
    return field, assignment.reshape(height, width)


def score_field(field, pi, mu, sigma, edges=None):
    """Return Stage A scores for a produced field.

    The primary scores are `NLL/N` and scale-free `R̃` (§3.1a). Power-spectrum
    scores are secondary necessary-condition diagnostics for high-frequency
    roughness, not standalone skill metrics.
    """

    field_grid = np.asarray(field, dtype=float)
    if field_grid.ndim == 1:
        field_grid = field_grid.reshape(np.asarray(pi).shape[:2])

    if edges is None:
        height, width = field_grid.shape
        edges = grid_edges_8(height, width)
    nll = gmm_nll_over_n(field_grid, pi, mu, sigma)
    r_tilde, collapsed = scale_free_roughness(field_grid, edges)
    return {
        "nll_over_n": nll,
        "r_tilde": r_tilde,
        "variance_collapsed": collapsed,
        **spectral_roughness(field_grid),
    }
