"""HISTORICAL (archived 11 Jun 2026) — toy TV min-cut certificate; job complete.

CUT from the real-data path per phase_4_plan.md "Scope verdicts": the quadratic
variant is O(|E|·G²) ≈ 7e8 arcs at N = 40,320 (infeasible) and the TV variant adds
no new claim. The verdict it produced — the exact TV mode is non-smearing only down
to R̃ ≈ 0.5 and then collapses rather than smears, while the global quadratic mode
still smears — now lives in large_notes.md (Phase 2 status) / phase_2.md; evidence in
notebooks/03_tv_ablation_exact_map.ipynb. Kept for the toy tests; do not port.

Exact MAP over a discretised value axis via Ishikawa min-cut (Stage B-TV).

The Stage B-TV ablation (phase_2_research_plan.md §4.1 field 6 / §8 #8) needs
the *global* optimum of

    J(x) = NLL(x)/N + lambda * (1/|E_8|) sum_(i,j) g(x_i - x_j)

for a convex per-edge penalty `g` (here `g(d) = |d|` for TV or `g(d) = d^2`
for Method 1's quadratic), with arbitrary multimodal per-cell unaries. After
restricting every cell to a shared uniform value grid `v_1 < ... < v_G`, that
discretised problem is solved *exactly* by a single s-t min-cut (Ishikawa,
TPAMI 2003): per cell a monotone chain of `G-1` binary "x_i >= v_{g+1}" nodes
whose data arcs carry the unaries, with convex pairwise costs encoded as arcs
between the chains. Every exactness claim made by this module is therefore
"exact for the discretised energy on this value grid" — the value-axis
resolution is controlled by the caller (G-vs-2G rerun) and degeneracy by a
deterministic unary tie-probe (`unary_dither`), both run by the Stage B-TV
script.

Construction notes (load-bearing):

* **Chains.** Cell `i` has nodes `n_{i,1} .. n_{i,G-1}`; node `n_{i,g}` on the
  source side means "label of `i` is > g". The data arc for label `g` runs
  `n_{i,g-1} -> n_{i,g}` (with `n_{i,0} = source`, `n_{i,G} = sink`) and
  carries that label's (shifted) unary; reverse arcs of capacity `B`
  ("infinity") make non-monotone cuts cost `> B`, so exactly one data arc per
  chain is cut. `B = min(sum of finite capacities + 1, integer cut value of
  the per-cell unary-argmin labeling + 1)` — both are strict upper bounds on
  the min cut, so no minimum cut ever pays `B`; asserted `< 2^31` (scipy's
  max-flow is int32).
* **TV pairwise.** `|x_i - x_j| = h * sum_g 1{n_{i,g}, n_{j,g} separated}` for
  monotone cuts, so each grid edge adds one arc pair per level: `O(|E| * G)`
  arcs of capacity `c = lambda*h/|E|`.
* **Quadratic pairwise.** The Ishikawa second-difference capacity of
  `g(d) = d^2` is constant, giving arcs between *all* level pairs
  (`O(|E| * G^2)` arcs of capacity `c = lambda*h^2/|E|`) plus a separable
  boundary correction `-c * deg(i) * a_i (G-1-a_i)` folded into the data arcs
  (the all-pairs cut counts `(a_i)(G-1-a_j) + (a_j)(G-1-a_i)` crossings, which
  exceeds `(a_i-a_j)^2` by exactly that separable term).
* **Integer capacities.** The shared pairwise capacity is rounded once,
  `c_int = round(c * scale)`, and the unary scale is then chosen as
  `scale_u = c_int / c` so the pairwise coefficient — and hence `lambda` — is
  represented *exactly*; quantisation only touches the `N` data arcs actually
  cut. Hence `quantisation_bound = N/2 / scale_u` (half an integer ulp per cut
  data arc) plus the dither allowance. Per-cell unaries are shifted to have
  integer minimum 0 ("per-column shift"); the shifts are restored when the cut
  value is reported in energy units.
* **Unary clipping.** Before scaling, `U_i(g)` is clipped at
  `min_g U_i(g) + N * lambda * deg(i) * W / |E| + 1` (with `W` the value-grid
  range for TV, range squared for quadratic). A label above that clip can
  never appear in an optimal labeling — swapping the cell to its unary-argmin
  label costs at most the clip margin in pairwise terms and wins more back in
  unary terms — so the optimum is unchanged while density-floor tails
  (`-log 1e-300 ~ 690`) stay out of the integer range. The float energy of the
  decoded field is recomputed from the *unclipped* densities and asserted to
  match the cut value within `quantisation_bound`, so a clip that somehow bit
  on the optimum would be caught, not silently absorbed.
* **Cut recovery.** The capacity matrix is built with a symmetrised CSR
  pattern (explicit zero reverse arcs) so `residual = capacities - flow` holds
  every traversable residual arc; BFS from the source over strictly positive
  residuals yields the minimal source side. Among tied optima this decodes the
  smallest labels — i.e. ties break toward the smallest value, matching the
  repo-wide C5 convention.

The shared scoreboard for the ablation is `tv_objective` (true TV, not the
Huber surrogate): both the Adam-on-Huber arm and the cut arm are scored with
it, which is what makes the `J_TV(adam) - J_TV(cut)` certificate meaningful.

Reuses the Stage A machinery: `mixture_pdf` / `gmm_nll_over_n` for the
unaries, the `E_8` graph + roughness metrics (`sampler_research.graph`), and
`spectral_roughness` for the secondary diagnostics. SciPy is used only for
`scipy.sparse.csgraph.maximum_flow` + `breadth_first_order`.
"""

from dataclasses import dataclass, field as _field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow

from sampler_research.baselines import gmm_nll_over_n
from sampler_research.gmm import mixture_pdf
from sampler_research.graph import grid_edges_8, roughness_edge_mean, scale_free_roughness
from sampler_research.spectral import spectral_roughness

# Matches the density floor used by method4_mrf for finite logs.
_DENSITY_FLOOR = 1e-300
_INT32_MAX = np.iinfo(np.int32).max

_DEFAULT_N_LEVELS = {"tv": 257, "quadratic": 129}


def value_grid(mu, sigma, *, n_levels=257, n_sigma=4.0):
    """Shared uniform value grid covering every cell's GMM support.

    Spans `[min(mu) - n_sigma*max(sigma), max(mu) + n_sigma*max(sigma)]` with
    `n_levels` points. Returns `(values [G], spacing h)`. Doubling the
    resolution as `2*G - 1` keeps the original `G` points as a subset (the
    G-control rerun relies on this).
    """

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    lo = float(mu.min() - n_sigma * sigma.max())
    hi = float(mu.max() + n_sigma * sigma.max())
    values = np.linspace(lo, hi, n_levels)
    return values, float(values[1] - values[0])


def unary_table(values, pi, mu, sigma):
    """Per-cell unaries `U[i, g] = -log p_i(v_g)`, shape `[N, G]` (raw nats).

    Broadcast pattern as in `method4_mrf.delta_nll_to_best_mode`; densities are
    floored at 1e-300 so the far tails stay finite (the solver clips them
    before integer scaling anyway).
    """

    values = np.asarray(values, dtype=float)
    pi = np.asarray(pi, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    k = pi.shape[-1]
    n = pi.size // k

    pi_f = pi.reshape(n, 1, k)
    mu_f = mu.reshape(n, 1, k)
    sigma_f = sigma.reshape(n, 1, k)
    density = mixture_pdf(values[None, :], pi_f, mu_f, sigma_f)  # [N, G]
    return -np.log(np.maximum(density, _DENSITY_FLOOR))


def tv_objective(field_grid, pi, mu, sigma, lam, edges=None, n_edges=None):
    """J_TV(x) = NLL(x)/N + lambda * (1/|E_8|) sum_(i,j) |x_i - x_j|.

    The shared scoreboard for the TV ablation: the Adam-on-Huber arm and the
    exact-cut arm are both scored with this (true TV, not the Huber surrogate),
    in the same per-cell / per-edge units as Method 1's `J_lambda` (§3.1a).
    """

    field_grid = np.asarray(field_grid, dtype=float)
    height, width = field_grid.shape
    if edges is None:
        edges = grid_edges_8(height, width)
    edges = np.asarray(edges, dtype=np.int64)
    if n_edges is None:
        n_edges = len(edges)

    flat = field_grid.reshape(-1)
    tv = float(np.sum(np.abs(flat[edges[:, 0]] - flat[edges[:, 1]]))) / n_edges
    return gmm_nll_over_n(field_grid, pi, mu, sigma) + lam * tv


def _quadratic_objective(field_grid, pi, mu, sigma, lam, edges):
    """Method 1's J_lambda recomputed edge-wise (== regularised_map.objective)."""

    return gmm_nll_over_n(field_grid, pi, mu, sigma) + lam * roughness_edge_mean(
        field_grid, edges
    )


def _ishikawa_min_cut(unary_j, c, penalty, edges, scale):
    """Solve `min_a sum_i unary_j[i, a_i] + c * sum_(i,j) g(a_i - a_j)` exactly.

    `unary_j` is `[N, G]` in energy units, `c` the per-edge pairwise
    coefficient in the same units per unit *label*-step cost (`g(d) = |d|` for
    `penalty="tv"`, `g(d) = d^2` for `"quadratic"`, `d` in label steps).
    Returns `(labels [N], cut_value, scale_u)` with `cut_value` in energy units
    (per-column shifts restored).
    """

    n, n_grid = unary_j.shape
    n_chain = n_grid - 1
    edges = np.asarray(edges, dtype=np.int64)

    # Shared pairwise capacity rounded once; unary scale chosen so the pairwise
    # coefficient (hence lambda) is exact in integer arithmetic.
    if c > 0.0:
        c_int = max(int(round(c * scale)), 1)
        scale_u = c_int / c
    else:
        c_int = 0
        scale_u = float(scale)

    if penalty == "quadratic":
        # Separable correction for the all-pairs construction (module docstring).
        labels_axis = np.arange(n_grid, dtype=float)
        h_corr = labels_axis * (n_grid - 1 - labels_axis)  # a * (G-1-a)
        deg = np.bincount(edges.ravel(), minlength=n).astype(float)
        unary_eff = unary_j - (c_int / scale_u) * deg[:, None] * h_corr[None, :]
    else:
        unary_eff = unary_j

    shifts = unary_eff.min(axis=1)
    unary_int = np.rint((unary_eff - shifts[:, None]) * scale_u).astype(np.int64)

    # "Infinity" for the monotonicity (reverse) arcs: strictly above the min
    # cut, via the cheaper of (sum of finite caps + 1) and (integer cut value
    # of the per-cell unary-argmin labeling + 1).
    g_star = np.argmin(unary_int, axis=1)
    p, q = g_star[edges[:, 0]], g_star[edges[:, 1]]
    if penalty == "quadratic":
        crossings = p * (n_grid - 1 - q) + q * (n_grid - 1 - p)
    else:
        crossings = np.abs(p - q)
    feasible_cut = int(unary_int[np.arange(n), g_star].sum()) + c_int * int(crossings.sum())
    if penalty == "quadratic":
        n_pair_arcs = 2 * len(edges) * n_chain * n_chain
    else:
        n_pair_arcs = 2 * len(edges) * n_chain
    finite_sum = int(unary_int.sum()) + c_int * n_pair_arcs
    big = min(finite_sum, feasible_cut) + 1
    if big >= _INT32_MAX or unary_int.max() >= _INT32_MAX:
        raise ValueError(
            "integer capacities exceed int32; lower `scale` or the unary range"
        )

    chain_base = np.arange(n, dtype=np.int64) * n_chain  # node(i, g) = base[i] + g - 1
    source = n * n_chain
    sink = source + 1
    n_nodes = sink + 1

    rows, cols, caps = [], [], []

    # Data arcs: label a=0 (source -> n_{i,1}), internal, a=G-1 (n_{i,G-1} -> sink).
    rows.append(np.full(n, source, dtype=np.int64))
    cols.append(chain_base)
    caps.append(unary_int[:, 0])
    if n_grid >= 3:
        a = np.arange(1, n_grid - 1, dtype=np.int64)
        rows.append((chain_base[:, None] + (a[None, :] - 1)).ravel())
        cols.append((chain_base[:, None] + a[None, :]).ravel())
        caps.append(unary_int[:, 1 : n_grid - 1].ravel())
    rows.append(chain_base + (n_chain - 1))
    cols.append(np.full(n, sink, dtype=np.int64))
    caps.append(unary_int[:, n_grid - 1])

    # Monotonicity: reverse arcs of capacity `big` along each chain.
    if n_grid >= 3:
        g = np.arange(1, n_grid - 1, dtype=np.int64)
        rows.append((chain_base[:, None] + g[None, :]).ravel())
        cols.append((chain_base[:, None] + (g[None, :] - 1)).ravel())
        caps.append(np.full(n * (n_grid - 2), big, dtype=np.int64))

    # Pairwise arcs (both directions).
    if c_int > 0:
        base_i = chain_base[edges[:, 0]]
        base_j = chain_base[edges[:, 1]]
        if penalty == "tv":
            lvl = np.arange(n_chain, dtype=np.int64)
            fwd = (base_i[:, None] + lvl[None, :]).ravel()
            bwd = (base_j[:, None] + lvl[None, :]).ravel()
        else:
            lvl = np.arange(n_chain, dtype=np.int64)
            fwd = np.broadcast_to(
                (base_i[:, None] + lvl[None, :])[:, :, None],
                (len(edges), n_chain, n_chain),
            ).ravel()
            bwd = np.broadcast_to(
                (base_j[:, None] + lvl[None, :])[:, None, :],
                (len(edges), n_chain, n_chain),
            ).ravel()
        rows.extend([fwd, bwd])
        cols.extend([bwd, fwd])
        caps.extend([np.full(fwd.size, c_int, dtype=np.int64)] * 2)

    row = np.concatenate(rows)
    col = np.concatenate(cols)
    cap = np.concatenate(caps)
    # Symmetrise the pattern with explicit zero reverse arcs so the residual
    # matrix below contains every traversable arc; coo->csr sums duplicates.
    graph = coo_matrix(
        (
            np.concatenate([cap, np.zeros_like(cap)]),
            (np.concatenate([row, col]), np.concatenate([col, row])),
        ),
        shape=(n_nodes, n_nodes),
    ).tocsr()
    graph = graph.astype(np.int32)

    result = maximum_flow(graph, int(source), int(sink))

    residual = graph - result.flow
    residual.data[residual.data < 0] = 0
    residual.eliminate_zeros()
    reached = breadth_first_order(residual, int(source), directed=True, return_predecessors=False)
    on_source_side = np.zeros(n_nodes, dtype=bool)
    on_source_side[reached] = True

    node_side = on_source_side[: n * n_chain].reshape(n, n_chain)
    labels = node_side.sum(axis=1).astype(np.int64)
    # Monotone-cut sanity: the source-side nodes of each chain form a prefix.
    expected = np.arange(n_chain)[None, :] < labels[:, None]
    if not np.array_equal(node_side, expected):
        raise RuntimeError("min-cut decoding produced a non-monotone chain cut")

    cut_value = float(result.flow_value) / scale_u + float(shifts.sum())
    return labels, cut_value, scale_u


@dataclass
class ExactMapResult:
    """Outcome of one exact (discretised) MAP solve at a fixed lambda."""

    field: np.ndarray  # decoded field, [H, W]
    labels: np.ndarray  # value-grid index per cell, [H, W]
    energy: float  # float J of `field` under the true (unclipped) objective
    cut_value: float  # min-cut value in energy units (shifts restored)
    quantisation_gap: float  # |energy - cut_value|
    quantisation_bound: float  # certified bound on the gap (integer rounding)
    nll_over_n: float
    r_tilde: float
    variance_collapsed: bool
    spectral_hf_ratio: float
    spectral_slope: float
    spectral_monotone_fraction: float
    spectral_collapsed: bool
    lam: float
    penalty: str
    n_levels: int
    grid_spacing: float


def exact_map_solve(
    pi,
    mu,
    sigma,
    lam,
    *,
    penalty="tv",
    n_levels=None,
    n_sigma=4.0,
    edges=None,
    scale=2**20,
    unary_dither=None,
):
    """Exact MAP of the discretised `NLL/N + lambda * penalty` by one min-cut.

    `penalty` is `"tv"` (`O(|E|*G)` arcs, default G=257) or `"quadratic"`
    (`O(|E|*G^2)` arcs, default G=129 — certificate mode only). `unary_dither`
    adds a deterministic ±dither (raw nats, fixed sign pattern) to the clipped
    unaries: the tie probe — rerunning with `unary_dither=1e-9` perturbs only
    knife-edge ties in the integer encoding, so a changed field flags a
    degenerate optimum. The decoded field's energy is recomputed in float from
    the unclipped densities and must match `cut_value` within
    `quantisation_bound`; "exact" always means exact for the discretised
    energy on this value grid.
    """

    if penalty not in ("tv", "quadratic"):
        raise ValueError(f"unknown penalty {penalty!r}; expected 'tv' or 'quadratic'")
    pi = np.asarray(pi, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    height, width, _ = mu.shape
    n = height * width
    if n_levels is None:
        n_levels = _DEFAULT_N_LEVELS[penalty]
    if edges is None:
        edges = grid_edges_8(height, width)
    edges = np.asarray(edges, dtype=np.int64)
    n_edges = len(edges)

    values, spacing = value_grid(mu, sigma, n_levels=n_levels, n_sigma=n_sigma)
    unary_raw = unary_table(values, pi, mu, sigma)  # [N, G] raw nats

    # Clip can-never-be-optimal tails (module docstring), then dither (after
    # the clip, so the probe survives it).
    value_range = float(values[-1] - values[0])
    range_pen = value_range if penalty == "tv" else value_range**2
    deg = np.bincount(edges.ravel(), minlength=n).astype(float)
    clip = unary_raw.min(axis=1) + n * lam * deg * range_pen / n_edges + 1.0
    unary = np.minimum(unary_raw, clip[:, None])
    if unary_dither is not None:
        signs = np.random.default_rng(0).integers(0, 2, size=unary.shape) * 2 - 1
        unary = unary + float(unary_dither) * signs

    pair_unit = spacing if penalty == "tv" else spacing**2
    c = lam * pair_unit / n_edges
    labels, cut_value, scale_u = _ishikawa_min_cut(unary / n, c, penalty, edges, scale)

    field_grid = values[labels].reshape(height, width)
    if penalty == "tv":
        energy = tv_objective(field_grid, pi, mu, sigma, lam, edges=edges, n_edges=n_edges)
    else:
        energy = _quadratic_objective(field_grid, pi, mu, sigma, lam, edges)

    quantisation_gap = abs(energy - cut_value)
    dither_allowance = abs(unary_dither) if unary_dither is not None else 0.0
    quantisation_bound = 0.5 * n / scale_u + dither_allowance + 1e-9
    if quantisation_gap > quantisation_bound:
        raise RuntimeError(
            f"decoded energy {energy!r} disagrees with cut value {cut_value!r} "
            f"beyond the quantisation bound {quantisation_bound!r}"
        )

    r_tilde, collapsed = scale_free_roughness(field_grid, edges)
    spectral = spectral_roughness(field_grid)
    return ExactMapResult(
        field=field_grid,
        labels=labels.reshape(height, width),
        energy=float(energy),
        cut_value=float(cut_value),
        quantisation_gap=float(quantisation_gap),
        quantisation_bound=float(quantisation_bound),
        nll_over_n=gmm_nll_over_n(field_grid, pi, mu, sigma),
        r_tilde=r_tilde,
        variance_collapsed=collapsed,
        spectral_hf_ratio=spectral["spectral_hf_ratio"],
        spectral_slope=spectral["spectral_slope"],
        spectral_monotone_fraction=spectral["spectral_monotone_fraction"],
        spectral_collapsed=spectral["spectral_collapsed"],
        lam=float(lam),
        penalty=penalty,
        n_levels=int(n_levels),
        grid_spacing=spacing,
    )


@dataclass
class ExactSweepPoint:
    """One row of the exact-MAP lambda sweep (mirrors `LambdaSweepPoint`)."""

    lam: float
    nll_over_n: float
    r_tilde: float
    variance_collapsed: bool
    spectral_hf_ratio: float
    spectral_slope: float
    spectral_monotone_fraction: float
    spectral_collapsed: bool
    energy: float
    cut_value: float
    quantisation_gap: float
    quantisation_bound: float
    n_levels: int
    grid_spacing: float
    penalty: str
    field: np.ndarray = _field(repr=False)
    labels: np.ndarray = _field(repr=False)


def exact_map_sweep(
    pi,
    mu,
    sigma,
    lambdas,
    *,
    penalty="tv",
    n_levels=None,
    n_sigma=4.0,
    edges=None,
    scale=2**20,
):
    """Exact discretised MAP at each lambda (deterministic — no RNG anywhere).

    Returns a list of `ExactSweepPoint`, one per lambda, all on the same value
    grid. Unlike `lambda_sweep` there are no restarts or seeds: each point is
    the global optimum of its discretised energy.
    """

    pi = np.asarray(pi, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    points = []
    for lam in lambdas:
        res = exact_map_solve(
            pi,
            mu,
            sigma,
            lam,
            penalty=penalty,
            n_levels=n_levels,
            n_sigma=n_sigma,
            edges=edges,
            scale=scale,
        )
        points.append(
            ExactSweepPoint(
                lam=float(lam),
                nll_over_n=res.nll_over_n,
                r_tilde=res.r_tilde,
                variance_collapsed=res.variance_collapsed,
                spectral_hf_ratio=res.spectral_hf_ratio,
                spectral_slope=res.spectral_slope,
                spectral_monotone_fraction=res.spectral_monotone_fraction,
                spectral_collapsed=res.spectral_collapsed,
                energy=res.energy,
                cut_value=res.cut_value,
                quantisation_gap=res.quantisation_gap,
                quantisation_bound=res.quantisation_bound,
                n_levels=res.n_levels,
                grid_spacing=res.grid_spacing,
                penalty=res.penalty,
                field=res.field,
                labels=res.labels,
            )
        )
    return points
