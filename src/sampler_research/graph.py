"""Default toy graph, Laplacian, and roughness metrics for Stage A.

The conventions: the default
edge set is the 8-neighbour grid `E_8` (horizontal, vertical, and diagonal
neighbours, `w_ij = 1`), and roughness is reported scale-free as
`R̃ = S_edge / Var_V`.
"""

import numpy as np

NEIGHBOUR_OFFSETS_8 = (
    (1, 0),
    (0, 1),
    (1, 1),
    (1, -1),
)


def grid_edges_8(height, width):
    """Return the `E_8` edge list for an `height x width` grid.

    Cells are indexed in row-major (C) order, `i = r * width + c`. Each edge is
    an ordered pair `(i, j)` with `i < j`; every neighbour pair appears once.
    For the 8x8 toy this yields `|E_8| = 210` edges.
    """

    edges = []
    for r in range(height):
        for c in range(width):
            i = r * width + c
            for dr, dc in NEIGHBOUR_OFFSETS_8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    j = nr * width + nc
                    edges.append((i, j) if i < j else (j, i))
    return np.array(sorted(set(edges)), dtype=np.int64)


def graph_laplacian(height, width, edges=None, weights=None):
    """Build the dense graph Laplacian for the `E_8` grid graph.

    `L_ii = sum_j w_ij`, `L_ij = -w_ij` for neighbours, `0` otherwise. Default
    weights are `w_ij = 1`, including diagonals.
    """

    if edges is None:
        edges = grid_edges_8(height, width)
    n = height * width
    if weights is None:
        weights = np.ones(len(edges), dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(edges),):
        raise ValueError("weights must have one entry per edge")

    laplacian = np.zeros((n, n), dtype=float)
    for (i, j), w in zip(edges, weights):
        laplacian[i, j] -= w
        laplacian[j, i] -= w
        laplacian[i, i] += w
        laplacian[j, j] += w
    return laplacian


EARTH_RADIUS_KM = 6371.0


def _unit_sphere_xyz(latlons_deg):
    latlons = np.asarray(latlons_deg, dtype=float)
    if latlons.ndim != 2 or latlons.shape[1] != 2:
        raise ValueError("latlons_deg must have shape [N, 2]")
    lat = np.radians(latlons[:, 0])
    lon = np.radians(latlons[:, 1])
    return np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1
    )


def knn_sphere_edges(latlons_deg, k=8):
    """k-NN neighbour edges on the unit sphere.

    Embeds lat/lon as 3D unit vectors (chord distance is monotone in
    great-circle distance, so 3D k-NN == spherical k-NN; longitude wrap and the
    pole rings need no special casing). Symmetrised as the union
    (`i in knn(j)` OR `j in knn(i)`), deduplicated, returned as `[E, 2]` int64
    with `i < j`. `k=8` mirrors the toy `E_8` default and gives
    `|E| = 162,406` at N = 40,320.
    """

    from scipy.spatial import cKDTree

    xyz = _unit_sphere_xyz(latlons_deg)
    tree = cKDTree(xyz)
    _, idx = tree.query(xyz, k=k + 1)
    rows = np.repeat(np.arange(xyz.shape[0], dtype=np.int64), k)
    cols = idx[:, 1:].reshape(-1).astype(np.int64)
    keep = rows != cols
    rows, cols = rows[keep], cols[keep]
    pairs = np.stack([np.minimum(rows, cols), np.maximum(rows, cols)], axis=1)
    return np.unique(pairs, axis=0)


def edge_arc_km(latlons_deg, edges):
    """Great-circle length of each edge in km (chord -> arc on the unit sphere)."""

    xyz = _unit_sphere_xyz(latlons_deg)
    edges = np.asarray(edges, dtype=np.int64)
    chord = np.linalg.norm(xyz[edges[:, 0]] - xyz[edges[:, 1]], axis=1)
    return 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0)) * EARTH_RADIUS_KM


def sparse_laplacian(n_cells, edges, weights=None):
    """scipy.sparse CSR graph Laplacian; default unit weights (toy convention).

    The dense N x N Laplacian is forbidden at real-grid size (13 GB at
    N = 40,320); this is the flat-path replacement for
    `graph_laplacian`. A distance-decay weighting stays a declared ablation,
    not the default.
    """

    from scipy import sparse

    edges = np.asarray(edges, dtype=np.int64)
    if weights is None:
        weights = np.ones(len(edges), dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(edges),):
        raise ValueError("weights must have one entry per edge")

    i, j = edges[:, 0], edges[:, 1]
    rows = np.concatenate([i, j, i, j])
    cols = np.concatenate([j, i, i, j])
    vals = np.concatenate([-weights, -weights, weights, weights])
    return sparse.csr_array((vals, (rows, cols)), shape=(n_cells, n_cells))


def _as_flat_field(x):
    return np.asarray(x, dtype=float).reshape(-1)


def roughness_sum(x, laplacian):
    """Raw smoothness penalty `S_sum(x) = x^T L x = sum_(i,j) w_ij (x_i - x_j)^2`.

    Evaluated as `x @ (L @ x)` so dense arrays and sparse CSR operators share
    one code path.
    """

    flat = _as_flat_field(x)
    return float(flat @ (laplacian @ flat))


def roughness_edge_mean(x, edges):
    """Edge-averaged roughness `S_edge(x) = (1/|E|) sum_(i,j) w_ij (x_i - x_j)^2`.

    Uses unit edge weights, matching the default `E_8` convention.
    """

    flat = _as_flat_field(x)
    edges = np.asarray(edges, dtype=np.int64)
    diffs = flat[edges[:, 0]] - flat[edges[:, 1]]
    return float(np.mean(diffs**2))


def field_variance(x):
    """Variance of the produced field over all cells, `Var_V(x)` (§3.1a)."""

    flat = _as_flat_field(x)
    return float(np.mean((flat - flat.mean()) ** 2))


def scale_free_roughness(x, edges, var_floor=1e-12):
    """Scale-free roughness `R̃(x) = S_edge(x) / Var_V(x)`.

    Returns `(value, variance_collapsed)`. When `Var_V(x)` is at or below
    `var_floor` the field is essentially constant: `R̃` is undefined for
    reporting (§3.1a), so `variance_collapsed=True` is flagged and the returned
    value uses `var_floor` only as a numerical guard, not as a clean score.
    """

    s_edge = roughness_edge_mean(x, edges)
    var_v = field_variance(x)
    collapsed = var_v <= var_floor
    return s_edge / max(var_v, var_floor), collapsed


def laplacian_blur(x, laplacian, step=0.1, n_iters=10):
    """Apply a simple graph-Laplacian diffusion blur to a field.

    Repeated `x <- x - step * L x` smooths over the same `E_8` graph used for
    roughness. The update is mean-preserving (`L` rows sum to zero) and, for a
    small enough `step`, contracts toward the field mean. Quantitatively:
    stability requires `step < 2 / lambda_max(L)`; eigenmodes with
    `step * lambda > 1` decay with oscillating sign, so monotone smoothing
    needs `step < 1 / lambda_max(L)`. On the real union-kNN graph
    `lambda_max ~ 14.8`, so the production `step = 0.1` is stable but not
    monotone (`step * lambda_max ~ 1.48`) -- guarded by a canary test; check
    this bound again if `k`, the weights, or `step` ever change.
    """

    flat = _as_flat_field(x).copy()
    for _ in range(n_iters):
        flat = flat - step * (laplacian @ flat)
    return flat.reshape(np.asarray(x).shape)
