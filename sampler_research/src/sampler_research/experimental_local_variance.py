"""Experimental local-variance roughness diagnostics for Phase 4.

TEST CODE ONLY: this module is deliberately outside the production
``graph.scale_free_roughness`` path. It is for review experiments on whether a
local variance denominator changes the interpretation of the Method 1 lambda
sweep. Do not use in submission figures, tables, or sampler selection rules
without explicit review.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sampler_research.graph import EARTH_RADIUS_KM, edge_arc_km


TEST_ONLY_NOTICE = (
    "EXPERIMENTAL TEST ONLY: local-variance roughness diagnostic. "
    "Review before submission, merge, or production sampler use."
)


@dataclass(frozen=True)
class LocalVarianceNeighborhoods:
    """Spherical-cap neighborhoods used to estimate each cell's local variance."""

    indices: tuple[np.ndarray, ...]
    radius_grid_points: int
    radius_km: float


@dataclass(frozen=True)
class LocalVarianceRoughness:
    """Experimental local-denominator roughness score for one field."""

    r_tilde: float
    variance_collapsed: bool
    radius_grid_points: int
    radius_km: float
    median_local_variance: float
    collapsed_cell_fraction: float


def _unit_sphere_xyz(latlons_deg):
    latlons = np.asarray(latlons_deg, dtype=float)
    if latlons.ndim != 2 or latlons.shape[1] != 2:
        raise ValueError("latlons_deg must have shape [N, 2]")
    lat = np.radians(latlons[:, 0])
    lon = np.radians(latlons[:, 1])
    return np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1
    )


def build_local_variance_neighborhoods(
    latlons_deg,
    edges,
    *,
    radius_grid_points=10,
    edge_lengths_km=None,
):
    """Build spherical-cap neighborhoods for the test local-variance metric.

    ``radius_grid_points`` is converted to kilometres as
    ``radius_grid_points * median(edge arc length)`` on the same spherical k-NN
    graph used by Method 1. The neighborhood is a true spherical circle/cap
    around each point, not a row/column crop.
    """

    if radius_grid_points <= 0:
        raise ValueError("radius_grid_points must be positive")

    edges = np.asarray(edges, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must have shape [E, 2]")

    if edge_lengths_km is None:
        edge_lengths = edge_arc_km(latlons_deg, edges)
    else:
        edge_lengths = np.asarray(edge_lengths_km, dtype=float)
    if edge_lengths.shape != (len(edges),):
        raise ValueError("edge_lengths_km must have one entry per edge")

    radius_km = float(radius_grid_points * np.median(edge_lengths))
    xyz = _unit_sphere_xyz(latlons_deg)

    from scipy.spatial import cKDTree

    tree = cKDTree(xyz)
    angular_radius = min(radius_km / EARTH_RADIUS_KM, np.pi)
    chord_radius = 2.0 * np.sin(0.5 * angular_radius)
    neighborhoods = tree.query_ball_point(xyz, r=chord_radius)
    indices = tuple(np.asarray(neigh, dtype=np.int64) for neigh in neighborhoods)
    return LocalVarianceNeighborhoods(
        indices=indices,
        radius_grid_points=int(radius_grid_points),
        radius_km=radius_km,
    )


def local_variance(field, neighborhoods):
    """Variance of ``field`` inside each precomputed local spherical cap."""

    flat = np.asarray(field, dtype=float).reshape(-1)
    if len(neighborhoods.indices) != flat.size:
        raise ValueError("neighborhood count must match field size")

    out = np.empty(flat.size, dtype=float)
    for i, idx in enumerate(neighborhoods.indices):
        values = flat[idx]
        out[i] = float(np.mean((values - values.mean()) ** 2)) if values.size else 0.0
    return out


def local_scale_free_roughness(field, edges, neighborhoods, *, var_floor=1e-12):
    """Experimental roughness with local, not global, variance denominators.

    For each edge, the denominator is the average of the two endpoint local
    variances. The reported score is the mean of
    ``(x_i - x_j)^2 / local_edge_variance`` over graph edges.
    """

    flat = np.asarray(field, dtype=float).reshape(-1)
    edges = np.asarray(edges, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must have shape [E, 2]")
    if edges.size and int(edges.max()) >= flat.size:
        raise ValueError("edges reference cells outside the field")

    local_var = local_variance(flat, neighborhoods)
    edge_var = 0.5 * (local_var[edges[:, 0]] + local_var[edges[:, 1]])
    diffs = flat[edges[:, 0]] - flat[edges[:, 1]]
    collapsed_cells = local_var <= var_floor
    collapsed = bool(np.all(collapsed_cells))
    r_tilde = float(np.mean(diffs**2 / np.maximum(edge_var, var_floor)))
    return LocalVarianceRoughness(
        r_tilde=r_tilde,
        variance_collapsed=collapsed,
        radius_grid_points=neighborhoods.radius_grid_points,
        radius_km=neighborhoods.radius_km,
        median_local_variance=float(np.median(local_var)),
        collapsed_cell_fraction=float(np.mean(collapsed_cells)),
    )
