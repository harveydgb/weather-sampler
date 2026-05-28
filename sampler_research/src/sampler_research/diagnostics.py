"""Diagnostic calculations for Phase 1 and real-output notebooks."""

import numpy as np


def empirical_variogram(
    coords,
    fields,
    n_bins=12,
):
    """Compute a full-pair Euclidean empirical variogram for small grids."""

    flat_coords = np.asarray(coords, dtype=float).reshape(-1, coords.shape[-1])
    diff = flat_coords[:, None, :] - flat_coords[None, :, :]
    pair_dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
    iu = np.triu_indices(flat_coords.shape[0], k=1)
    pair_dist = pair_dist_matrix[iu]

    bins = np.linspace(0.0, float(pair_dist.max()), n_bins + 1)
    bin_ids = np.clip(np.digitize(pair_dist, bins) - 1, 0, n_bins - 1)
    centres = 0.5 * (bins[:-1] + bins[1:])

    variograms = {}
    for name, field in fields.items():
        flat = np.asarray(field, dtype=float).reshape(-1)
        sq_diff = (flat[:, None] - flat[None, :]) ** 2
        pair_sq = sq_diff[iu]
        variograms[name] = np.array(
            [
                0.5 * pair_sq[bin_ids == b].mean() if np.any(bin_ids == b) else np.nan
                for b in range(n_bins)
            ]
        )
    return centres, variograms


def spherical_pair_dist_deg(
    lat1_deg,
    lon1_deg,
    lat2_deg,
    lon2_deg,
):
    """Great-circle angular separation in degrees."""

    lat1 = np.radians(lat1_deg)
    lon1 = np.radians(lon1_deg)
    lat2 = np.radians(lat2_deg)
    lon2 = np.radians(lon2_deg)
    cos_d = (
        np.sin(lat1) * np.sin(lat2)
        + np.cos(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    )
    dist = np.degrees(np.arccos(np.clip(cos_d, -1.0, 1.0)))
    return np.where(np.isclose(cos_d, 1.0), 0.0, dist)


def sampled_spherical_variogram(
    latlons_deg,
    fields,
    n_pairs=60_000,
    bin_width_deg=2.0,
    seed=42,
):
    """Sample point pairs and compute angular-distance variograms for large grids."""

    latlons = np.asarray(latlons_deg, dtype=float)
    if latlons.ndim != 2 or latlons.shape[1] != 2:
        raise ValueError("latlons_deg must have shape [N, 2]")

    rng = np.random.default_rng(seed)
    n = latlons.shape[0]
    i_p = rng.integers(0, n, size=n_pairs)
    j_p = rng.integers(0, n, size=n_pairs)
    keep = i_p != j_p
    i_p = i_p[keep]
    j_p = j_p[keep]

    dist_deg = spherical_pair_dist_deg(
        latlons[i_p, 0],
        latlons[i_p, 1],
        latlons[j_p, 0],
        latlons[j_p, 1],
    )
    bins = np.arange(0.0, 180.0 + bin_width_deg, bin_width_deg)
    centres = 0.5 * (bins[:-1] + bins[1:])
    bin_ids = np.clip(np.digitize(dist_deg, bins) - 1, 0, len(centres) - 1)

    variograms = {}
    for name, field in fields.items():
        values = np.asarray(field, dtype=float).reshape(-1)
        semivariance = 0.5 * (values[i_p] - values[j_p]) ** 2
        variograms[name] = np.array(
            [
                semivariance[bin_ids == b].mean() if np.any(bin_ids == b) else np.nan
                for b in range(len(centres))
            ]
        )
    return centres, variograms, dist_deg
