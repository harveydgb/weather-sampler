"""Phase 4 evaluation protocol: strata, stratified scores, and the lambda* rule.

Shared by the runner (`scripts/run_real_eval.py`), the report notebook, and
the test suite so all three quote identical numbers. The protocol facts live in
phase_4_plan.md (S5 diagnostics, S6 lambda-selection rule) and the data facts in
phase_4_data_audit.md (S4 multimodality census, S5 grid). A global mean dilutes
the ~604-cell practically-bimodal subset ~67:1, so every metric is reported per
stratum; the bimodal stratum is the headline read.
"""

from dataclasses import dataclass, field

import numpy as np


# Order is part of the artifact contract (masks.npz / scores.csv columns).
STRATUM_ORDER = (
    "global",
    "lat_polar",
    "lat_mid",
    "lat_tropics",
    "bimodal",
    "unimodal",
    "keff_gt_1p5",
)

# Delta-NLL drift thresholds, v1 working values (method4_mrf.DRIFT_THRESHOLDS_V1).
DNLL_THRESHOLDS = (0.125, 0.5)


def effective_k(pi):
    """Effective component count per cell, `exp` of the weight entropy."""

    pi = np.asarray(pi, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(pi > 0.0, pi * np.log(pi), 0.0)
    return np.exp(-np.sum(terms, axis=-1))


def practically_bimodal_mask(pi, mu, sigma, pi_min=0.1, min_separation=2.0):
    """Audit S4 rule: some component pair with both `pi >= pi_min` separated
    by more than `min_separation * max(sigma_i, sigma_j)`.

    Gates on the real 2t marginal: 604 cells at 2 sigma, 5,576 at 1 sigma.
    """

    pi = np.asarray(pi, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    k = pi.shape[-1]

    mask = np.zeros(pi.shape[:-1], dtype=bool)
    for a in range(k):
        for b in range(a + 1, k):
            sep = np.abs(mu[..., a] - mu[..., b]) > min_separation * np.maximum(
                sigma[..., a], sigma[..., b]
            )
            mask |= (pi[..., a] >= pi_min) & (pi[..., b] >= pi_min) & sep
    return mask


def stratum_masks(pi, mu, sigma, latlons):
    """Ordered dict of the seven scoring strata (`STRATUM_ORDER`) `[N]` bool.

    Latitude bands: polar `|lat| >= 55 deg`, mid `23 <= |lat| < 55`, tropics
    `|lat| < 23`. `bimodal` uses the 2-sigma audit rule; `unimodal` is its
    complement; `keff_gt_1p5` is the diffuse-weight tail (30.0% on this data).
    """

    latlons = np.asarray(latlons, dtype=float)
    abs_lat = np.abs(latlons[:, 0])
    bimodal = practically_bimodal_mask(pi, mu, sigma)
    return {
        "global": np.ones(latlons.shape[0], dtype=bool),
        "lat_polar": abs_lat >= 55.0,
        "lat_mid": (abs_lat >= 23.0) & (abs_lat < 55.0),
        "lat_tropics": abs_lat < 23.0,
        "bimodal": bimodal,
        "unimodal": ~bimodal,
        "keff_gt_1p5": effective_k(pi) > 1.5,
    }


def subgraph_edges(edges, mask):
    """Edges with *both* endpoints inside `mask` (per-stratum roughness graph)."""

    edges = np.asarray(edges, dtype=np.int64)
    mask = np.asarray(mask, dtype=bool)
    keep = mask[edges[:, 0]] & mask[edges[:, 1]]
    return edges[keep]


def _stratum_roughness(flat_field, edges_sub, values, var_floor=1e-12):
    """`(s_edge, r_tilde, collapsed)` on a stratum subgraph and cell subset."""

    if len(edges_sub) == 0:
        return float("nan"), float("nan"), True
    diffs = flat_field[edges_sub[:, 0]] - flat_field[edges_sub[:, 1]]
    s_edge = float(np.mean(diffs**2))
    var_v = float(np.mean((values - values.mean()) ** 2))
    collapsed = var_v <= var_floor
    return s_edge, s_edge / max(var_v, var_floor), collapsed


def stratified_scores(field_values, nll_per_cell, edges, masks, delta_per_cell):
    """One flat scores row: per-stratum NLL/N, S_edge, R-tilde (+collapse), dNLL tail.

    The `global` stratum keeps the house column names; every other stratum
    suffixes `__<stratum>`. R-tilde per stratum uses the both-endpoints-in-stratum
    subgraph and the stratum's own cell variance, with its own collapse flag.
    """

    flat = np.asarray(field_values, dtype=float).reshape(-1)
    nll_per_cell = np.asarray(nll_per_cell, dtype=float).reshape(-1)
    delta_per_cell = np.asarray(delta_per_cell, dtype=float).reshape(-1)

    row = {}
    for name, mask in masks.items():
        suffix = "" if name == "global" else f"__{name}"
        values = flat[mask]
        s_edge, r_tilde, collapsed = _stratum_roughness(flat, subgraph_edges(edges, mask), values)
        delta = delta_per_cell[mask]
        row[f"n_cells{suffix}"] = int(mask.sum())
        row[f"nll_over_n{suffix}"] = float(np.mean(nll_per_cell[mask]))
        row[f"s_edge{suffix}"] = s_edge
        row[f"r_tilde{suffix}"] = r_tilde
        row[f"variance_collapsed{suffix}"] = bool(collapsed)
        row[f"dnll_mean{suffix}"] = float(np.mean(delta))
        row[f"dnll_p95{suffix}"] = float(np.percentile(delta, 95))
        row[f"dnll_frac_gt_0p125{suffix}"] = float(np.mean(delta > DNLL_THRESHOLDS[0]))
        row[f"dnll_frac_gt_0p5{suffix}"] = float(np.mean(delta > DNLL_THRESHOLDS[1]))
    return row


def wrap_seam_ratio(field_values, latlons, edges, lon_threshold_deg=350.0):
    """Date-line seam guard (phase_4_plan S4.9).

    Mean edge-squared-diff over edges whose endpoints differ by more than
    `lon_threshold_deg` in longitude, divided by the all-edge mean. ~1 for a
    healthy field; a longitude-wrap bug shows up as a large ratio. Returns NaN
    if the graph has no wrap edges (then there is nothing to guard).
    """

    flat = np.asarray(field_values, dtype=float).reshape(-1)
    latlons = np.asarray(latlons, dtype=float)
    edges = np.asarray(edges, dtype=np.int64)

    dlon = np.abs(latlons[edges[:, 0], 1] - latlons[edges[:, 1], 1])
    wrap = dlon > lon_threshold_deg
    if not np.any(wrap):
        return float("nan")
    diffs_sq = (flat[edges[:, 0]] - flat[edges[:, 1]]) ** 2
    return float(diffs_sq[wrap].mean() / diffs_sq.mean())


@dataclass
class LambdaStarResult:
    """Outcome of the S6 roughness-matching lambda* rule, with fallback bookkeeping."""

    lambda_star: float
    target_r_tilde: float
    matched_r_tilde: float  # R-tilde implied at lambda* (== target when bracketed)
    bracketed: bool
    extend_direction: str = None  # "up" / "down" when unbracketed, else None
    clauses: list = field(default_factory=list)  # which S6 fallback clauses fired


def select_lambda_star(lambdas, r_tildes, collapsed, target_r_tilde, valid=None):
    """Roughness-matching rule lambda* (phase_4_plan S6), total over its fallbacks.

    (a) collapse-flagged rows are excluded (R-tilde undefined under collapse);
    (b) interpolation is linear in R-tilde vs **log lambda over positive lambda
    only** (the lambda=0 row is an anchor, not an interpolation point);
    (c) multiple crossings -> the smallest crossing lambda; (d) no bracket ->
    nearest valid row, with `bracketed=False` and the direction the sweep should
    be extended (decade-by-decade, handled by the caller). `valid` can mask out
    additional rows (e.g. `warm_start_sane=False`). Raises `ValueError` when no
    positive-lambda row is valid -- the caller reports the smallest-lambda row
    and flags the mismatch.
    """

    lambdas = np.asarray(lambdas, dtype=float)
    r_tildes = np.asarray(r_tildes, dtype=float)
    collapsed = np.asarray(collapsed, dtype=bool)
    valid = np.ones(len(lambdas), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)

    clauses = []
    if np.any(collapsed & (lambdas > 0.0) & valid):
        clauses.append("a_collapsed_rows_excluded")
    keep = (lambdas > 0.0) & ~collapsed & valid & np.isfinite(r_tildes)
    if not np.any(keep):
        raise ValueError("no valid positive-lambda rows for the roughness match")

    order = np.argsort(lambdas[keep])
    lam = lambdas[keep][order]
    r = r_tildes[keep][order]
    log_lam = np.log(lam)

    crossings = []
    for i in range(len(lam) - 1):
        lo, hi = r[i], r[i + 1]
        if (lo - target_r_tilde) * (hi - target_r_tilde) <= 0.0:
            if np.isclose(lo, hi):
                crossings.append(log_lam[i])
            else:
                t = (target_r_tilde - lo) / (hi - lo)
                crossings.append(log_lam[i] + t * (log_lam[i + 1] - log_lam[i]))
    if crossings:
        if len(crossings) > 1:
            clauses.append("c_smallest_crossing")
        return LambdaStarResult(
            lambda_star=float(np.exp(min(crossings))),
            target_r_tilde=float(target_r_tilde),
            matched_r_tilde=float(target_r_tilde),
            bracketed=True,
            extend_direction=None,
            clauses=clauses,
        )

    clauses.append("d_unbracketed_nearest_row")
    nearest = int(np.argmin(np.abs(r - target_r_tilde)))
    # R-tilde decreases with lambda: target below every valid R-tilde needs more smoothing.
    extend_direction = "up" if target_r_tilde < r.min() else "down"
    return LambdaStarResult(
        lambda_star=float(lam[nearest]),
        target_r_tilde=float(target_r_tilde),
        matched_r_tilde=float(r[nearest]),
        bracketed=False,
        extend_direction=extend_direction,
        clauses=clauses,
    )
