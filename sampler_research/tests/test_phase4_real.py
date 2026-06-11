"""Phase 4 port tests: real-marginal loader, sphere graph, flat paths, lambda*.

Covers the edge-case list of phase_4_plan.md S4 plus the evaluation-protocol
unit tests. Real-data cases skip when the converted npz is absent (it is a
local, gitignored artifact); toy regressions skip when the persisted Stage A/B
artifacts are absent (regenerate via scripts/run_stage_*.py).
"""

import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest

from sampler_research.baselines import (
    gmm_nll_per_cell,
    mode_field,
    score_field,
    smoothed_map_baseline,
)
from sampler_research.gmm import gmm_log_pdf, mixture_pdf, sample_iid_gmm
from sampler_research.graph import (
    edge_arc_km,
    grid_edges_8,
    knn_sphere_edges,
    laplacian_blur,
    roughness_sum,
    scale_free_roughness,
    sparse_laplacian,
)
from sampler_research.io import load_real_marginal, load_sampler_arrays
from sampler_research.method4_mrf import (
    delta_nll_to_best_mode,
    extract_gmm_modes,
    solve_value_mrf,
)
from sampler_research.phase4_eval import (
    effective_k,
    practically_bimodal_mask,
    select_lambda_star,
    stratified_scores,
    stratum_masks,
    wrap_seam_ratio,
)
from sampler_research.regularised_map import minimise_at_lambda, nll_gradient

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_NPZ = REPO_ROOT / "outputs" / "data" / "phase_4_real_2t.npz"
TOY_NPZ = REPO_ROOT / "outputs" / "data" / "phase_1_homoscedastic.npz"
STAGE_A_DIR = REPO_ROOT / "outputs" / "runs" / "stage_a_baselines"
STAGE_B_NPZ = (
    REPO_ROOT / "outputs" / "runs" / "stage_b_regularised_map"
    / "phase_1_homoscedastic_regularised_map.npz"
)
PHASE4_RUN_DIR = REPO_ROOT / "outputs" / "runs" / "phase_4_real"

needs_real = pytest.mark.skipif(not REAL_NPZ.exists(), reason="real Phase 4 npz absent")
needs_toy = pytest.mark.skipif(not TOY_NPZ.exists(), reason="phase 1 toy npz absent")
needs_phase4_run = pytest.mark.skipif(
    not (PHASE4_RUN_DIR / "method1_sweep.npz").exists(),
    reason="persisted phase_4_real run artifacts absent",
)
slow_guard = pytest.mark.skipif(
    not os.environ.get("PHASE4_SLOW"),
    reason="slow convergence guard; set PHASE4_SLOW=1 to run",
)


@pytest.fixture(scope="module")
def real_data():
    if not REAL_NPZ.exists():
        pytest.skip("real Phase 4 npz absent")
    return load_real_marginal(REAL_NPZ)


@pytest.fixture(scope="module")
def real_edges(real_data):
    return knn_sphere_edges(real_data["latlons"], k=8)


@pytest.fixture(scope="module")
def toy_arrays():
    if not TOY_NPZ.exists():
        pytest.skip("phase 1 toy npz absent")
    data = load_sampler_arrays(TOY_NPZ)
    return data["pi"], data["mu"], data["sigma"]


# --- S1: loader -------------------------------------------------------------


def _write_real_npz(path, **overrides):
    n, k = 6, 4
    rng = np.random.default_rng(0)
    pi = rng.dirichlet(np.ones(k), size=n).astype(np.float32)
    arrays = {
        "pi": pi,
        "mu_2t": rng.normal(size=(n, k)).astype(np.float32),
        "sigma_2t": (0.1 + rng.random((n, k))).astype(np.float32),
        "latlons": rng.uniform(-80, 80, size=(n, 2)).astype(np.float32),
    }
    arrays.update(overrides)
    np.savez(path, **arrays)
    return path


def test_load_real_marginal_casts_and_renames(tmp_path):
    path = _write_real_npz(tmp_path / "ok.npz")
    out = load_real_marginal(path)
    assert sorted(out) == ["latlons", "mu", "pi", "sigma"]
    for arr in out.values():
        assert arr.dtype == np.float64
    assert out["pi"].shape == (6, 4)
    assert out["latlons"].shape == (6, 2)


def test_load_real_marginal_missing_key_raises(tmp_path):
    n, k = 6, 4
    rng = np.random.default_rng(0)
    np.savez(
        tmp_path / "missing.npz",
        pi=rng.dirichlet(np.ones(k), size=n),
        latlons=rng.uniform(-80, 80, size=(n, 2)),
    )
    with pytest.raises(KeyError):
        load_real_marginal(tmp_path / "missing.npz")


def test_load_real_marginal_validates_values(tmp_path):
    bad_pi = _write_real_npz(tmp_path / "badpi.npz")
    data = dict(np.load(bad_pi))
    data["pi"][0, 0] += 0.5  # rows no longer sum to 1
    np.savez(tmp_path / "badpi2.npz", **data)
    with pytest.raises(ValueError):
        load_real_marginal(tmp_path / "badpi2.npz")

    data = dict(np.load(_write_real_npz(tmp_path / "badsig.npz")))
    data["sigma_2t"][0, 0] = 0.0
    np.savez(tmp_path / "badsig2.npz", **data)
    with pytest.raises(ValueError):
        load_real_marginal(tmp_path / "badsig2.npz")

    data = dict(np.load(_write_real_npz(tmp_path / "badfin.npz")))
    data["mu_2t"][0, 0] = np.nan
    np.savez(tmp_path / "badfin2.npz", **data)
    with pytest.raises(ValueError):
        load_real_marginal(tmp_path / "badfin2.npz")


# --- S1: log-space NLL (S4.8) ------------------------------------------------


@needs_toy
def test_gmm_log_pdf_matches_linear_space_on_toy(toy_arrays):
    pi, mu, sigma = toy_arrays
    field, _ = mode_field(pi, mu, sigma)
    linear = -np.log(mixture_pdf(field, pi, mu, sigma))
    log_space = -gmm_log_pdf(field, pi, mu, sigma)
    assert np.allclose(linear, log_space, atol=1e-10)
    assert np.allclose(gmm_nll_per_cell(field, pi, mu, sigma), linear, atol=1e-10)


def test_gmm_log_pdf_finite_at_40_sigma_probe():
    pi = np.array([[0.6, 0.4]])
    mu = np.array([[0.0, 0.5]])
    sigma = np.array([[1.0, 1.0]])
    x = np.array([40.0])
    assert mixture_pdf(x, pi, mu, sigma)[0] == 0.0  # float64 true zero
    val = gmm_log_pdf(x, pi, mu, sigma)[0]
    assert np.isfinite(val)
    assert val < -700.0


def test_gmm_log_pdf_handles_dead_components():
    pi = np.array([[1.0, 0.0]])
    mu = np.array([[0.0, 5.0]])
    sigma = np.array([[1.0, 1.0]])
    expected = -0.5 * np.log(2 * np.pi)
    assert np.isclose(gmm_log_pdf(np.array([0.0]), pi, mu, sigma)[0], expected)


# --- S1: vectorised iid draw -------------------------------------------------


def test_vectorised_iid_draw_matches_weights():
    n = 20000
    pi = np.tile([0.2, 0.8], (n, 1))
    mu = np.tile([-3.0, 3.0], (n, 1))
    sigma = np.full((n, 2), 0.1)
    sample, comp = sample_iid_gmm(pi, mu, sigma, np.random.default_rng(0))
    assert sample.shape == (n,)
    assert comp.shape == (n,)
    assert abs(np.mean(comp == 1) - 0.8) < 0.01
    # values follow their chosen component
    assert np.all(np.abs(sample[comp == 0] + 3.0) < 1.0)


def test_vectorised_iid_draw_keeps_2d_shape(toy_arrays):
    pi, mu, sigma = toy_arrays
    sample, comp = sample_iid_gmm(pi, mu, sigma, np.random.default_rng(0))
    assert sample.shape == pi.shape[:2]
    assert comp.shape == pi.shape[:2]
    assert np.all((comp >= 0) & (comp < pi.shape[-1]))


# --- S2: sphere graph (S4.4, S4.5) -------------------------------------------


@needs_real
def test_knn_sphere_edge_count_matches_audit(real_edges):
    assert real_edges.shape[1] == 2
    assert len(real_edges) == 162406
    assert np.all(real_edges[:, 0] < real_edges[:, 1])


@needs_real
def test_date_line_wrap_edge_exists_near_equator(real_data, real_edges):
    latlons = real_data["latlons"]
    dlon = np.abs(latlons[real_edges[:, 0], 1] - latlons[real_edges[:, 1], 1])
    wrap = real_edges[dlon > 350.0]
    assert len(wrap) > 0
    near_equator = np.abs(latlons[wrap[:, 0], 0]) < 5.0
    assert np.any(near_equator)


@needs_real
def test_pole_rings_have_full_degree(real_data, real_edges):
    latlons = real_data["latlons"]
    assert np.all(real_edges[:, 0] != real_edges[:, 1])  # no self-edges
    degree = np.zeros(latlons.shape[0], dtype=int)
    np.add.at(degree, real_edges[:, 0], 1)
    np.add.at(degree, real_edges[:, 1], 1)
    for pole_lat in (latlons[:, 0].max(), latlons[:, 0].min()):
        ring = np.flatnonzero(latlons[:, 0] == pole_lat)
        assert len(ring) == 20  # O96 polar ring
        assert np.all(degree[ring] >= 8)


@needs_real
def test_edge_arc_km_matches_audit_stats(real_data, real_edges):
    arc = edge_arc_km(real_data["latlons"], real_edges)
    assert abs(arc.min() - 24.9) < 1.0
    assert abs(np.median(arc) - 131.1) < 2.0
    assert abs(arc.max() - 212.7) < 2.0


def test_sparse_laplacian_matches_dense_quadratic_form():
    edges = grid_edges_8(8, 8)
    lap_sparse = sparse_laplacian(64, edges)
    from sampler_research.graph import graph_laplacian

    lap_dense = graph_laplacian(8, 8)
    rng = np.random.default_rng(0)
    x = rng.normal(size=64)
    assert np.isclose(roughness_sum(x, lap_sparse), roughness_sum(x, lap_dense))
    assert np.allclose(lap_sparse.todense(), lap_dense)
    blurred = laplacian_blur(x, lap_sparse, step=0.1, n_iters=5)
    assert np.allclose(blurred, laplacian_blur(x, lap_dense, step=0.1, n_iters=5))


# --- S4.1 near-one-hot cell / S4.2 duplicates / S4.3 heteroscedastic ----------


def test_near_one_hot_cell():
    pi = np.array([[0.99, 0.0070, 0.0029942, 5.8e-6]])
    pi = pi / pi.sum()
    mu = np.array([[0.0, 1.0, -1.0, 2.0]])
    sigma = np.full((1, 4), 0.15)
    ext = extract_gmm_modes(pi, mu, sigma)
    assert ext.mode_counts[0] >= 1
    assert np.all(np.isfinite(ext.mode_unary[0][ext.valid_mask[0]]))
    assert np.all(np.isfinite(-np.log(pi)))
    sample, comp = sample_iid_gmm(pi, mu, sigma, np.random.default_rng(0))
    assert np.isfinite(sample[0])


def test_duplicate_components_merge_and_padding_never_selected():
    n = 2
    pi = np.tile([0.4, 0.4, 0.1, 0.1], (n, 1))
    mu = np.tile([0.0, 0.01, 1.0, 1.005], (n, 1))
    sigma = np.full((n, 4), 0.5)
    ext = extract_gmm_modes(pi, mu, sigma)
    assert ext.height is None and ext.width is None
    assert np.all(ext.mode_counts < 4)  # duplicates merged
    edges = np.array([[0, 1]], dtype=np.int64)
    res = solve_value_mrf(pi, mu, sigma, ext, 0.5, n_restarts=3,
                          edges=edges, rng=np.random.default_rng(0))
    flat_assign = np.asarray(res.assignment).reshape(-1)
    assert np.all(flat_assign < ext.mode_counts)  # padded slots never chosen


def _mixture_density_derivative(v, pi, mu, sigma):
    comp = pi * np.exp(-0.5 * ((v - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    return float(np.sum(comp * (mu - v) / sigma**2))


def test_heteroscedastic_fixed_point_is_stationary_and_old_update_is_not():
    # Overlapping heteroscedastic pair: the old shared-sigma update stalls
    # visibly off-mode here (fully separated cells would not show the bug).
    pi = np.array([[0.6, 0.4]])
    mu = np.array([[0.0, 0.6]])
    sigma = np.array([[0.05, 0.6]])

    ext = extract_gmm_modes(pi, mu, sigma)
    for mode in ext.mode_values[0][ext.valid_mask[0]]:
        assert abs(_mixture_density_derivative(mode, pi[0], mu[0], sigma[0])) < 1e-6

    # Old (shared-sigma) update from the narrow component's mean: its fixed
    # point v = sum_k r_k mu_k sits visibly off-mode (|p'| ~ 36 here) because
    # it ignores the per-component precisions.
    v = 0.0
    for _ in range(500):
        comp = pi[0] * np.exp(-0.5 * ((v - mu[0]) / sigma[0]) ** 2) / (
            sigma[0] * np.sqrt(2 * np.pi)
        )
        resp = comp / comp.sum()
        v_new = float(np.sum(resp * mu[0]))
        if abs(v_new - v) < 1e-12:
            v = v_new
            break
        v = v_new
    assert abs(_mixture_density_derivative(v, pi[0], mu[0], sigma[0])) > 0.1


@needs_toy
def test_heteroscedastic_update_reduces_to_shared_sigma_on_toy(toy_arrays):
    # Component-shared sigma within each cell: new update must reproduce the
    # old modes bit-near (the added multiply/divide only moves round-off).
    pi, mu, sigma = toy_arrays
    ext = extract_gmm_modes(pi, mu, sigma)
    stage_c = REPO_ROOT / "outputs" / "runs" / "stage_c_method4_mrf" / (
        "phase_1_homoscedastic_method4_mrf.npz"
    )
    if not stage_c.exists():
        pytest.skip("persisted Stage C artifact absent")
    committed = np.load(stage_c)
    assert np.allclose(ext.mode_values, committed["mode_values"], atol=1e-8)
    assert np.array_equal(ext.mode_counts, committed["mode_counts"])


# --- S4.6: lambda -> 0 refined-mode floor (flat path) -------------------------


@needs_toy
def test_lambda_zero_flat_refines_the_mode_field(toy_arrays):
    pi, mu, sigma = toy_arrays
    k = pi.shape[-1]
    pi_f, mu_f, sigma_f = (a.reshape(-1, k) for a in (pi, mu, sigma))
    edges = grid_edges_8(*pi.shape[:2])
    lap = sparse_laplacian(pi_f.shape[0], edges)

    warm, _ = mode_field(pi_f, mu_f, sigma_f)
    res = minimise_at_lambda(
        pi_f, mu_f, sigma_f, 0.0, n_restarts=1, n_steps=400,
        rng=np.random.default_rng(0), edges=edges, laplacian=lap,
    )
    assert res.field.shape == warm.shape
    warm_nll = gmm_nll_per_cell(warm, pi_f, mu_f, sigma_f)
    opt_nll = gmm_nll_per_cell(res.field, pi_f, mu_f, sigma_f)
    assert np.all(opt_nll <= warm_nll + 1e-6)  # refined-mode floor, not equality
    grad = nll_gradient(res.field, pi_f, mu_f, sigma_f)
    assert np.max(np.abs(grad)) < 1e-3  # per-cell stationarity
    assert res.spectral_hf_ratio is None  # flat path gates planar-FFT diagnostics


# --- S4.7: toy regressions through the refactored paths ----------------------


@needs_toy
def test_stage_a_fields_rescore_identically_2d_and_flat(toy_arrays):
    npz = STAGE_A_DIR / "phase_1_homoscedastic_baselines.npz"
    csv_path = STAGE_A_DIR / "stage_a_scores.csv"
    if not (npz.exists() and csv_path.exists()):
        pytest.skip("persisted Stage A artifacts absent")
    pi, mu, sigma = toy_arrays
    k = pi.shape[-1]
    fields = np.load(npz)
    with open(csv_path) as fh:
        rows = {row["baseline"]: row for row in csv.DictReader(fh)}

    edges = grid_edges_8(*pi.shape[:2])
    for name in fields.files:
        committed = rows[name]
        s2d = score_field(fields[name], pi, mu, sigma)
        assert np.isclose(s2d["nll_over_n"], float(committed["nll_over_n"]), atol=1e-9)
        assert np.isclose(s2d["r_tilde"], float(committed["r_tilde"]), atol=1e-9)
        flat = score_field(
            fields[name].reshape(-1), pi.reshape(-1, k), mu.reshape(-1, k),
            sigma.reshape(-1, k), edges=edges,
        )
        assert np.isclose(flat["nll_over_n"], s2d["nll_over_n"], atol=1e-12)
        assert np.isclose(flat["r_tilde"], s2d["r_tilde"], atol=1e-12)
        assert np.isnan(flat["spectral_hf_ratio"])  # flat spectral placeholders


@needs_toy
def test_stage_b_lambda_point_reproduces_committed_field(toy_arrays):
    if not STAGE_B_NPZ.exists():
        pytest.skip("persisted Stage B artifact absent")
    committed = np.load(STAGE_B_NPZ)
    lambdas = committed["lambdas"]
    idx = int(np.flatnonzero(np.isclose(lambdas, 0.2))[0])
    pi, mu, sigma = toy_arrays
    res = minimise_at_lambda(
        pi, mu, sigma, 0.2, n_restarts=4, n_steps=4000, lr=0.05,
        restart_scale=1.0, rng=np.random.default_rng(idx),
    )
    assert np.max(np.abs(res.field - committed["fields"][idx])) < 1e-9


@needs_toy
def test_method1_flat_path_agrees_with_2d_path(toy_arrays):
    pi, mu, sigma = toy_arrays
    k = pi.shape[-1]
    res_2d = minimise_at_lambda(
        pi, mu, sigma, 0.2, n_restarts=2, n_steps=200,
        rng=np.random.default_rng(0),
    )
    edges = grid_edges_8(*pi.shape[:2])
    lap = sparse_laplacian(pi.shape[0] * pi.shape[1], edges)
    res_flat = minimise_at_lambda(
        pi.reshape(-1, k), mu.reshape(-1, k), sigma.reshape(-1, k), 0.2,
        n_restarts=2, n_steps=200, rng=np.random.default_rng(0),
        edges=edges, laplacian=lap,
    )
    assert np.max(np.abs(res_flat.field - res_2d.field.reshape(-1))) < 1e-4
    assert np.isclose(res_flat.nll_over_n, res_2d.nll_over_n, atol=1e-6)
    assert res_flat.restart_nll_over_n.shape == (2,)


@needs_toy
def test_flat_inputs_require_edges_and_laplacian(toy_arrays):
    pi, mu, sigma = toy_arrays
    k = pi.shape[-1]
    pi_f, mu_f, sigma_f = (a.reshape(-1, k) for a in (pi, mu, sigma))
    with pytest.raises(ValueError):
        minimise_at_lambda(pi_f, mu_f, sigma_f, 0.1)
    with pytest.raises(ValueError):
        smoothed_map_baseline(pi_f, mu_f, sigma_f)
    with pytest.raises(ValueError):
        score_field(mode_field(pi_f, mu_f, sigma_f)[0], pi_f, mu_f, sigma_f)


@needs_toy
def test_flat_smoothed_map_baseline_smooths(toy_arrays):
    pi, mu, sigma = toy_arrays
    k = pi.shape[-1]
    pi_f, mu_f, sigma_f = (a.reshape(-1, k) for a in (pi, mu, sigma))
    edges = grid_edges_8(*pi.shape[:2])
    lap = sparse_laplacian(pi_f.shape[0], edges)
    blurred = smoothed_map_baseline(pi_f, mu_f, sigma_f, laplacian=lap)
    raw, _ = mode_field(pi_f, mu_f, sigma_f)
    assert blurred.shape == raw.shape
    assert np.all(np.isfinite(blurred))
    r_blur, _ = scale_free_roughness(blurred, edges)
    r_raw, _ = scale_free_roughness(raw, edges)
    assert r_blur < r_raw
    # 2D default path unchanged
    blurred_2d = smoothed_map_baseline(pi, mu, sigma)
    assert np.allclose(blurred_2d.reshape(-1), blurred)


# --- flat delta-NLL -----------------------------------------------------------


@needs_toy
def test_delta_nll_flat_matches_2d(toy_arrays):
    pi, mu, sigma = toy_arrays
    k = pi.shape[-1]
    ext = extract_gmm_modes(pi, mu, sigma)
    field, _ = mode_field(pi, mu, sigma)
    d2 = delta_nll_to_best_mode(field, pi, mu, sigma, ext.mode_values, ext.valid_mask)
    d1 = delta_nll_to_best_mode(
        field.reshape(-1), pi.reshape(-1, k), mu.reshape(-1, k), sigma.reshape(-1, k),
        ext.mode_values, ext.valid_mask,
    )
    assert d1["per_cell"].shape == (field.size,)
    assert d2["per_cell"].shape == field.shape
    assert np.allclose(d1["per_cell"], d2["per_cell"].reshape(-1), atol=1e-10)
    assert np.isclose(d1["mean"], d2["mean"])


# --- evaluation protocol ------------------------------------------------------


@needs_real
def test_real_multimodality_censuses_match_audit(real_data):
    pi, mu, sigma = real_data["pi"], real_data["mu"], real_data["sigma"]
    assert int(practically_bimodal_mask(pi, mu, sigma).sum()) == 604
    assert int(practically_bimodal_mask(pi, mu, sigma, min_separation=1.0).sum()) == 5576
    frac = float(np.mean(effective_k(pi) > 1.5))
    assert abs(frac - 0.300) < 0.002


@needs_real
def test_stratum_masks_partition(real_data):
    masks = stratum_masks(
        real_data["pi"], real_data["mu"], real_data["sigma"], real_data["latlons"]
    )
    n = real_data["pi"].shape[0]
    assert masks["global"].sum() == n
    lat_total = masks["lat_polar"].sum() + masks["lat_mid"].sum() + masks["lat_tropics"].sum()
    assert lat_total == n
    assert masks["bimodal"].sum() + masks["unimodal"].sum() == n


def test_stratified_scores_columns_and_values():
    field = np.array([0.0, 1.0, 0.0, 1.0])
    nll = np.array([1.0, 2.0, 3.0, 4.0])
    delta = np.array([0.0, 0.2, 0.0, 0.6])
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    masks = {"global": np.ones(4, dtype=bool), "half": np.array([True, True, False, False])}
    row = stratified_scores(field, nll, edges, masks, delta)
    assert row["n_cells"] == 4
    assert np.isclose(row["nll_over_n"], 2.5)
    assert np.isclose(row["dnll_frac_gt_0p125"], 0.5)
    assert np.isclose(row["dnll_frac_gt_0p5"], 0.25)
    assert row["n_cells__half"] == 2
    assert np.isclose(row["nll_over_n__half"], 1.5)
    assert np.isclose(row["s_edge__half"], 1.0)  # only edge (0,1) survives


@needs_real
def test_wrap_seam_guard_flags_longitude_bugs(real_data, real_edges):
    latlons = real_data["latlons"]
    smooth = np.sin(np.radians(latlons[:, 1])) * np.cos(np.radians(latlons[:, 0]))
    healthy = wrap_seam_ratio(smooth, latlons, real_edges)
    assert 0.0 <= healthy < 5.0
    buggy = wrap_seam_ratio(latlons[:, 1], latlons, real_edges)  # raw lon field
    assert buggy > 100.0


# --- select_lambda_star clause unit tests -------------------------------------


def test_lambda_star_plain_bracket_interpolates_in_log_lambda():
    res = select_lambda_star([1.0, 100.0], [0.1, 0.01], [False, False], 0.055)
    assert res.bracketed
    assert np.isclose(res.lambda_star, 10.0)
    assert res.clauses == []


def test_lambda_star_excludes_lambda_zero_and_collapsed():
    res = select_lambda_star(
        [0.0, 1.0, 10.0, 100.0],
        [0.5, 0.1, 0.05, 1e-9],
        [False, False, False, True],
        0.07,
    )
    assert res.bracketed
    assert "a_collapsed_rows_excluded" in res.clauses
    assert 1.0 < res.lambda_star < 10.0


def test_lambda_star_smallest_crossing_wins():
    res = select_lambda_star(
        [1.0, 10.0, 100.0, 1000.0],
        [0.1, 0.02, 0.06, 0.01],
        [False] * 4,
        0.05,
    )
    assert res.bracketed
    assert "c_smallest_crossing" in res.clauses
    assert res.lambda_star < 10.0


def test_lambda_star_unbracketed_directions():
    up = select_lambda_star([1.0, 10.0], [0.5, 0.2], [False, False], 0.1)
    assert not up.bracketed
    assert up.extend_direction == "up"
    assert up.lambda_star == 10.0
    down = select_lambda_star([1.0, 10.0], [0.5, 0.2], [False, False], 0.9)
    assert not down.bracketed
    assert down.extend_direction == "down"
    assert down.lambda_star == 1.0


def test_lambda_star_no_valid_rows_raises():
    with pytest.raises(ValueError):
        select_lambda_star([0.0, 1.0], [0.5, 0.1], [False, True], 0.3)


# --- robustness canaries (11 Jun post-implementation review, D-tests) ----------


def test_beta_scale_canary_solver_switches_when_gap_is_small():
    """Protects: "Method 4's pinned beta sweep is a data property (near-one-hot
    unary gaps), not a dead solver". On a 2-cell fixture whose second cell has
    a small (~0.4 nat) unary gap, beta=10 must move at least one cell off its
    unary-best mode and beta=0.001 must move none.
    """

    pi = np.array([[0.9, 0.1], [0.6, 0.4]])
    mu = np.array([[0.0, 2.0], [2.0, 0.0]])
    sigma = np.full((2, 2), 0.2)
    ext = extract_gmm_modes(pi, mu, sigma)
    assert np.all(ext.mode_counts == 2)  # well-separated 10-sigma pairs
    unary_best = np.argmin(np.where(ext.valid_mask, ext.mode_unary, np.inf), axis=1)
    edges = np.array([[0, 1]], dtype=np.int64)

    moved = {}
    for beta in (10.0, 0.001):
        res = solve_value_mrf(pi, mu, sigma, ext, beta, n_restarts=2,
                              edges=edges, rng=np.random.default_rng(0), max_sweeps=30)
        moved[beta] = int(np.sum(np.asarray(res.assignment) != unary_best))
    assert moved[10.0] > 0  # pairwise term can switch a small-gap mode
    assert moved[0.001] == 0  # unary pins the assignment at tiny beta


@needs_phase4_run
def test_lambda_star_replay_regression():
    """Protects the lambda* = 93.74 headline against silent artifact drift:
    re-deriving the target from anchors.npz and replaying select_lambda_star on
    the persisted sweep must return 93.74 +- 0.5, bracketed, with no fallback
    clauses (matching lambda_star.json).
    """

    with np.load(PHASE4_RUN_DIR / "masks.npz") as f:
        edges = f["edges"]
    with np.load(PHASE4_RUN_DIR / "anchors.npz") as f:
        target_field = f["smoothed_map_n10"]
    target, collapsed = scale_free_roughness(target_field, edges)
    assert not collapsed
    star = json.loads((PHASE4_RUN_DIR / "lambda_star.json").read_text())
    assert np.isclose(target, star["target_r_tilde"], atol=1e-9)

    sweep = np.load(PHASE4_RUN_DIR / "method1_sweep.npz")
    sel = select_lambda_star(
        sweep["lambdas"], sweep["r_tilde"], sweep["variance_collapsed"],
        target, valid=sweep["warm_start_sane"],
    )
    assert abs(sel.lambda_star - 93.74) < 0.5
    assert sel.bracketed
    assert sel.clauses == []
    assert np.isclose(sel.lambda_star, star["lambda_star"], atol=1e-9)


@needs_real
def test_blur_stability_canary_real_graph(real_data, real_edges):
    """Protects the smoothed-MAP baseline that anchors lambda*: the blur
    iteration x <- x - step*Lx is stable only for step < 2/lambda_max(L), and
    on the real union-kNN graph step 0.1 sits at ~74% of that bound
    (lambda_max ~ 14.8) -- any graph change that pushes step*lambda_max past 2
    silently diverges the target.
    """

    from scipy.sparse.linalg import eigsh

    lap = sparse_laplacian(real_data["pi"].shape[0], real_edges)
    lmax = float(eigsh(lap, k=1, which="LM", return_eigenvectors=False)[0])
    assert 8.0 < lmax < 20.0  # sanity: k=8 union graph, max degree ~11
    assert 0.1 * lmax < 2.0  # stability bound for the production step


@needs_real
@slow_guard
def test_lambda_tail_convergence_guard(real_data, real_edges):
    """Protects "no variance collapse anywhere" and the Pareto tail: at the
    largest sweep lambda (1000), doubling the Adam steps from the production
    400 must move NLL/N by < 0.02 and R-tilde by < 2% relative, with no
    collapse flag -- i.e. the tail rows are converged, not under-optimised.
    Warm-start restart only (deterministic); ~5 s, gated behind PHASE4_SLOW=1.
    """

    pi, mu, sigma = real_data["pi"], real_data["mu"], real_data["sigma"]
    lap = sparse_laplacian(pi.shape[0], real_edges)
    results = {}
    for n_steps in (400, 800):
        results[n_steps] = minimise_at_lambda(
            pi, mu, sigma, 1000.0, n_restarts=1, n_steps=n_steps, lr=0.05,
            restart_scale=0.15, rng=np.random.default_rng(0),
            laplacian=lap, edges=real_edges,
        )
    base, doubled = results[400], results[800]
    assert not doubled.variance_collapsed
    assert abs(doubled.nll_over_n - base.nll_over_n) < 0.02
    assert abs(doubled.r_tilde - base.r_tilde) / base.r_tilde < 0.02
