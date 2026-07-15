"""Tests for the separate variance-normalised +48 h optimisation experiment."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import REPO_ROOT
from sampler_research.baselines import gmm_nll_over_n
from sampler_research.graph import sparse_laplacian

SCRIPT = REPO_ROOT / "scripts" / "run_variance_normalised_map.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_variance_normalised_map", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _problem():
    edges = np.asarray([[0, 1], [1, 2], [2, 3], [0, 3]], dtype=np.int64)
    laplacian = sparse_laplacian(4, edges)
    pi = np.asarray(
        [[0.65, 0.35], [0.45, 0.55], [0.7, 0.3], [0.4, 0.6]], dtype=float
    )
    mu = np.asarray(
        [[-0.8, 0.6], [-0.5, 0.9], [-0.2, 1.1], [0.1, 1.4]], dtype=float
    )
    sigma = np.full_like(mu, 0.7)
    field = np.asarray([-0.35, 0.15, 0.7, 1.2], dtype=float)
    return pi, mu, sigma, field, edges, laplacian


def test_objective_gradient_matches_finite_difference():
    experiment = _load_script()
    pi, mu, sigma, field, edges, laplacian = _problem()
    lam = 0.37

    analytic = experiment.variance_normalised_gradient(
        field, pi, mu, sigma, lam, laplacian, len(edges)
    )
    numeric = np.empty_like(field)
    eps = 1e-6
    for idx in range(field.size):
        plus = field.copy()
        minus = field.copy()
        plus[idx] += eps
        minus[idx] -= eps
        f_plus = experiment.variance_normalised_objective(
            plus, pi, mu, sigma, lam, laplacian, len(edges)
        )
        f_minus = experiment.variance_normalised_objective(
            minus, pi, mu, sigma, lam, laplacian, len(edges)
        )
        numeric[idx] = (f_plus - f_minus) / (2.0 * eps)

    assert np.allclose(analytic, numeric, rtol=2e-5, atol=2e-6)


def test_roughness_term_is_shift_and_scale_invariant():
    experiment = _load_script()
    _, _, _, field, edges, laplacian = _problem()
    base = experiment.variance_normalised_roughness(field, laplacian, len(edges))
    transformed = experiment.variance_normalised_roughness(
        3.7 * field - 8.2, laplacian, len(edges)
    )
    assert transformed == pytest.approx(base, rel=1e-12, abs=1e-12)


def test_lambda_zero_is_exactly_the_likelihood_only_objective():
    experiment = _load_script()
    pi, mu, sigma, field, edges, laplacian = _problem()
    observed = experiment.variance_normalised_objective(
        field, pi, mu, sigma, 0.0, laplacian, len(edges)
    )
    assert observed == gmm_nll_over_n(field, pi, mu, sigma)


def test_variance_collapse_is_rejected_instead_of_hidden_by_a_floor():
    experiment = _load_script()
    pi, mu, sigma, _, edges, laplacian = _problem()
    constant = np.ones(4)
    with pytest.raises(FloatingPointError, match="variance"):
        experiment.variance_normalised_objective(
            constant, pi, mu, sigma, 1.0, laplacian, len(edges)
        )


def test_budget_lambda_uses_log_interpolation():
    experiment = _load_script()
    lam = experiment.interpolate_budget_lambda((10.0, 0.02), (100.0, 0.08), target=0.05)
    assert lam == pytest.approx(np.sqrt(10.0 * 100.0))


def test_figure_fields_keep_figure_5_2_panel_roles(tmp_path):
    experiment = _load_script()
    field = np.arange(4, dtype=float)
    np.savez(
        tmp_path / "anchors.npz",
        mode_map=field,
        iid_seed0=field + 1.0,
        smoothed_map_n10=field + 2.0,
    )
    np.savez(tmp_path / "era5_reference.npz", era5=field + 3.0)

    fields = experiment._figure_fields(
        tmp_path, field + 4.0, 12.5, field + 5.0, 3.25
    )
    names = list(fields)
    assert names[0:2] == ["Per-cell MAP", "Independent draw"]
    assert "lambda^\\star=12.5" in names[2]
    assert names[3] == "Smoothed MAP (n=10)"
    assert "budget" in names[4] and "lambda=3.25" in names[4]
    assert names[5] == "ERA5 reference"


def test_defaults_are_separate_from_the_canonical_48h_outputs():
    experiment = _load_script()
    assert experiment.DATA_NPZ.name == "phase_4_fc48_14ep_step8_2t.npz"
    assert experiment.OUT_DIR != experiment.BASE_RUN_DIR
    assert experiment.FIGURE_PATH.name == "region_maps_variance_normalised.png"
    assert experiment.FIGURE_PATH.name != "region_maps.png"
