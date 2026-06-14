"""Report-results emitter: pure builders over fixtures (no recompute, no I/O)."""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "emit_report_results.py"


def _load():
    spec = importlib.util.spec_from_file_location("emit_report_results", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SOFT = [
    {"run": "6ep", "step": 1, "lead_hours": 6.0, "median_max_pi": 0.953,
     "one_hot_fraction": 0.713, "median_second_mode": 0.042},
    {"run": "6ep", "step": 8, "lead_hours": 48.0, "median_max_pi": 0.803,
     "one_hot_fraction": 0.112, "median_second_mode": 0.151},
    {"run": "14ep", "step": 1, "lead_hours": 6.0, "median_max_pi": 0.944,
     "one_hot_fraction": 0.679, "median_second_mode": 0.044},
    {"run": "14ep", "step": 8, "lead_hours": 48.0, "median_max_pi": 0.766,
     "one_hot_fraction": 0.093, "median_second_mode": 0.178},
]
LAM = {
    ("6ep", 8): {"lambda_star": 120.5, "matched_r_tilde": 0.0031, "bracketed": True},
    ("14ep", 8): {"lambda_star": 81.3, "matched_r_tilde": 0.0030, "bracketed": False},
}
FAITH = [
    {"run": "6ep", "step": 8, "iid_delta_crps": -0.004, "m1_star_cov90": 0.91, "m1_star_pit_ks": 0.05},
    {"run": "14ep", "step": 8, "iid_delta_crps": 0.002, "m1_star_cov90": 0.88, "m1_star_pit_ks": 0.07},
]


def test_build_macros_fills_known_and_placeholders_missing():
    m = _load()
    macros = m.build_macros(SOFT, LAM, FAITH, {("6ep", 8): None, ("14ep", 8): 0.37},
                            {"lambda_star": 93.74, "beta": None})
    assert len(macros) == 20
    assert macros["maxPiSixHourSixEp"] == "0.953"
    assert macros["maxPiFortyEightConverged"] == "0.766"
    assert macros["oneHotSixHourSixEp"] == r"71.3\%"
    assert macros["secondModeMassFortyEightConverged"] == "0.178"
    assert macros["reconstructionLambdaStar"] == "93.7"
    assert macros["forecastLambdaStarSixEp"] == "120.5"
    assert macros["deltaCrpsSixEp"] == "-0.004"
    assert macros["deltaCrpsConverged"] == "+0.002"
    assert macros["forecastBetaConverged"] == "0.37"
    # M4 pinned at 6ep, recon beta absent -> placeholders
    assert macros["forecastBetaSixEp"] == m.PLACEHOLDER
    assert macros["reconstructionBeta"] == m.PLACEHOLDER


def test_build_macros_is_pure():
    m = _load()
    import copy
    soft_copy = copy.deepcopy(SOFT)
    a = m.build_macros(SOFT, LAM, FAITH, {}, {"lambda_star": None, "beta": None})
    b = m.build_macros(SOFT, LAM, FAITH, {}, {"lambda_star": None, "beta": None})
    assert a == b           # deterministic
    assert SOFT == soft_copy  # no mutation of inputs


def test_missing_softening_run_yields_placeholders():
    m = _load()
    macros = m.build_macros([], {}, [], {}, {"lambda_star": None, "beta": None})
    assert all(v == m.PLACEHOLDER for v in macros.values())


def test_render_macros_defines_placeholder_and_all_commands():
    m = _load()
    macros = m.build_macros(SOFT, LAM, FAITH, {}, {"lambda_star": 93.74, "beta": None})
    text = m.render_macros(macros)
    assert r"\newcommand{\ResultPlaceholder}{\textemdash}" in text
    for name in macros:
        assert rf"\newcommand{{\{name}}}" in text


def test_tables_render_rows_and_dagger():
    m = _load()
    pi_tbl = m.build_pi_softening_table(SOFT)
    assert r"\begin{tabular}" in pi_tbl and r"\bottomrule" in pi_tbl
    assert "+6h" in pi_tbl and "+48h" in pi_tbl
    assert "0.953" in pi_tbl and r"71.3\%" in pi_tbl
    lam_tbl = m.build_lambda_table(LAM, SOFT)
    assert "120.5" in lam_tbl
    assert r"$^{\dagger}$" in lam_tbl  # 14ep@8 is unbracketed in the fixture
    faith_tbl = m.build_faithfulness_table(FAITH, SOFT)
    assert "-0.004" in faith_tbl and "+0.002" in faith_tbl


def test_placeholder_never_gets_dagger():
    m = _load()
    lam = {("6ep", 8): {"lambda_star": None, "matched_r_tilde": None, "bracketed": False}}
    tbl = m.build_lambda_table(lam, SOFT)
    assert r"\ResultPlaceholder$^{\dagger}$" not in tbl


def test_m4_operating_beta_reachability_gate(tmp_path):
    """beta is None unless some beta's R~ reaches M1@lambda*'s R~ (else em-dash)."""
    import numpy as np
    m = _load()
    # M4 R~ all ~0.0098, target (m1_star) ~0.0031 -> unreachable -> None
    np.savez(tmp_path / "method4_sweep.npz",
             betas=np.array([0.01, 0.1, 1.0]),
             fields=np.array([[0.0, 1.0], [0.0, 1.1], [0.0, 1.2]]),
             r_tilde=np.array([0.0098, 0.0098, 0.0099]),
             nll_over_n=np.array([-1.70, -1.70, -1.70]))
    (tmp_path / "scores.csv").write_text("name,r_tilde\nm1_star,0.0031\n")
    assert m._m4_operating_beta(tmp_path) is None
    # Now make beta=1.0's R~ match the target within 10% -> returns 1.0
    np.savez(tmp_path / "method4_sweep.npz",
             betas=np.array([0.01, 0.1, 1.0]),
             fields=np.array([[0.0, 1.0], [0.0, 1.1], [0.0, 1.2]]),
             r_tilde=np.array([0.0098, 0.0060, 0.0032]),
             nll_over_n=np.array([-1.70, -1.65, -1.55]))
    assert m._m4_operating_beta(tmp_path) == 1.0


def test_m4_operating_beta_pinned_returns_none(tmp_path):
    import numpy as np
    m = _load()
    np.savez(tmp_path / "method4_sweep.npz",
             betas=np.array([0.01, 1.0]),
             fields=np.array([[0.0, 1.0], [0.0, 1.0]]),  # identical -> pinned
             r_tilde=np.array([0.0032, 0.0032]),
             nll_over_n=np.array([-1.6, -1.6]))
    (tmp_path / "scores.csv").write_text("name,r_tilde\nm1_star,0.0031\n")
    assert m._m4_operating_beta(tmp_path) is None
