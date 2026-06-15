"""Report-results emitter: pure builders over fixtures (no recompute, no I/O)."""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "emit_report_results.py"
FC_CONVERGED_RUN = REPO_ROOT / "outputs" / "runs" / "phase_4_fc48_14ep_step8"
needs_fc_converged = pytest.mark.skipif(
    not (FC_CONVERGED_RUN / "delta_per_cell.npz").exists(),
    reason="converged +48h forecast run artifact absent",
)


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
    # C1: across-init range footnote is macro-driven (not a hand-typed number).
    assert r"\forecastLambdaStarConvergedLo" in lam_tbl
    assert r"\forecastLambdaStarConvergedHi" in lam_tbl
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


def _ci(point, lo, hi):
    return {"point": point, "lo": lo, "hi": hi, "mean": point,
            "se": 0.01, "ci": 0.95, "n_boot": 2000}


def test_build_bootstrap_macros_emits_points_and_ci_bounds():
    m = _load()
    recon = {
        "global": {"m1": _ci(0.169, 0.165, 0.173), "blur": _ci(0.466, 0.461, 0.471),
                   "gap": _ci(-0.297, -0.303, -0.291), "n": 40320},
        "bimodal": {"gap": _ci(-0.025, -0.060, 0.010), "n": 3000},
    }
    fc = {"bimodal": {"m1": _ci(0.198, 0.180, 0.216), "blur": _ci(0.294, 0.275, 0.313),
                      "gap": _ci(-0.096, -0.120, -0.072), "n": 1500}}
    macros = m.build_bootstrap_macros(recon, fc)
    assert macros["reconFracMethod"] == "0.169"
    assert macros["reconFracGapLo"] == "-0.303" and macros["reconFracGapHi"] == "-0.291"
    assert macros["fcBimodalFracBlur"] == "0.294"
    # recon-bimodal gap CI straddles 0 (thin stratum) -> honest non-separation
    assert macros["reconBimodalFracGapLo"] == "-0.060"
    assert macros["reconBimodalFracGapHi"] == "0.010"
    assert macros["bootstrapNDraws"] == "2000" and macros["bootstrapCIPct"] == "95"


def test_build_bootstrap_macros_placeholders_when_absent():
    m = _load()
    macros = m.build_bootstrap_macros({}, {})
    assert macros["reconFracMethod"] == m.PLACEHOLDER
    assert macros["fcBimodalFracGapHi"] == m.PLACEHOLDER
    # the descriptive constants are always present
    assert macros["bootstrapNDraws"] == "2000"


@needs_fc_converged
def test_forecast_bootstrap_pins_converged_bimodal_gap():
    """A2 guard: the +48h forecast smear-fraction comparison must be computed on
    the CONVERGED 14-epoch run, never the 6-epoch context column. Replays the
    persisted delta_per_cell.npz + masks.npz through `_bootstrap_frac_ci` at the
    production seed and pins the bimodal gap + spatial-bootstrap CI to that
    artifact. The 6ep run returns gap ~ -0.096; the converged run returns -0.103,
    so the tight tolerance below fails if the call site silently reverts (the
    golden round-trip F2 then also fails, since the committed -0.103 no longer
    reproduces)."""
    m = _load()
    out = m._bootstrap_frac_ci(FC_CONVERGED_RUN, ("global", "bimodal"))
    gap = out["bimodal"]["gap"]
    assert gap["point"] == pytest.approx(-0.103, abs=2e-3)
    assert gap["lo"] == pytest.approx(-0.117, abs=2e-3)
    assert gap["hi"] == pytest.approx(-0.088, abs=2e-3)
    # method/blur point estimates also lock to the converged artifact.
    assert out["bimodal"]["m1"]["point"] == pytest.approx(0.090, abs=2e-3)
    assert out["bimodal"]["blur"]["point"] == pytest.approx(0.193, abs=2e-3)


# ---------------------------------------------- F1 track-2 across-init spread
SOFT_MEAN = [
    {"run": "14ep", "step": 1, "lead_hours": 6.0, "kind": "init_mean", "n_inits": 3,
     "median_max_pi": 0.942, "median_max_pi_lo": 0.937, "median_max_pi_hi": 0.946,
     "one_hot_fraction": 0.66, "one_hot_fraction_lo": 0.64, "one_hot_fraction_hi": 0.68,
     "median_second_mode": 0.045, "median_second_mode_lo": 0.043, "median_second_mode_hi": 0.047},
    {"run": "14ep", "step": 8, "lead_hours": 48.0, "kind": "init_mean", "n_inits": 3,
     "median_max_pi": 0.769, "median_max_pi_lo": 0.766, "median_max_pi_hi": 0.776,
     "one_hot_fraction": 0.10, "one_hot_fraction_lo": 0.09, "one_hot_fraction_hi": 0.11,
     "median_second_mode": 0.176, "median_second_mode_lo": 0.170, "median_second_mode_hi": 0.180},
]
FAITH_MEAN = [
    {"run": "14ep", "step": 8, "kind": "init_mean", "n_inits": 3,
     "lambda_star": 84.0, "lambda_star_lo": 81.3, "lambda_star_hi": 88.0,
     "iid_delta_crps": -0.00001, "iid_delta_crps_lo": -0.00005, "iid_delta_crps_hi": 0.00003,
     "m1_star_cov90": 0.989, "m1_star_cov90_lo": 0.986, "m1_star_cov90_hi": 0.990,
     "m1_star_pit_ks": 0.309, "m1_star_pit_ks_lo": 0.305, "m1_star_pit_ks_hi": 0.312},
]


def test_build_softening_spread_macros_emits_mean_and_range():
    m = _load()
    macros = m.build_softening_spread_macros(SOFT + SOFT_MEAN)
    assert macros["maxPiFortyEightConvergedMean"] == "0.769"
    assert macros["maxPiFortyEightConvergedLo"] == "0.766"
    assert macros["maxPiFortyEightConvergedHi"] == "0.776"
    assert macros["maxPiSixHourConvergedMean"] == "0.942"
    assert macros["oneHotFortyEightConvergedLo"] == r"9.0\%"
    assert macros["secondModeMassFortyEightConvergedHi"] == "0.180"
    assert macros["softeningNInits"] == "3"


def test_build_faithfulness_spread_macros_emits_headline_range():
    m = _load()
    macros = m.build_faithfulness_spread_macros(FAITH + FAITH_MEAN)
    assert macros["forecastLambdaStarConvergedMean"] == "84.0"
    assert macros["forecastLambdaStarConvergedLo"] == "81.3"
    assert macros["forecastLambdaStarConvergedHi"] == "88.0"
    assert macros["mOneCovNinetyConvergedHi"] == "0.990"
    assert macros["mOnePitKsConvergedLo"] == "0.305"
    assert macros["deltaCrpsConvergedMean"].startswith(("+", "-"))  # signed delta
    assert macros["faithfulnessNInits"] == "3"


def test_tables_ignore_init_mean_rows():
    """Regression: the conv. table column must stay the canonical init-A point
    (single row), never silently flip to the 3-init mean when init_mean rows are
    present in the CSV."""
    m = _load()
    pi_tbl = m.build_pi_softening_table(SOFT + SOFT_MEAN)
    assert "0.766" in pi_tbl          # init A +48h conv. point
    assert "0.769" not in pi_tbl      # the 3-init mean must NOT appear in the table
    faith_tbl = m.build_faithfulness_table(FAITH + FAITH_MEAN, SOFT + SOFT_MEAN)
    assert "0.88" in faith_tbl        # init A cov90 point (0.88), not mean 0.989


def test_spread_macros_placeholder_when_no_aggregate():
    m = _load()
    soft = m.build_softening_spread_macros(SOFT)  # single rows only -> no init_mean
    faith = m.build_faithfulness_spread_macros(FAITH)
    assert soft["maxPiFortyEightConvergedMean"] == m.PLACEHOLDER
    assert soft["softeningNInits"] == m.PLACEHOLDER
    assert faith["forecastLambdaStarConvergedLo"] == m.PLACEHOLDER
    assert faith["faithfulnessNInits"] == m.PLACEHOLDER


def test_build_macros_ignores_init_mean_rows():
    """The init_mean aggregate row for 14ep@8 must NOT displace the single-init
    headline point (init A) in build_macros."""
    m = _load()
    macros = m.build_macros(SOFT + SOFT_MEAN, LAM, FAITH + FAITH_MEAN,
                            {("6ep", 8): None, ("14ep", 8): 0.37},
                            {"lambda_star": 93.74, "beta": None})
    # 0.766 is init A's single-row value; the 3-init mean is 0.769 (must not win)
    assert macros["maxPiFortyEightConverged"] == "0.766"
    assert macros["deltaCrpsConverged"] == "+0.002"  # init A single, not the mean


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


# ------------------------------------------------------ C2 dCRPS extremum macro
def test_build_crps_extremum_macros_max_over_converged_single_and_init():
    """Worst-case |dCRPS| over converged single+init rows, in sci-notation; the
    6ep context column and the init_mean aggregate are excluded."""
    m = _load()
    rows = [
        {"run": "6ep", "kind": "single", "iid_delta_crps": -0.5},      # 6ep -> ignored
        {"run": "14ep", "kind": "single", "iid_delta_crps": 3.0e-5},
        {"run": "14ep", "kind": "init", "iid_delta_crps": -9.5e-5},    # worst |.|
        {"run": "14ep", "kind": "init_mean", "iid_delta_crps": -0.9},  # aggregate -> ignored
    ]
    macros = m.build_crps_extremum_macros(rows)
    assert macros["deltaCrpsMaxAbs"] == r"9.5\times10^{-5}"


def test_build_crps_extremum_macros_placeholder_when_absent():
    m = _load()
    assert m.build_crps_extremum_macros([])["deltaCrpsMaxAbs"] == m.PLACEHOLDER
    # only 6ep / init_mean rows present -> still a placeholder (no eligible row)
    rows = [{"run": "6ep", "kind": "single", "iid_delta_crps": -1.0},
            {"run": "14ep", "kind": "init_mean", "iid_delta_crps": -1.0}]
    assert m.build_crps_extremum_macros(rows)["deltaCrpsMaxAbs"] == m.PLACEHOLDER


def test_fmt_sci_two_sig_figs_and_placeholder():
    m = _load()
    assert m.fmt_sci(9.4556e-5) == r"9.5\times10^{-5}"
    assert m.fmt_sci(-1.2e-3) == r"-1.2\times10^{-3}"
    assert m.fmt_sci(0) == "0"
    assert m.fmt_sci(None) == m.PLACEHOLDER
    assert m.fmt_sci(float("nan")) == m.PLACEHOLDER


# ----------------------------------------------------- F2 golden round-trip
_SOFT_CSV = REPO_ROOT / "outputs" / "runs" / "phase_4_forecast_softening" / "softening_by_lead.csv"
_FAITH_CSV = REPO_ROOT / "outputs" / "runs" / "phase_4_forecast_faithfulness" / "faithfulness_by_lead.csv"
_COMMITTED = REPO_ROOT / "report" / "construction"
needs_emit_artifacts = pytest.mark.skipif(
    not (_SOFT_CSV.exists() and _FAITH_CSV.exists()
         and (_COMMITTED / "macros-results.tex").exists()),
    reason="emit artifacts or committed report dir absent",
)


@needs_emit_artifacts
def test_emit_golden_roundtrip_matches_committed(tmp_path):
    """F2: emit into a tmp report-dir from the committed artifacts and assert the
    output is byte-identical to the in-repo macros + tables. Locks the 'no Ch 5
    number typed by hand' guarantee -- a hand-edit to macros-results.tex, or any
    drift between the emit source and its committed output, fails here."""
    m = _load()
    out = tmp_path / "construction"
    out.mkdir(parents=True, exist_ok=True)  # main() writes macros before mkdir
    saved = sys.argv
    sys.argv = ["emit_report_results.py", "--report-dir", str(out)]
    try:
        m.main()
    finally:
        sys.argv = saved
    assert (out / "macros-results.tex").read_text() == \
        (_COMMITTED / "macros-results.tex").read_text()
    for name in ("pi_softening.tex", "lambda_calibration.tex", "faithfulness.tex"):
        assert (out / "tables" / name).read_text() == \
            (_COMMITTED / "tables" / name).read_text(), name
