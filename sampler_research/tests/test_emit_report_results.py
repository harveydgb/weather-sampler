"""Report-results emitter: pure builders over fixtures (no recompute, no I/O)."""

import importlib.util
import sys

import pytest

from conftest import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "emit_report_results.py"
FC_CONVERGED_RUN = REPO_ROOT / "outputs" / "runs" / "phase_4_fc48_14ep_step8"
needs_fc_converged = pytest.mark.skipif(
    not (FC_CONVERGED_RUN / "delta_per_cell.npz").exists(),
    reason="converged +48h forecast run artifact absent",
)
needs_fc_spectra = pytest.mark.skipif(
    not (FC_CONVERGED_RUN / "spectra.npz").exists(),
    reason="converged +48h spectra artifact absent",
)


def _load():
    spec = importlib.util.spec_from_file_location("emit_report_results", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SOFT = [
    {"run": "6ep", "step": 1, "lead_hours": 6.0, "median_max_pi": 0.953,
     "one_hot_fraction": 0.713, "median_second_mode": 0.042, "bimodal_frac_2sigma": 0.0277},
    {"run": "6ep", "step": 8, "lead_hours": 48.0, "median_max_pi": 0.803,
     "one_hot_fraction": 0.112, "median_second_mode": 0.151, "bimodal_frac_2sigma": 0.0289},
    {"run": "14ep", "step": 1, "lead_hours": 6.0, "median_max_pi": 0.944,
     "one_hot_fraction": 0.679, "median_second_mode": 0.044, "bimodal_frac_2sigma": 0.0252},
    {"run": "14ep", "step": 8, "lead_hours": 48.0, "median_max_pi": 0.766,
     "one_hot_fraction": 0.093, "median_second_mode": 0.178, "bimodal_frac_2sigma": 0.0626},
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
    assert len(macros) == 24
    assert macros["maxPiSixHourSixEp"] == "0.953"
    assert macros["maxPiFortyEightConverged"] == "0.766"
    assert macros["oneHotSixHourSixEp"] == r"71.3\%"
    assert macros["secondModeMassFortyEightConverged"] == "0.178"
    # >2sigma well-separated bimodal fraction: rare/flat at 6ep, grows by +48h conv.
    assert macros["bimodalTwoSigmaFortyEightSixEp"] == r"2.9\%"
    assert macros["bimodalTwoSigmaFortyEightConverged"] == r"6.3\%"
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
    # 29 Jun: the bracketed-audit + across-init range moved OUT of a note under the
    # table INTO the caption (thesis.tex); the table is now a caption-only float.
    assert "footnotesize" not in lam_tbl
    assert r"\forecastLambdaStarConvergedLo" not in lam_tbl
    faith_tbl = m.build_faithfulness_table(FAITH, SOFT)
    # 29 Jun table review: the all-zero do-no-harm ΔCRPS column is dropped to the
    # caption null; the misread-prone "M1 cov." header is renamed "marg.-pos.".
    # DEC-R28 (3 Jul): the PIT-KS columns are dropped too — CRPS and PIT-KS are out
    # of the report; only the central marginal-position coverage is tabulated.
    assert r"marg.-pos." in faith_tbl and "M1" not in faith_tbl
    assert "0.910" in faith_tbl and "0.880" in faith_tbl   # cov90 (6ep, 14ep)
    assert "PIT KS" not in faith_tbl                        # no PIT-KS columns
    assert "0.050" not in faith_tbl and "0.070" not in faith_tbl
    assert "-0.004" not in faith_tbl and "+0.002" not in faith_tbl  # no ΔCRPS column


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
    # levels (shares of cells) render as %; gaps (differences of shares) as pp
    # -- harvey, 4 Jul: propagate the %/pp distinction across this macro family.
    assert macros["reconFracMethod"] == r"16.9\%"
    assert macros["reconFracGapLo"] == r"-30.3\,pp" and macros["reconFracGapHi"] == r"-29.1\,pp"
    assert macros["fcBimodalFracBlur"] == r"29.4\%"
    # recon-bimodal gap CI straddles 0 (thin stratum) -> honest non-separation
    assert macros["reconBimodalFracGapLo"] == r"-6.0\,pp"
    assert macros["reconBimodalFracGapHi"] == r"+1.0\,pp"
    assert macros["bootstrapNDraws"] == "2000" and macros["bootstrapCIPct"] == "95"


def test_build_bootstrap_macros_placeholders_when_absent():
    m = _load()
    macros = m.build_bootstrap_macros({}, {})
    assert macros["reconFracMethod"] == m.PLACEHOLDER
    assert macros["fcBimodalFracGapHi"] == m.PLACEHOLDER
    # the descriptive constants are always present
    assert macros["bootstrapNDraws"] == "2000"


@needs_fc_spectra
def test_build_spectrum_macros_pins_era5_bracket():
    """The two ERA5-bracket scalars the Section 5.5 spectrum prose quotes are read
    from the canonical +48h step-8 spectra.npz, not hand-typed off the figure. Pins
    them to the 29-Jun log values: the worst-case large-scale agreement is 0.054
    decades (JointMAP, every field at or below it) and ERA5 carries ~199x JointMAP's
    fine-scale band power. A pipeline change that moved either would fail here before
    the committed macro could silently drift from the prose."""
    m = _load()
    macros = m.build_spectrum_macros(FC_CONVERGED_RUN.parent)
    assert macros["spectrumDecadeAgreementMax"] == "0.054"
    assert macros["spectrumEraRatioMethod"] == "199"


def test_build_spectrum_macros_placeholders_when_absent(tmp_path):
    m = _load()
    macros = m.build_spectrum_macros(tmp_path)  # no spectra.npz under here
    assert macros["spectrumDecadeAgreementMax"] == m.PLACEHOLDER
    assert macros["spectrumEraRatioMethod"] == m.PLACEHOLDER


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


# ------------------------------------- across-init M1 vs smoothed-MAP gap (#4)
def _write_cmp_scores(run_dir, *, lead_hours, m1_nll, blur_nll, m1_smear, blur_smear,
                      m1_rt=0.0031, blur_rt=0.0032, init_dt=None):
    """Minimal scores.csv (+ optional lambda_star.json) inside one lead run dir.

    DictReader ignores the absent columns, so only the keys the comparison reads
    are written. Mirrors the synthetic-run pattern in the m4 reachability test.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    header = "name,lead_hours,nll_over_n,r_tilde,dnll_frac_gt_0p125\n"
    (run_dir / "scores.csv").write_text(
        header
        + f"smoothed_map_n10,{lead_hours},{blur_nll},{blur_rt},{blur_smear}\n"
        + f"m1_star,{lead_hours},{m1_nll},{m1_rt},{m1_smear}\n"
    )
    if init_dt is not None:
        import json as _json
        (run_dir / "lambda_star.json").write_text(
            _json.dumps({"regime": {"init_datetime": init_dt}}))


def test_comparison_gap_rows_pairs_m1_minus_blur(tmp_path):
    m = _load()
    rd = tmp_path / "phase_4_fc48_v2_init20231101_step8"
    _write_cmp_scores(rd, lead_hours=48.0, m1_nll=-1.647, blur_nll=-1.533,
                      m1_smear=0.132, blur_smear=0.327, init_dt="2023-11-01T00:00")
    rows = m.comparison_gap_rows(tmp_path, "phase_4_fc48_v2_init20231101")
    assert len(rows) == 1
    r = rows[0]
    assert r["step"] == 8 and r["lead_hours"] == 48.0
    assert r["init_datetime"] == "2023-11-01T00:00"
    assert r["gap_nll"] == pytest.approx(-1.647 - (-1.533))   # -0.114, M1 wins
    assert r["gap_smear"] == pytest.approx(0.132 - 0.327)     # -0.195, M1 wins
    # a step missing one of the two rows is skipped (single-init-safe)
    bad = tmp_path / "phase_4_fc48_v2_init20231101_step1"
    bad.mkdir()
    (bad / "scores.csv").write_text("name,lead_hours,nll_over_n,r_tilde,dnll_frac_gt_0p125\n"
                                    "m1_star,6.0,-1.7,0.004,0.10\n")
    assert len(m.comparison_gap_rows(tmp_path, "phase_4_fc48_v2_init20231101")) == 1


def test_aggregate_comparison_gaps_mean_and_range(tmp_path):
    m = _load()
    specs = {
        "init_a": dict(m1_nll=-1.6, blur_nll=-1.5, m1_smear=0.13, blur_smear=0.33),
        "init_b": dict(m1_nll=-1.7, blur_nll=-1.5, m1_smear=0.10, blur_smear=0.30),
        "init_c": dict(m1_nll=-1.5, blur_nll=-1.4, m1_smear=0.20, blur_smear=0.35),
    }
    per_init = []
    for prefix, s in specs.items():
        _write_cmp_scores(tmp_path / f"{prefix}_step8", lead_hours=48.0, init_dt=prefix, **s)
        per_init.append((prefix, m.comparison_gap_rows(tmp_path, prefix)))
    agg = m.aggregate_comparison_gaps(per_init, label="14ep", column="Converged")
    assert len(agg) == 1
    a = agg[0]
    assert a["kind"] == "init_mean" and a["n_inits"] == 3 and a["step"] == 8
    gaps_nll = [s["m1_nll"] - s["blur_nll"] for s in specs.values()]
    assert a["gap_nll"] == pytest.approx(sum(gaps_nll) / 3)
    assert a["gap_nll_lo"] == pytest.approx(min(gaps_nll))
    assert a["gap_nll_hi"] == pytest.approx(max(gaps_nll))
    gaps_smear = [s["m1_smear"] - s["blur_smear"] for s in specs.values()]
    assert a["gap_smear"] == pytest.approx(sum(gaps_smear) / 3)
    assert a["gap_smear_lo"] == pytest.approx(min(gaps_smear))
    assert a["gap_smear_hi"] == pytest.approx(max(gaps_smear))


def test_aggregate_comparison_gaps_needs_two_inits(tmp_path):
    m = _load()
    _write_cmp_scores(tmp_path / "lonely_step8", lead_hours=48.0, m1_nll=-1.6,
                      blur_nll=-1.5, m1_smear=0.13, blur_smear=0.33)
    per_init = [("lonely", m.comparison_gap_rows(tmp_path, "lonely"))]
    assert m.aggregate_comparison_gaps(per_init, label="14ep", column="Converged") == []


GAP_MEAN = [
    {"run": "14ep", "column": "Converged", "kind": "init_mean", "n_inits": 3, "step": 1,
     "lead_hours": 6.0, "gap_nll": -0.010, "gap_nll_lo": -0.020, "gap_nll_hi": -0.005,
     "gap_smear": -0.030, "gap_smear_lo": -0.050, "gap_smear_hi": -0.010},
    {"run": "14ep", "column": "Converged", "kind": "init_mean", "n_inits": 3, "step": 8,
     "lead_hours": 48.0, "gap_nll": -0.114, "gap_nll_lo": -0.130, "gap_nll_hi": -0.100,
     "gap_smear": -0.195, "gap_smear_lo": -0.210, "gap_smear_hi": -0.180},
]


def test_build_comparison_gap_macros_emits_mean_and_range():
    m = _load()
    macros = m.build_comparison_gap_macros(GAP_MEAN)
    assert macros["gapNllFortyEightConvergedMean"] == "-0.114"   # negative = M1 wins
    assert macros["gapNllFortyEightConvergedLo"] == "-0.130"
    assert macros["gapNllFortyEightConvergedHi"] == "-0.100"
    assert macros["gapSmearFortyEightConvergedMean"] == "-0.195"
    assert macros["gapSmearSixHourConvergedMean"] == "-0.030"
    assert macros["comparisonNInits"] == "3"


def test_build_comparison_gap_macros_placeholders_when_absent():
    m = _load()
    macros = m.build_comparison_gap_macros([])
    assert macros["gapNllFortyEightConvergedMean"] == m.PLACEHOLDER
    assert macros["comparisonNInits"] == m.PLACEHOLDER


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


def test_build_crps_extremum_macros_analytic_min_anchor():
    """R-12 anchor: the SMALLEST mean per-cell analytic CRPS over the same
    converged single+init rows the |dCRPS| max scans; 6ep and init_mean rows
    are excluded, and a missing column falls back to the placeholder without
    disturbing the delta extremum."""
    m = _load()
    rows = [
        {"run": "6ep", "kind": "single",
         "iid_delta_crps": -0.5, "iid_analytic_crps": 0.001},   # 6ep -> ignored
        {"run": "14ep", "kind": "single",
         "iid_delta_crps": 3.0e-5, "iid_analytic_crps": 0.0561},
        {"run": "14ep", "kind": "init",
         "iid_delta_crps": -9.5e-5, "iid_analytic_crps": 0.0441},  # min anchor
        {"run": "14ep", "kind": "init_mean",
         "iid_delta_crps": -0.9, "iid_analytic_crps": 0.002},   # aggregate -> ignored
    ]
    macros = m.build_crps_extremum_macros(rows)
    assert macros["crpsAnalyticMinConverged"] == "0.044"
    assert macros["deltaCrpsMaxAbs"] == r"9.5\times10^{-5}"

    # column absent entirely -> placeholder for the anchor only
    bare = [{"run": "14ep", "kind": "single", "iid_delta_crps": 3.0e-5}]
    macros = m.build_crps_extremum_macros(bare)
    assert macros["crpsAnalyticMinConverged"] == m.PLACEHOLDER
    assert macros["deltaCrpsMaxAbs"] == r"3.0\times10^{-5}"


def test_fmt_sci_two_sig_figs_and_placeholder():
    m = _load()
    assert m.fmt_sci(9.4556e-5) == r"9.5\times10^{-5}"
    assert m.fmt_sci(-1.2e-3) == r"-1.2\times10^{-3}"
    assert m.fmt_sci(0) == "0"
    assert m.fmt_sci(None) == m.PLACEHOLDER
    assert m.fmt_sci(float("nan")) == m.PLACEHOLDER


# -------------------------------------------- Ch 4 synthetic-testbed macros (I2)
TOY = [
    {"dataset": "phase_1_homoscedastic", "baseline": "iid",
     "nll_over_n": 1.9771706, "r_tilde": 2.0774265},
    {"dataset": "phase_1_homoscedastic", "baseline": "mode_map",
     "nll_over_n": 1.5510744, "r_tilde": 1.4619713},
    {"dataset": "phase_1_homoscedastic", "baseline": "mixture_mean",
     "nll_over_n": 1.7020445, "r_tilde": 0.6275017},
    {"dataset": "phase_1_homoscedastic", "baseline": "smoothed_map",
     "nll_over_n": 1.7304362, "r_tilde": 0.1840136},
    {"dataset": "phase_1_homoscedastic", "baseline": "a_star",
     "nll_over_n": 1.5913850, "r_tilde": 0.4949174},
]


def test_build_toy_baseline_macros_fills_and_placeholders():
    m = _load()
    macros = m.build_toy_baseline_macros(TOY)
    assert len(macros) == 10  # 5 anchors x {Nll, Rtilde}
    assert macros["toyIidNll"] == "1.977" and macros["toyIidRtilde"] == "2.077"
    assert macros["toySmoothedMapRtilde"] == "0.184"   # over-smoothed skeptic baseline
    assert macros["toyAStarNll"] == "1.591"             # smoothest faithful anchor
    assert macros["toyMixtureMeanRtilde"] == "0.628"
    assert macros["toyModeMapRtilde"] == "1.462"
    # a baseline absent from the rows -> placeholders (never invented)
    partial = m.build_toy_baseline_macros([r for r in TOY if r["baseline"] != "a_star"])
    assert partial["toyAStarNll"] == m.PLACEHOLDER
    assert partial["toyAStarRtilde"] == m.PLACEHOLDER
    # empty -> all placeholders
    assert all(v == m.PLACEHOLDER for v in m.build_toy_baseline_macros([]).values())


def test_build_toy_baseline_macros_is_pure():
    m = _load()
    import copy
    toy_copy = copy.deepcopy(TOY)
    a = m.build_toy_baseline_macros(TOY)
    b = m.build_toy_baseline_macros(TOY)
    assert a == b and TOY == toy_copy


def test_build_toy_baseline_table_renders_rows_and_placeholders():
    m = _load()
    tbl = m.build_toy_baseline_table(TOY)
    assert r"\begin{tabular}{lcc}" in tbl and r"\bottomrule" in tbl
    assert r"Reference field & NLL/$N$ & $\widetilde{R}$" in tbl
    assert "1.977" in tbl and "2.077" in tbl          # iid row
    assert r"Smoothest faithful ($a^\star$) & 1.591 & 0.495" in tbl
    # a missing anchor -> em-dash (placeholder) cells, table still complete
    tbl2 = m.build_toy_baseline_table([r for r in TOY if r["baseline"] != "iid"])
    assert f"Independent draw (iid) & {m.PLACEHOLDER} & {m.PLACEHOLDER}" in tbl2


# ---------------------------------------- Section 4.2 operating-point smear (B5)
_TOY_RUNS = REPO_ROOT / "outputs" / "runs"
_TOY_DATA = REPO_ROOT / "outputs" / "data"
needs_toy_arrays = pytest.mark.skipif(
    not (_TOY_DATA / "phase_1_homoscedastic.npz").exists(),
    reason="toy field artifacts absent (regenerated, not git-tracked)",
)


@needs_toy_arrays
def test_build_toy_smear_macros_pins_figure_operating_points():
    # Drift guard: recompute the off-mode smear fraction at each method's figure
    # operating point and pin it to the value the non-smearing figure shows. If a
    # toy artifact is ever regenerated to different numbers, this fails loudly
    # rather than letting the Section 4.2 prose silently diverge from its figure.
    m = _load()
    macros = m.build_toy_smear_macros(_TOY_RUNS, _TOY_DATA)
    assert set(macros) == set(m.TOY_SMEAR_MACROS)
    # Joint MAP (quadratic) at its R~~0.25 point smears ~14%; exact TV and Mode-MRF
    # hold their non-smearing knees at ~6%.
    assert macros["toyQuadSmearFrac"] == r"14.1\%"
    assert macros["toyQuadSmearRtilde"] == "0.252"
    assert macros["toyTvExactSmearFrac"] == r"6.2\%"
    assert macros["toyTvExactSmearRtilde"] == "0.575"
    assert macros["toyMrfSmearFrac"] == r"6.2\%"
    assert macros["toyMrfSmearRtilde"] == "0.604"


def test_build_toy_smear_macros_placeholders_when_absent(tmp_path):
    # Absent artifacts -> all placeholders, never invented (I2).
    m = _load()
    macros = m.build_toy_smear_macros(tmp_path / "runs", tmp_path / "data")
    assert set(macros) == set(m.TOY_SMEAR_MACROS)
    assert all(v == m.PLACEHOLDER for v in macros.values())


# ----------------------------------------------------- F2 golden round-trip
_SOFT_CSV = REPO_ROOT / "outputs" / "runs" / "phase_4_forecast_softening" / "softening_by_lead.csv"
_FAITH_CSV = REPO_ROOT / "outputs" / "runs" / "phase_4_forecast_faithfulness" / "faithfulness_by_lead.csv"
_TOY_CSV = REPO_ROOT / "outputs" / "runs" / "stage_a_baselines" / "stage_a_scores.csv"
_COMMITTED = REPO_ROOT / "report" / "construction"
needs_emit_artifacts = pytest.mark.skipif(
    not (_SOFT_CSV.exists() and _FAITH_CSV.exists() and _TOY_CSV.exists()
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
    for name in ("pi_softening.tex", "lambda_calibration.tex", "faithfulness.tex",
                 "toy_baselines.tex"):
        assert (out / "tables" / name).read_text() == \
            (_COMMITTED / "tables" / name).read_text(), name


# ------------------------------------------------ clobber guard on empty inputs
def _run_main(argv):
    m = _load()
    saved = sys.argv
    sys.argv = ["emit_report_results.py", *argv]
    try:
        m.main()
    finally:
        sys.argv = saved


def test_emit_refuses_to_clobber_on_empty_inputs(tmp_path):
    """A run with no forecast artifacts (fresh clone) must NOT overwrite the
    committed macros: it exits non-zero and leaves the target untouched."""
    sentinel = tmp_path / "macros-results.tex"
    sentinel.write_text("KEEP-ME")
    with pytest.raises(SystemExit):
        _run_main(["--soft-csv", str(tmp_path / "absent.csv"),
                   "--faith-csv", str(tmp_path / "absent2.csv"),
                   "--runs-dir", str(tmp_path / "empty_runs"),
                   "--ae-run-dir", str(tmp_path / "empty_ae"),
                   "--report-dir", str(tmp_path)])
    assert sentinel.read_text() == "KEEP-ME"


def test_emit_allow_empty_writes_placeholders(tmp_path):
    """--allow-empty is the on-purpose escape hatch: it proceeds and writes the
    placeholder macros (with the documented AE lambda* fallback)."""
    (tmp_path / "tables").mkdir()
    _run_main(["--soft-csv", str(tmp_path / "absent.csv"),
               "--faith-csv", str(tmp_path / "absent2.csv"),
               "--runs-dir", str(tmp_path / "empty_runs"),
               "--ae-run-dir", str(tmp_path / "empty_ae"),
               "--report-dir", str(tmp_path), "--allow-empty"])
    text = (tmp_path / "macros-results.tex").read_text()
    assert r"\newcommand{\reconstructionLambdaStar}{93.7}" in text  # documented fallback
    assert r"\maxPiFortyEightConverged" in text  # placeholder line still emitted


# -------------------------------- Round-1 review emissions (2 Jul worklist)
def test_build_stratum_constant_macros_match_pipeline_rule():
    """DEC-1(c): the rendered stratum constants are read from the pipeline rule's
    own signature defaults (phase4_eval.practically_bimodal_mask), so the S5.2
    definition and the mask every stratified number uses cannot diverge. Pins the
    rendered strings AND re-derives them from the rule."""
    import inspect

    from sampler_research.phase4_eval import practically_bimodal_mask

    m = _load()
    macros = m.build_stratum_constant_macros()
    params = inspect.signature(practically_bimodal_mask).parameters
    assert macros["stratumPiGate"] == format(params["pi_min"].default, "g") == "0.1"
    assert (macros["stratumSepMultiplier"]
            == format(params["min_separation"].default, "g") == "2")


@needs_fc_converged
def test_build_enrichment_ratio_macros_pin_pr1_values():
    """PR-1 condition 1: the eight S5.5 enrichment-rewrite ratios reproduce the
    2 Jul log/ruling values from the persisted canonical +48h artifacts -- Joint
    MAP de-enriched (0.69x) in the 2-sigma stratum at the 0.125-nat headline cut
    but 1.34x at the 0.5-nat deep cut; blur 0.59x / 0.44x there; 1-sigma headline
    1.23x vs 0.77x. Computed exactly as the enrichment figure's bars."""
    m = _load()
    macros = m.build_enrichment_ratio_macros(FC_CONVERGED_RUN)
    assert macros["enrichMethodTwoSigmaHeadline"] == "0.69"
    assert macros["enrichMethodTwoSigmaDeep"] == "1.34"
    assert macros["enrichMethodOneSigmaHeadline"] == "1.23"
    assert macros["enrichMethodOneSigmaDeep"] == "1.27"
    assert macros["enrichBlurTwoSigmaHeadline"] == "0.59"
    assert macros["enrichBlurTwoSigmaDeep"] == "0.44"
    assert macros["enrichBlurOneSigmaHeadline"] == "0.77"
    assert macros["enrichBlurOneSigmaDeep"] == "0.67"


def test_build_enrichment_ratio_macros_placeholders_when_absent(tmp_path):
    m = _load()
    macros = m.build_enrichment_ratio_macros(tmp_path)  # no artifacts under here
    assert set(macros) == set(m.ENRICH_MACROS)
    assert all(v == m.PLACEHOLDER for v in macros.values())


@needs_fc_converged
def test_build_threshold_sensitivity_macros_pin_dec4_values():
    """DEC-4: the sensitivity macros reproduce the 2 Jul log numbers exactly via
    the pipeline's own `_bootstrap_frac_ci` at varied threshold (production seed,
    no new runs) -- the stratified gap still favours the sampler at the 0.25-nat
    (~0.7-sigma-equivalent) cut, -3.4 pp CI [-4.5, -2.3] pp, reverses only at the
    0.5-nat deep cut, +1.7 pp CI [+0.8, +2.7] pp, where Joint MAP's deep tail is
    6.3% of the stratum. The headline 0.125-nat macros are untouched (their own
    pins above cover that). Rendered in pp, not a bare decimal, since these are
    differences between two cell shares (harvey, 4 Jul, DEC-38 r3 follow-up)."""
    m = _load()
    macros = m.build_threshold_sensitivity_macros(FC_CONVERGED_RUN)
    assert macros["fcBimodalFracGapQuarterNat"] == r"-3.4\,pp"
    assert macros["fcBimodalFracGapQuarterNatLo"] == r"-4.5\,pp"
    assert macros["fcBimodalFracGapQuarterNatHi"] == r"-2.3\,pp"
    assert macros["fcBimodalFracGapHalfNat"] == r"+1.7\,pp"
    assert macros["fcBimodalFracGapHalfNatLo"] == r"+0.8\,pp"
    assert macros["fcBimodalFracGapHalfNatHi"] == r"+2.7\,pp"
    assert macros["fcDeepTailStratumFrac"] == r"6.3\%"


def test_build_threshold_sensitivity_macros_placeholders_when_absent(tmp_path):
    m = _load()
    macros = m.build_threshold_sensitivity_macros(tmp_path)
    assert macros["fcBimodalFracGapQuarterNat"] == m.PLACEHOLDER
    assert macros["fcBimodalFracGapHalfNatHi"] == m.PLACEHOLDER
    assert macros["fcDeepTailStratumFrac"] == m.PLACEHOLDER


@needs_fc_spectra
def test_build_spectrum_resolution_macro_pins_ell_res():
    """DEC-9: \\spectrumEllRes is the artifact's own lmax_resolved (the empirical
    Parseval ceiling, 16 Jun log: 191), never hand-typed."""
    m = _load()
    macros = m.build_spectrum_resolution_macro(FC_CONVERGED_RUN.parent)
    assert macros["spectrumEllRes"] == "191"


def test_build_spectrum_resolution_macro_placeholder_when_absent(tmp_path):
    m = _load()
    assert m.build_spectrum_resolution_macro(tmp_path) == {
        "spectrumEllRes": m.PLACEHOLDER}


@needs_fc_converged
def test_build_restart_stability_macro_pins_probe_shift():
    """PR-4: the restart-stability sentence's evidence is the persisted
    seed-stability probe, never the across-init lambda* range (which measures
    initial-condition variation, not restarts). Pins the worst-case reseeded
    lambda* shift on the canonical +48h run: 81.3387 -> 81.3403, i.e. 1.6e-3."""
    m = _load()
    macros = m.build_restart_stability_macro(FC_CONVERGED_RUN)
    assert macros["lambdaStarRestartMaxShift"] == r"1.6\times10^{-3}"


def test_build_restart_stability_macro_placeholder_when_absent(tmp_path):
    m = _load()
    assert m.build_restart_stability_macro(tmp_path) == {
        "lambdaStarRestartMaxShift": m.PLACEHOLDER}


# ---------------------- App D certificate + App E compute macros (2 Jul PM rulings)
STAGE_B_TV_CSV = REPO_ROOT / "outputs" / "runs" / "stage_b_tv_ablation" / "stage_b_tv_scores.csv"
STAGE_D_CSV = REPO_ROOT / "outputs" / "runs" / "stage_d_method5_langevin" / "stage_d_scores.csv"
needs_certificate_csvs = pytest.mark.skipif(
    not (STAGE_B_TV_CSV.exists() and STAGE_D_CSV.exists()),
    reason="toy certificate CSV artifacts absent",
)


def _quad_row(lam, delta_j, bound, beats):
    return {"arm": "cut-quad", "lambda": lam, "delta_j_vs_stage_b": delta_j,
            "quantisation_bound": bound, "beats_stage_b": beats}


def _cut_tv_row(lam, matches):
    return {"arm": "cut-tv", "lambda": lam, "adam_matches_cut": matches}


CERT_TV_ROWS = [
    _quad_row(0.0, -0.023, 3e-05, "True"),
    _quad_row(0.05, -0.089, 3e-05, "True"),
    _quad_row(0.1, -0.051, 3e-05, "False"),
    _quad_row(0.2, -0.007, 3e-05, "False"),
    _quad_row(0.5, 2.7e-06, 3e-05, "False"),
    _quad_row(1.0, 3.3e-06, 3e-05, "False"),
    _quad_row(2.0, 2.9e-07, 3e-05, "False"),
    _cut_tv_row(0.0, "False"),
    _cut_tv_row(0.1, "False"),
    _cut_tv_row(0.2, "False"),
    _cut_tv_row(0.5, "True"),
    _cut_tv_row(1.0, "True"),
]
CERT_M5_ROWS = [
    {"lambda": 0.05, "delta_j_vs_method1": -0.02813800736,
     "chain_energy_spread": 0.03090128230, "material_improvement": "True"},
    {"lambda": 0.2, "delta_j_vs_method1": -6.0e-08,
     "chain_energy_spread": 0.074, "material_improvement": "False"},
]


def test_build_certificate_macros_derives_thresholds():
    """Decision 5: the five certificate numbers are DERIVED from the CSV rows
    (certified suffix / within-floor band / failure prefix / single flagged
    Langevin case), never pinned in the emitter."""
    m = _load()
    macros = m.build_certificate_macros(CERT_TV_ROWS, CERT_M5_ROWS)
    assert macros["certQuadLambdaGlobalMin"] == "0.5"
    assert macros["certQuadFloorLambdaLo"] == "0.1"
    assert macros["certQuadFloorLambdaHi"] == "0.2"
    assert macros["certTvFailLambdaMax"] == "0.2"
    assert macros["certLangevinDeltaJ"] == "-0.0281"
    assert macros["certLangevinChainSpread"] == "0.031"


def test_build_certificate_macros_placeholders_when_absent():
    m = _load()
    macros = m.build_certificate_macros([], [])
    assert all(v == m.PLACEHOLDER for v in macros.values())


def test_build_certificate_macros_rejects_broken_structure():
    """The structural assertions fail loudly if the certificate story changes:
    a non-suffix certified set, or a flagged Langevin case OUTSIDE its own
    chain spread, must raise rather than silently re-number the appendix."""
    m = _load()
    broken_quad = [r if r["lambda"] != 1.0 else dict(r, delta_j_vs_stage_b=-0.5)
                   for r in CERT_TV_ROWS]
    with pytest.raises(ValueError):
        m.build_certificate_macros(broken_quad, CERT_M5_ROWS)
    outside_spread = [dict(CERT_M5_ROWS[0], chain_energy_spread=0.01)]
    with pytest.raises(ValueError):
        m.build_certificate_macros(CERT_TV_ROWS, outside_spread)


@needs_certificate_csvs
def test_build_certificate_macros_pins_committed_artifacts():
    """The committed certificate CSVs must yield exactly the numbers the App D
    passage quotes (handoff-verified 2 Jul): certified-global from lambda=0.5,
    floor band 0.1-0.2, TV failure through 0.2, flagged dJ=-0.0281 inside its
    0.031 chain spread."""
    m = _load()
    macros = m.build_certificate_macros(
        m._read_csv_numeric(STAGE_B_TV_CSV), m._read_csv_numeric(STAGE_D_CSV))
    assert macros == {
        "certQuadLambdaGlobalMin": "0.5",
        "certQuadFloorLambdaLo": "0.1",
        "certQuadFloorLambdaHi": "0.2",
        "certTvFailLambdaMax": "0.2",
        "certLangevinDeltaJ": "-0.0281",
        "certLangevinChainSpread": "0.031",
    }


COMPUTE_RECORD = {
    "jobs": {
        "ae_training": {"elapsed": "11:49:35"},
        "forecast_training_v2": {"elapsed": "10:07:05"},
        "extraction": {"elapsed": "00:01:11"},
    }
}
COMPUTE_TIMINGS = {
    "knn_graph_s": 0.558, "anchors_s": 0.022, "mode_extraction_s": 3.098,
    "m1_sweep_s": 37.482, "m4_sweep_s": 35.008, "icm_s_per_sweep": 1.094,
    "scores_s": 0.979, "variograms_s": 0.083,
}


def test_build_compute_macros_formats_wall_times():
    """Decision 6: wall-times from the once-persisted sacct record; the audit
    total sums the timed stages and excludes the icm per-sweep RATE entry."""
    m = _load()
    macros = m.build_compute_macros(COMPUTE_RECORD, COMPUTE_TIMINGS)
    assert macros["computeAeWallHours"] == "11.8"
    assert macros["computeForecastWallHours"] == "10.1"
    assert macros["computeExtractionWallMinutes"] == "1.2"
    assert macros["computeAuditWallSeconds"] == "77"


def test_build_compute_macros_placeholders_when_absent():
    m = _load()
    macros = m.build_compute_macros(None, None)
    assert all(v == m.PLACEHOLDER for v in macros.values())
