"""F5: the robustness figure must skip gracefully when its probe artifact is
absent (the forecast/per-lead regimes never generate robustness_probes.json).

These tests load the script module and exercise only the existence guard in
`fig_robustness`; they do not render the full figure (that path needs the whole
phase_4_real run dir and is exercised by the pipeline).
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

from conftest import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "make_real_figures.py"


def _load_figures():
    spec = importlib.util.spec_from_file_location("make_real_figures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fig_robustness_skips_when_probes_absent(tmp_path, capsys):
    figures = _load_figures()
    figures.RUN_DIR = tmp_path  # empty: no robustness_probes.json
    figures.FIG_DIR = tmp_path

    # Must not raise even though modes.npz and the probes json are both missing.
    assert figures.fig_robustness({"lambda_star": 93.74}) is None
    out = capsys.readouterr().out
    assert "skip robustness.png" in out
    assert not (tmp_path / "robustness.png").exists()


def test_fig_robustness_passes_guard_when_probes_present(tmp_path):
    """With the json present the guard does NOT short-circuit: the function
    proceeds and fails loading the (here-absent) modes.npz, proving the skip is
    keyed only on the probe file's absence."""
    figures = _load_figures()
    figures.RUN_DIR = tmp_path
    figures.FIG_DIR = tmp_path
    (tmp_path / "robustness_probes.json").write_text("{}")

    with pytest.raises(FileNotFoundError):
        figures.fig_robustness({"lambda_star": 93.74})


import re

THESIS = REPO_ROOT / "report" / "thesis.tex"
FIG_DIR = REPO_ROOT / "outputs" / "figures"
# Forecast-chapter figures must show the +48h converged render (step 8), not the
# reconstruction render (now under recon/) that shares the bare filename. Pins
# audit T1. This names the FIGURE dir (de-phased); the matching RUN dir keeps its
# internal phase_4_fc48_ prefix and is spelled out separately in _RUN_STEP8 below.
_FORECAST_STEP8 = "forecast_14ep_step8"


@pytest.mark.skipif(not THESIS.exists(), reason="report/thesis.tex absent")
@pytest.mark.parametrize(
    "stem",
    [
        "pareto_smear.png",
        "spectrum.png",
        "bimodal_enrichment.png",
    ],
)
def test_forecast_figures_point_at_step8_render(stem):
    """The Ch5 forecast field figures must \\includegraphics the +48h
    converged step-8 render, and that asset must exist. The appendices may
    show the OTHER regimes of the same stems on purpose (App B recon-regime
    sweep, App C recon + per-lead maps), but only from the known re-rendered
    set, never a stale path. (The maps stem has its own tests below: the
    main text carries the REGIONAL render, the whole-globe renders live in
    Appendix C.)"""
    # Anchor the stem to a path boundary (start of the {…} or a directory
    # slash) so a de-phased bare filename like maps.png cannot substring-match a
    # longer sibling such as region_maps.png / region_lambda_sweep_maps.png.
    pattern = r"\\includegraphics(?:\[[^\]]*\])?\{((?:[^}]*/)?" + re.escape(stem) + r")\}"
    main_text, _, appendix = THESIS.read_text().partition("\n\\appendix")
    includes = re.findall(pattern, main_text)
    assert includes, f"no \\includegraphics for {stem} in thesis.tex main text"
    for path in includes:
        assert path == f"{_FORECAST_STEP8}/{stem}", (
            f"{stem} should resolve to the +48h step-8 render, got {path!r}")
        assert (FIG_DIR / path).exists(), f"missing forecast render asset: {path}"
    allowed_appendix = {
        f"recon/{stem}",  # reconstruction regime (step 0) now lives under recon/
        f"forecast_14ep_step1/{stem}",  # +6h per-lead render
        f"forecast_14ep_step4/{stem}",  # +24h per-lead render
        f"{_FORECAST_STEP8}/{stem}",
    }
    for path in re.findall(pattern, appendix):
        assert path in allowed_appendix, (
            f"appendix include of {stem} outside the allowed renders: {path!r}")
        assert (FIG_DIR / path).exists(), f"missing appendix render asset: {path}"


def _thesis_split():
    main_text, _, appendix = THESIS.read_text().partition("\n\\appendix")
    return main_text, appendix


def _includes(text, stem):
    # Anchor the stem to a path boundary (start of the {…} or a directory
    # slash) so a de-phased bare filename like maps.png cannot substring-match a
    # longer sibling such as region_maps.png / region_lambda_sweep_maps.png.
    pattern = r"\\includegraphics(?:\[[^\]]*\])?\{((?:[^}]*/)?" + re.escape(stem) + r")\}"
    return re.findall(pattern, text)


@pytest.mark.skipif(not THESIS.exists(), reason="report/thesis.tex absent")
def test_main_text_maps_figure_is_region_render():
    """The main-text field figure is the North Atlantic / Europe REGIONAL
    render of the +48h step-8 run (the whole-globe projection cannot display
    grid-scale texture at O96), and no whole-globe maps render remains in
    the main text."""
    main_text, _ = _thesis_split()
    region = _includes(main_text, "region_maps.png")
    assert region == [f"{_FORECAST_STEP8}/region_maps.png"], (
        f"main text must include exactly the step-8 region render, got {region!r}")
    assert (FIG_DIR / region[0]).exists(), f"missing region render asset: {region[0]}"
    stray = _includes(main_text, "maps.png")
    assert stray == [], (
        f"whole-globe maps renders belong in Appendix C, found in main text: {stray!r}")


@pytest.mark.skipif(not THESIS.exists(), reason="report/thesis.tex absent")
def test_appendix_global_maps_renders_from_allowed_set():
    """Appendix C carries the whole-globe maps renders: recon (root), +6h,
    +24h, and the +48h step-8 view of the main-text region figure. Only known
    re-rendered paths are allowed, and the step-8 view must be present."""
    _, appendix = _thesis_split()
    allowed = {
        "recon/maps.png",
        "forecast_14ep_step1/maps.png",
        "forecast_14ep_step4/maps.png",
        f"{_FORECAST_STEP8}/maps.png",
    }
    found = _includes(appendix, "maps.png")
    assert set(found) <= allowed, f"appendix maps include outside allowed set: {found!r}"
    assert f"{_FORECAST_STEP8}/maps.png" in found, (
        "Appendix C must carry the whole-globe step-8 render")
    for path in found:
        assert (FIG_DIR / path).exists(), f"missing appendix render asset: {path}"


@pytest.mark.skipif(not THESIS.exists(), reason="report/thesis.tex absent")
def test_appendix_region_lambda_sweep_is_step8_render():
    """The Appendix B lambda-sweep strip is the +48h step-8 regional render,
    provenance-pinned like the other forecast figures."""
    _, appendix = _thesis_split()
    found = _includes(appendix, "region_lambda_sweep_maps.png")
    assert found == [f"{_FORECAST_STEP8}/region_lambda_sweep_maps.png"], (
        f"App B lambda strip must be the step-8 region render, got {found!r}")
    assert (FIG_DIR / found[0]).exists(), f"missing lambda-strip asset: {found[0]}"


# --- Spectrum curve inventory ---------------------------------------------------
# The run dir keeps its internal phase_4_fc48_ prefix (only figure dirs were
# de-phased), so it is spelled out here rather than derived from _FORECAST_STEP8.
_RUN_STEP8 = REPO_ROOT / "outputs" / "runs" / "phase_4_fc48_14ep_step8"


def test_spectrum_curve_set_drops_duplicates_and_keeps_budget():
    """The main-text angular power spectrum shows exactly the fig:fc-maps method
    set: the two mid-band duplicates (Mixture mean, Mode-selection MRF) are
    dropped and the faithfulness-budget Joint MAP curve is included, in the
    maps figure's panel order."""
    figures = _load_figures()
    assert figures.SPECTRUM_CURVES == (
        "mode_map", "iid_seed0", "m1_star", "smoothed_map_n10", "m1_budget", "era5")
    assert "mixture_mean" not in figures.SPECTRUM_CURVES
    assert "m4_beta1" not in figures.SPECTRUM_CURVES


@pytest.mark.skipif(not (_RUN_STEP8 / "spectra.npz").exists(),
                    reason="converged +48h spectra artifact absent")
def test_fig_spectrum_plots_maps_panel_set_only(tmp_path, monkeypatch):
    """Driving fig_spectrum on the canonical run dir plots ONLY the fig:fc-maps
    method set (in panel order) even though spectra.npz still carries the retired
    mixture_mean / m4_beta1 curves, and labels the budget curve with the same
    lambda the maps figure's budget panel uses (single-sourced from
    budget_point.npz), so the two figures cannot drift."""
    import numpy as np

    figures = _load_figures()
    figures.RUN_DIR = _RUN_STEP8
    figures.FIG_DIR = tmp_path

    captured = {}

    def _capture(ell, spectra, *, title, labels=None, figsize=None):
        captured["keys"] = list(spectra)
        captured["labels"] = labels or {}
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([0], [0], label="stub")  # a labelled artist so fig_spectrum's legend() is quiet
        return fig, ax

    monkeypatch.setattr(figures, "plot_spectra", _capture)
    figures.fig_spectrum()

    with np.load(_RUN_STEP8 / "spectra.npz") as f:
        available = set(f.files) - {"ell", "lmax_resolved"}
    expected = [k for k in figures.SPECTRUM_CURVES if k in available]
    assert captured["keys"] == expected, "spectrum curve set / order drifted from fig:fc-maps"
    # The retired duplicates remain in the artifact but must never reach the plot.
    assert "mixture_mean" in available and "mixture_mean" not in captured["keys"]
    assert "m4_beta1" in available and "m4_beta1" not in captured["keys"]

    budget_path = _RUN_STEP8 / "budget_point.npz"
    if budget_path.exists():
        with np.load(budget_path) as f:
            lam = float(f["lambda_budget"])
        assert captured["labels"].get("m1_budget") == \
            f"Joint MAP ($\\lambda$={lam:.0f})", \
            "budget curve label must match the maps figure's budget panel"
