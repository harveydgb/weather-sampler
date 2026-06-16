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

SCRIPT = REPO_ROOT / "scripts" / "make_phase4_figures.py"


def _load_figures():
    spec = importlib.util.spec_from_file_location("make_phase4_figures", SCRIPT)
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
    assert "skip phase_4_robustness.png" in out
    assert not (tmp_path / "phase_4_robustness.png").exists()


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
# default reconstruction render that shares the bare filename. Pins audit T1.
_FORECAST_STEP8 = "phase_4_fc48_14ep_step8"


@pytest.mark.skipif(not THESIS.exists(), reason="report/thesis.tex absent")
@pytest.mark.parametrize(
    "stem",
    ["phase_4_pareto_smear.png", "phase_4_variogram.png", "phase_4_spectrum.png"],
)
def test_forecast_figures_point_at_step8_render(stem):
    """The Ch5 pareto-smear, the main-text spectrum, and the descriptive variogram
    must \\includegraphics the +48h converged render, and that asset must exist."""
    text = THESIS.read_text()
    includes = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*" + re.escape(stem) + r")\}", text)
    assert includes, f"no \\includegraphics for {stem} in thesis.tex"
    for path in includes:
        assert path == f"{_FORECAST_STEP8}/{stem}", (
            f"{stem} should resolve to the +48h step-8 render, got {path!r}")
        assert (FIG_DIR / path).exists(), f"missing forecast render asset: {path}"
