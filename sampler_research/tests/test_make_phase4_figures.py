"""F5: the robustness figure must skip gracefully when its probe artifact is
absent (the forecast/per-lead regimes never generate robustness_probes.json).

These tests load the script module and exercise only the existence guard in
`fig_robustness`; they do not render the full figure (that path needs the whole
phase_4_real run dir and is exercised by the pipeline).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
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
