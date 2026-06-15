"""Tests for the forecast-lead Phase 4 batch runner.

These tests inspect command planning and synthetic lead discovery only. They do
not import torch and do not require the real forecast artifact.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from conftest import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "run_phase4_forecast_leads.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_phase4_forecast_leads", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _arg_after(argv, flag):
    return argv[argv.index(flag) + 1]


def test_discover_lead_files_sorts_numerically_and_filters(tmp_path):
    runner = _load_runner()
    prefix = "phase_4_fc48_6ep"
    (tmp_path / f"{prefix}_step8_2t.npz").touch()
    (tmp_path / f"{prefix}_step1_2t.npz").touch()
    (tmp_path / f"{prefix}_step10_2t.npz").touch()
    (tmp_path / f"{prefix}_stepx_2t.npz").touch()
    (tmp_path / "other_step2_2t.npz").touch()

    assert [step for step, _ in runner.discover_lead_files(tmp_path, prefix)] == [1, 8, 10]
    assert runner.parse_lead_steps("1,8") == (1, 8)
    filtered = runner.discover_lead_files(tmp_path, prefix, runner.parse_lead_steps("1,8"))
    assert [step for step, _ in filtered] == [1, 8]
    assert [path.name for _, path in filtered] == [
        f"{prefix}_step1_2t.npz",
        f"{prefix}_step8_2t.npz",
    ]


def test_dry_run_command_plan_uses_prefix_format_distinct_dirs_and_shared_graph(tmp_path):
    runner = _load_runner()
    prefix = "phase_4_fc48_6ep"
    forecast_pt = tmp_path / "gmm_params_gmm_fc48_v1_me5_2t_f8.pt"
    data_dir = tmp_path / "data"
    runs_dir = tmp_path / "runs"
    graph_cache = runs_dir / "o96_knn_k8_graph.npz"
    lead_files = runner.expected_lead_files(data_dir, prefix, (1, 8))

    commands = runner.build_commands(
        forecast_pt=forecast_pt,
        prefix=prefix,
        data_dir=data_dir,
        runs_dir=runs_dir,
        graph_cache=graph_cache,
        lead_files=lead_files,
        skip_convert=False,
        quick=True,
        figures=False,
        figures_dir=tmp_path / "figures",
        convert_python="convert-python",
        phase4_python="phase4-python",
    )

    convert = commands[0].argv
    assert commands[0].label == "convert"
    assert "--format" in convert
    assert _arg_after(convert, "--format") == "forecast"
    assert _arg_after(convert, "--prefix") == prefix
    assert _arg_after(convert, "--input") == str(forecast_pt)

    phase4 = [cmd.argv for cmd in commands if "run_phase4_real.py" in cmd.argv[1]]
    assert len(phase4) == 6
    full_runs = [argv for argv in phase4 if "--lambda-star" not in argv and "--stages" not in argv]
    assert [_arg_after(argv, "--out-dir") for argv in full_runs] == [
        str(runs_dir / f"{prefix}_step1"),
        str(runs_dir / f"{prefix}_step8"),
    ]
    assert {Path(_arg_after(argv, "--graph-cache")) for argv in phase4} == {graph_cache}
    assert all("--quick" in argv for argv in phase4)


def test_skip_convert_plans_only_existing_per_lead_npz_files(tmp_path):
    runner = _load_runner()
    prefix = "phase_4_fc48_6ep"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    step1 = data_dir / f"{prefix}_step1_2t.npz"
    step8 = data_dir / f"{prefix}_step8_2t.npz"
    step1.touch()
    step8.touch()
    lead_files = runner.discover_lead_files(data_dir, prefix, runner.parse_lead_steps("1,8"))

    commands = runner.build_commands(
        forecast_pt=tmp_path / "unused.pt",
        prefix=prefix,
        data_dir=data_dir,
        runs_dir=tmp_path / "runs",
        graph_cache=tmp_path / "runs" / "o96_knn_k8_graph.npz",
        lead_files=lead_files,
        skip_convert=True,
        quick=False,
        figures=False,
        figures_dir=tmp_path / "figures",
        convert_python="convert-python",
        phase4_python="phase4-python",
    )

    assert all(cmd.label != "convert" for cmd in commands)
    first_phase4_per_step = [
        cmd.argv for cmd in commands
        if cmd.label in ("step1: phase4", "step8: phase4")
    ]
    assert [_arg_after(argv, "--data") for argv in first_phase4_per_step] == [
        str(step1),
        str(step8),
    ]
    assert [_arg_after(argv, "--out-dir") for argv in first_phase4_per_step] == [
        str(tmp_path / "runs" / f"{prefix}_step1"),
        str(tmp_path / "runs" / f"{prefix}_step8"),
    ]


def test_per_lead_figures_exclude_robustness_via_only(tmp_path):
    """Per-lead figure command must pass --only (excludes the robustness figure,
    which reads robustness_probes.json the forecast path never generates)."""
    runner = _load_runner()
    prefix = "phase_4_fc48_14ep"
    data_dir = tmp_path / "data"
    lead_files = runner.expected_lead_files(data_dir, prefix, (8,))
    commands = runner.build_commands(
        forecast_pt=tmp_path / "x.pt", prefix=prefix, data_dir=data_dir,
        runs_dir=tmp_path / "runs", graph_cache=tmp_path / "g.npz",
        lead_files=lead_files, skip_convert=True, quick=False, figures=True,
        figures_dir=tmp_path / "figures", convert_python="cp", phase4_python="pp",
    )
    fig = next(c.argv for c in commands if c.label == "step8: figures")
    assert "--only" in fig
    only_idx = fig.index("--only")
    only_vals = fig[only_idx + 1: only_idx + 1 + len(runner.LEAD_FIGURE_NAMES)]
    assert tuple(only_vals) == runner.LEAD_FIGURE_NAMES
    assert "robustness" not in fig
    assert "make_phase4_figures.py" in fig[1]
