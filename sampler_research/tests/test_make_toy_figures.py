"""Provenance + smoke tests for scripts/make_toy_figures.py.

The toy report figures are now script-generated (notebooks 02/03 are exploratory
only), so these tests pin two things: (1) the figure's live per-cell drift-off-mode
numbers reproduce the committed stage_c CSV to machine precision -- i.e. porting
the analysis out of the notebook changed no result -- and (2) both figures render
to a PNG. They load the script as a module (it guards `main()` behind
`__main__`), so importing it renders nothing.
"""

from __future__ import annotations

import csv
import importlib.util
import sys

import pytest

from conftest import REPO_ROOT, RUNS_DIR, needs_toy

SCRIPT = REPO_ROOT / "scripts" / "make_toy_figures.py"
STAGE_C_CSV = RUNS_DIR / "stage_c_method4_mrf" / "stage_c_scores.csv"

needs_stage_runs = pytest.mark.skipif(
    not STAGE_C_CSV.exists(), reason="persisted stage_* toy artifacts absent"
)


def _load():
    spec = importlib.util.spec_from_file_location("make_toy_figures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@needs_toy
@needs_stage_runs
def test_smear_matches_committed_stage_c_csv():
    """Live delta-NLL-to-mode (mean and frac>0.125) for every Method-4 field
    reproduces the committed stage_c_scores.csv -- the analysis is unchanged."""
    m = _load()
    art = m.load_artifacts()
    with open(STAGE_C_CSV, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(art["C"]["fields"])
    for field, row in zip(art["C"]["fields"], rows):
        d = m.smear(field, art)
        assert d["mean"] == pytest.approx(float(row["dnll_to_mode_mean"]), abs=1e-9)
        assert d["frac_over"][0] == pytest.approx(
            float(row["dnll_frac_gt_0p125_v1"]), abs=1e-12)


@needs_toy
@needs_stage_runs
def test_smear_tail_ci_brackets_point():
    """The within-field bootstrap `point` is the full-sample frac>0.125, and the
    95% interval brackets it (seeded, so this is deterministic)."""
    m = _load()
    art = m.load_artifacts()
    beta_idx = list(art["C"]["betas"]).index(0.1)
    d = m.smear(art["C"]["fields"][beta_idx], art)  # beta=0.1: a non-zero tail
    ci = m.smear_tail_ci(d["per_cell"])
    assert ci["point"] == pytest.approx(d["frac_over"][0], abs=1e-12)
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert ci["n_boot"] == m.BOOT_N_DRAWS


@needs_toy
@needs_stage_runs
def test_both_figures_render(tmp_path):
    """Both report figures render to a non-empty PNG under the expected names."""
    m = _load()
    art = m.load_artifacts()
    out1 = m.fig_nonsmearing(art, tmp_path)
    out2 = m.fig_pareto_plane(art, tmp_path)
    assert out1.name == "phase_2_stage_bc_nonsmearing_homoscedastic.png"
    assert out2.name == "phase_2_tv_pareto_plane_homoscedastic.png"
    for out in (out1, out2):
        assert out.exists() and out.stat().st_size > 0
