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


@needs_toy
def test_component_fields_figure_renders(tmp_path):
    """Component-fields figure renders to a non-empty PNG under its expected
    name. Needs only the toy `.npz` -- unlike the other two figures, no
    persisted stage_* run is required (`load_component_fields` ignores
    `runs_dir`)."""
    m = _load()
    art = m.load_component_fields()
    out = m.fig_component_fields(art, tmp_path)
    assert out.name == "phase_1_component_fields.png"
    assert out.exists() and out.stat().st_size > 0


def test_figures_registry_shares_one_loader_signature():
    """Every FIGURES loader accepts (runs_dir, data_dir) so `main` can dispatch
    without special-casing any entry -- this is what lets a figure that needs
    no stage_* run (`components`) sit in the same registry as ones that do."""
    m = _load()
    import inspect

    for loader, fig_fn in m.FIGURES.values():
        params = list(inspect.signature(loader).parameters)
        assert params[:2] == ["runs_dir", "data_dir"]
        assert callable(fig_fn)


@needs_toy
@needs_stage_runs
def test_stage_a_baselines_figure_renders(tmp_path):
    """Stage A baselines figure renders to a non-empty PNG under its expected
    name, with one panel per TOY_BASELINE_LABELS entry."""
    m = _load()
    art = m.load_artifacts()
    out = m.fig_stage_a_baselines(art, tmp_path)
    assert out.name == "phase_2_stage_a_baselines.png"
    assert out.exists() and out.stat().st_size > 0


TOY_BASELINES_TABLE = REPO_ROOT / "report" / "construction" / "tables" / "toy_baselines.tex"


# The figure intentionally drops these two qualifiers from the table's
# wording (crowds the larger panel titles otherwise; see TOY_BASELINE_LABELS'
# comment in make_toy_figures.py). Everything else -- row order, the other
# three labels, and any future table row -- must still match verbatim.
_DROPPED_QUALIFIERS = {
    "Independent draw (iid)": "Independent draw",
    "Per-cell MAP (mode)": "Per-cell MAP",
}


@pytest.mark.skipif(not TOY_BASELINES_TABLE.exists(), reason="toy_baselines.tex absent")
def test_stage_a_baselines_labels_match_table_4_1():
    """TOY_BASELINE_LABELS (this figure's panel order + titles) must match
    Table 4.1's rendered rows, up to the two known dropped qualifiers -- the
    whole point of the figure is to be a drop-in replacement for that table,
    so anything else must never drift."""
    m = _load()
    lines = TOY_BASELINES_TABLE.read_text().splitlines()
    body = lines[lines.index(r"\midrule") + 1:lines.index(r"\bottomrule")]
    table_rows = [line.split("&", 1)[0].strip() for line in body]
    expected = [_DROPPED_QUALIFIERS.get(row, row) for row in table_rows]
    assert expected == [label for _key, label in m.TOY_BASELINE_LABELS]
