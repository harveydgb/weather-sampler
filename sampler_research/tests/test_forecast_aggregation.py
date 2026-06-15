"""Across-init aggregation helpers in the forecast runners (F1 track-2).

Pure functions over row-dict fixtures (no npz / run-dir I/O): the
`kind='init_mean'` rows carry the across-init MEAN in the base metric column and
the min-max RANGE in the `{metric}_lo`/`{metric}_hi` columns, over the inits
present at each lead step. Fewer than two inits => no aggregate (single-init
behaviour preserved).
"""

import csv
import importlib.util
import json
import sys

import numpy as np
import pytest

from sampler_research.phase4_eval import select_lambda_star

from conftest import REPO_ROOT
SOFT_CSV = REPO_ROOT / "outputs" / "runs" / "phase_4_forecast_softening" / "softening_by_lead.csv"
FAITH_CSV = REPO_ROOT / "outputs" / "runs" / "phase_4_forecast_faithfulness" / "faithfulness_by_lead.csv"
FC_RUN = REPO_ROOT / "outputs" / "runs" / "phase_4_fc48_14ep_step8"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _soft_row(init, step, max_pi):
    # only the fields aggregate_converged_rows reads/copies
    return {"run": "14ep", "column": "Converged", "run_id": "gmm_fc48_v2",
            "kind": "single", "init_datetime": init, "step": step, "lead_hours": 6.0 * step,
            "median_max_pi": max_pi, "one_hot_fraction": 0.1, "median_second_mode": 0.17,
            "bimodal_frac_1sigma": 0.3, "bimodal_frac_2sigma": 0.05, "keff_frac_gt_1p5": 0.9}


def test_softening_aggregate_mean_and_range():
    m = _load("run_forecast_softening")
    per_init = [
        ("pA", [_soft_row("2023-11-01T00:00", 8, 0.766)]),
        ("pB", [_soft_row("2023-10-10T12:00", 8, 0.776)]),
        ("pC", [_soft_row("2023-12-15T00:00", 8, 0.760)]),
    ]
    out = m.aggregate_converged_rows(per_init)
    assert len(out) == 1
    row = out[0]
    assert row["kind"] == "init_mean" and row["n_inits"] == 3
    assert abs(row["median_max_pi"] - (0.766 + 0.776 + 0.760) / 3) < 1e-12
    assert row["median_max_pi_lo"] == 0.760 and row["median_max_pi_hi"] == 0.776
    # init dates are recorded (sorted, ;-joined) and the lead is carried through
    assert row["init_datetime"] == "2023-10-10T12:00;2023-11-01T00:00;2023-12-15T00:00"
    assert row["lead_hours"] == 48.0


def test_softening_aggregate_needs_two_inits():
    m = _load("run_forecast_softening")
    assert m.aggregate_converged_rows([("pA", [_soft_row("2023-11-01T00:00", 8, 0.766)])]) == []
    assert m.aggregate_converged_rows([]) == []


def _faith_row(init, step, lam, cov90):
    return {"run": "14ep", "column": "Converged", "kind": "single", "init_datetime": init,
            "step": step, "lead_hours": 6.0 * step, "lambda_star": lam,
            "iid_delta_crps": -1e-5, "iid_ensemble_crps": 0.05, "iid_analytic_crps": 0.05,
            "iid_pit_ks": 0.004, "iid_n_members": 50, "m1_star_pit_ks": 0.31,
            "m1_star_cov50": 0.91, "m1_star_cov90": cov90,
            "m1_star_mean_abs_pit_centre": 0.08, "m1_star_nll_over_n": -1.6}


def test_faithfulness_aggregate_mean_and_spread_keys():
    m = _load("run_forecast_faithfulness")
    per_init = [
        ("pA", [_faith_row("2023-11-01T00:00", 8, 81.3, 0.990)]),
        ("pB", [_faith_row("2023-10-10T12:00", 8, 63.8, 0.992)]),
        ("pC", [_faith_row("2023-12-15T00:00", 8, 89.4, 0.991)]),
    ]
    out = m.aggregate_converged_faith(per_init)
    assert len(out) == 1
    row = out[0]
    assert row["kind"] == "init_mean" and row["n_inits"] == 3
    assert abs(row["lambda_star"] - (81.3 + 63.8 + 89.4) / 3) < 1e-9
    # spread keys get a min-max range
    assert row["lambda_star_lo"] == 63.8 and row["lambda_star_hi"] == 89.4
    assert row["m1_star_cov90_lo"] == 0.990 and row["m1_star_cov90_hi"] == 0.992
    # a non-spread metric is still mean-aggregated but carries no _lo/_hi column
    assert row["m1_star_cov50"] == 0.91
    assert "m1_star_cov50_lo" not in row
    # constant integer-like field (n_members) means cleanly to itself
    assert row["iid_n_members"] == 50


def test_faithfulness_aggregate_needs_two_inits():
    m = _load("run_forecast_faithfulness")
    assert m.aggregate_converged_faith([("pA", [_faith_row("2023-11-01T00:00", 8, 81.3, 0.99)])]) == []


# ----------------------------------------- F1: artifact-replay over committed CSVs
def _read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _assert_init_mean_replays(rows):
    """Every `init_mean` row must reproduce mean/min/max of its own (run, step)
    single+init group, with n_inits = group size. Grouping is keyed by run so the
    converged aggregate never picks up the 6ep context row. Catches a partially
    regenerated CSV where the aggregate drifts from the per-init rows."""
    group = {}
    for r in rows:
        if r.get("kind") in ("single", "init"):
            group.setdefault((r["run"], r["step"]), []).append(r)
    means = [r for r in rows if r.get("kind") == "init_mean"]
    assert means, "no init_mean rows in CSV"
    for mrow in means:
        g = group.get((mrow["run"], mrow["step"]))
        assert g, f"init_mean ({mrow['run']},{mrow['step']}) has no single/init group"
        assert int(mrow["n_inits"]) == len(g)
        for col in list(mrow):
            lo, hi = f"{col}_lo", f"{col}_hi"
            if col.endswith(("_lo", "_hi")) or lo not in mrow or hi not in mrow:
                continue
            try:
                vals = [float(r[col]) for r in g]
            except (ValueError, KeyError, TypeError):
                continue
            assert float(mrow[col]) == pytest.approx(sum(vals) / len(vals), abs=1e-9), col
            assert float(mrow[lo]) == pytest.approx(min(vals), abs=1e-12), lo
            assert float(mrow[hi]) == pytest.approx(max(vals), abs=1e-12), hi
    return len(means)


@pytest.mark.skipif(not SOFT_CSV.exists(), reason="softening_by_lead.csv absent")
def test_softening_csv_init_mean_replays_from_per_init_rows():
    assert _assert_init_mean_replays(_read_csv(SOFT_CSV)) >= 1


@pytest.mark.skipif(not FAITH_CSV.exists(), reason="faithfulness_by_lead.csv absent")
def test_faithfulness_csv_init_mean_replays_from_per_init_rows():
    assert _assert_init_mean_replays(_read_csv(FAITH_CSV)) >= 1


# --------------------------------- F3: per-init lambda* reproduces from the sweep
@pytest.mark.skipif(not (FC_RUN / "method1_sweep.npz").exists(),
                    reason="converged forecast sweep absent")
def test_per_init_lambda_star_reproduces_from_sweep():
    """Replay the persisted method1_sweep.npz + lambda_star.json through
    select_lambda_star and assert the recorded lambda* (its target, crossing and
    log-lambda interpolation) reproduces from the sweep arrays alone."""
    star = json.loads((FC_RUN / "lambda_star.json").read_text())
    with np.load(FC_RUN / "method1_sweep.npz") as f:
        sel = select_lambda_star(
            f["lambdas"], f["r_tilde"], f["variance_collapsed"],
            float(star["target_r_tilde"]), valid=f["warm_start_sane"],
        )
    assert sel.lambda_star == pytest.approx(star["lambda_star"], rel=1e-6)
    assert sel.bracketed == star["bracketed"]
    assert sel.matched_r_tilde == pytest.approx(star["matched_r_tilde"], rel=1e-6)
