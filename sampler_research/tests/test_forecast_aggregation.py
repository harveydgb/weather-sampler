"""Across-init aggregation helpers in the forecast runners (F1 track-2).

Pure functions over row-dict fixtures (no npz / run-dir I/O): the
`kind='init_mean'` rows carry the across-init MEAN in the base metric column and
the min-max RANGE in the `{metric}_lo`/`{metric}_hi` columns, over the inits
present at each lead step. Fewer than two inits => no aggregate (single-init
behaviour preserved).
"""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


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
