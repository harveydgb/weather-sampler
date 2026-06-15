"""Forecast artifact validation + per-lead writer norm-stats carry (plan Steps 0/1).

Torch-free: `validate_forecast_dict` and `write_forecast_marginals` operate on
plain numpy dicts, so the contract that the WeatherGenerator-venv .pt must meet
is testable in the research venv.
"""

import json

import numpy as np
import pytest

from sampler_research.io import (
    load_real_marginal,
    validate_forecast_dict,
    write_forecast_marginals,
)


def _good_dict(n=50, k=4, n_steps=3, n_channels=8, ch_idx=3):
    rng = np.random.default_rng(0)
    steps = {}
    for s in range(1, n_steps + 1):
        pi = rng.dirichlet(np.ones(k), size=n)
        mu = rng.normal(size=(n, k))
        sigma = rng.uniform(0.2, 1.0, size=(n, k))
        latlons = rng.uniform(-90, 90, size=(n, 2))
        steps[s] = {"pi": pi, "mu_channel": mu, "sigma_channel": sigma,
                    "latlons": latlons, "lead_hours": 6.0 * s,
                    "valid_datetime": f"2023-11-0{s}T00:00:00.000"}
    return {
        "steps": steps,
        "target_datetime": "2023-11-01T00:00",
        "from_run_id": "gmm_test", "mini_epoch": 1, "ch_idx": ch_idx,
        "channel_of_interest": "2t", "forecast_steps": n_steps,
        "norm_mean": np.arange(n_channels, dtype=float) + 100.0,
        "norm_std": np.arange(n_channels, dtype=float) + 1.0,
        "norm_stats_source": "test",
    }


def test_validate_good_dict_has_no_problems():
    report, problems = validate_forecast_dict(_good_dict(), expected_n_steps=3, expected_ch_idx=3)
    assert problems == []
    assert report["n_steps"] == 3
    assert report["n_cells"] == 50 and report["n_components"] == 4
    assert report["lead_hours"] == [6.0, 12.0, 18.0]
    assert report["has_norm_stats"] is True
    assert report["norm_mean_channel"] == pytest.approx(103.0)
    assert report["norm_std_channel"] == pytest.approx(4.0)


def test_validate_flags_missing_steps():
    _, problems = validate_forecast_dict({"target_datetime": "x"})
    assert any("steps" in p for p in problems)


def test_validate_flags_noncontiguous_steps():
    d = _good_dict(n_steps=3)
    d["steps"] = {1: d["steps"][1], 3: d["steps"][3]}  # gap at 2
    _, problems = validate_forecast_dict(d)
    assert any("contiguous" in p for p in problems)


def test_validate_flags_bad_pi_and_sigma():
    d = _good_dict(n_steps=1)
    d["steps"][1]["pi"] = d["steps"][1]["pi"] * 2.0  # rows no longer sum to 1
    d["steps"][1]["sigma_channel"][0, 0] = -1.0
    _, problems = validate_forecast_dict(d)
    assert any("sum to 1" in p for p in problems)
    assert any("non-positive" in p for p in problems)


def test_validate_flags_missing_norm_stats_and_expectations():
    d = _good_dict(n_steps=2)
    del d["norm_mean"]
    _, problems = validate_forecast_dict(d, expected_n_steps=8, expected_ch_idx=0)
    assert any("norm_mean" in p for p in problems)
    assert any("8 steps" in p for p in problems)
    assert any("ch_idx" in p for p in problems)


def test_validate_flags_nonmonotone_leads():
    d = _good_dict(n_steps=3)
    d["steps"][2]["lead_hours"] = 99.0  # breaks monotonicity (6, 99, 18)
    _, problems = validate_forecast_dict(d)
    assert any("increasing" in p for p in problems)


def test_write_forecast_marginals_carries_norm_stats(tmp_path):
    written = write_forecast_marginals(_good_dict(n_steps=2), tmp_path, prefix="phase_4_fctest")
    assert len(written) == 2
    meta = json.loads((tmp_path / "phase_4_fctest_step1_2t_meta.json").read_text())
    assert meta["norm_mean_channel"] == pytest.approx(103.0)
    assert meta["norm_std_channel"] == pytest.approx(4.0)
    assert meta["lead_hours"] == 6.0
    assert meta["from_run_id"] == "gmm_test"
    # round-trips through the AE marginal loader
    data = load_real_marginal(written[0])
    assert data["pi"].shape == (50, 4)


def test_write_forecast_marginals_distinct_prefixes_no_collision(tmp_path):
    write_forecast_marginals(_good_dict(n_steps=2), tmp_path, prefix="phase_4_fc48_6ep")
    write_forecast_marginals(_good_dict(n_steps=2), tmp_path, prefix="phase_4_fc48_14ep")
    names = {p.name for p in tmp_path.glob("*.npz")}
    assert names == {
        "phase_4_fc48_6ep_step1_2t.npz", "phase_4_fc48_6ep_step2_2t.npz",
        "phase_4_fc48_14ep_step1_2t.npz", "phase_4_fc48_14ep_step2_2t.npz",
    }


def test_write_forecast_marginals_guards_prefix_collision_across_runs(tmp_path):
    """F2: a second run reusing the same prefix must not silently clobber the
    first run's per-lead npz (the runner defaults to a fixed prefix regardless
    of --forecast-pt). The guard keys on the (from_run_id, init_datetime) pair."""
    v1 = _good_dict(n_steps=2)
    v1["from_run_id"] = "gmm_fc48_v1"
    write_forecast_marginals(v1, tmp_path, prefix="phase_4_fc48_6ep")

    # Same prefix, different source run -> refuse to overwrite.
    v2 = _good_dict(n_steps=2)
    v2["from_run_id"] = "gmm_fc48_v2"
    with pytest.raises(FileExistsError, match="from_run_id"):
        write_forecast_marginals(v2, tmp_path, prefix="phase_4_fc48_6ep")
    # The v1 provenance on disk is untouched after the refused write.
    meta = json.loads((tmp_path / "phase_4_fc48_6ep_step1_2t_meta.json").read_text())
    assert meta["from_run_id"] == "gmm_fc48_v1"

    # Re-converting the SAME run into the same prefix is an allowed overwrite.
    written = write_forecast_marginals(v1, tmp_path, prefix="phase_4_fc48_6ep")
    assert len(written) == 2


def test_write_forecast_marginals_guards_init_collision_same_run(tmp_path):
    """F1 track-2: the multi-init replicates of one trained model share
    `from_run_id` and differ only by `init_datetime`, so the guard must key on
    the (from_run_id, init_datetime) pair -- a run-id-only guard would let a
    second init silently clobber the first under a shared prefix."""
    init_a = _good_dict(n_steps=2)
    init_a["from_run_id"] = "gmm_fc48_v2"
    init_a["target_datetime"] = "2023-11-01T00:00"
    write_forecast_marginals(init_a, tmp_path, prefix="phase_4_fc48_14ep")

    # Same prefix, SAME run id, DIFFERENT init date -> refuse to overwrite.
    init_b = _good_dict(n_steps=2)
    init_b["from_run_id"] = "gmm_fc48_v2"
    init_b["target_datetime"] = "2023-10-10T12:00"
    with pytest.raises(FileExistsError, match="init_datetime"):
        write_forecast_marginals(init_b, tmp_path, prefix="phase_4_fc48_14ep")
    # init A's provenance on disk is untouched after the refused write.
    meta = json.loads((tmp_path / "phase_4_fc48_14ep_step1_2t_meta.json").read_text())
    assert meta["init_datetime"] == "2023-11-01T00:00"

    # init B under its OWN distinct prefix is fine (no collision).
    written_b = write_forecast_marginals(init_b, tmp_path, prefix="phase_4_fc48_v2_init20231010T12")
    assert len(written_b) == 2
    # Re-converting the SAME (run, init) into the same prefix is idempotent.
    written_a = write_forecast_marginals(init_a, tmp_path, prefix="phase_4_fc48_14ep")
    assert len(written_a) == 2
