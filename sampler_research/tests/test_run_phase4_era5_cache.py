"""The scores stage must consume a pre-computed era5_reference.npz from the run
dir (dumped under the WeatherGenerator venv) without needing anemoi/zarr in the
sampler venv -- this is the bridge that keeps the anemoi dependency isolated
while still landing the ERA5 rung-3 direction reference on the spectrum/variogram.
"""

from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "run_phase4_real.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_phase4_real", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cached_era5_is_loaded_without_anemoi(tmp_path, capsys):
    runner = _load_runner()
    latlons = np.zeros((40320, 2))
    z = np.random.default_rng(0).normal(size=40320)
    np.savez(tmp_path / "era5_reference.npz", era5=z)

    out = runner._maybe_load_era5_reference(
        regime={"valid_datetime": "2023-11-03T00:00:00"},
        latlons=latlons, out_dir=tmp_path, args=SimpleNamespace(no_era5=False),
    )
    assert out is not None
    assert np.array_equal(out, z)
    assert "loaded from cache" in capsys.readouterr().out


def test_cached_era5_wrong_length_is_skipped(tmp_path, capsys):
    runner = _load_runner()
    np.savez(tmp_path / "era5_reference.npz", era5=np.zeros(5))  # wrong N
    out = runner._maybe_load_era5_reference(
        regime={"valid_datetime": "2023-11-03T00:00:00"},
        latlons=np.zeros((40320, 2)), out_dir=tmp_path, args=SimpleNamespace(no_era5=False),
    )
    assert out is None
    assert "skipped" in capsys.readouterr().out


def test_no_era5_flag_short_circuits_cache(tmp_path):
    runner = _load_runner()
    np.savez(tmp_path / "era5_reference.npz", era5=np.zeros(40320))
    out = runner._maybe_load_era5_reference(
        regime={}, latlons=np.zeros((40320, 2)), out_dir=tmp_path,
        args=SimpleNamespace(no_era5=True),
    )
    assert out is None
