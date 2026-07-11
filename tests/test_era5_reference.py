"""Unit tests for the ERA5 reference loader's grid-alignment (cut criterion C4).

The ERA5 anemoi stream is the SAME O96 grid as the extracted samples but in a
different point order; `_match_permutation` must recover the exact reorder (no
interpolation) and still reject a genuinely different grid. These tests exercise
that logic directly via a fake dataset, so they need neither anemoi nor the zarr.
"""

from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "load_era5_reference.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("load_era5_reference", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _toy_grid():
    """A small multi-ring grid of distinct points: 4 rings x 5 longitudes."""
    lats = np.array([-40.0, -10.0, 10.0, 40.0])
    lons = np.array([0.0, 72.0, 144.0, 216.0, 288.0])
    grid = np.array([(la, lo) for la in lats for lo in lons], dtype=float)
    return grid  # [20, 2]


def _dataset(latlons, jitter=0.0):
    ll = np.asarray(latlons, dtype=float)
    if jitter:
        rng = np.random.default_rng(0)
        ll = ll + rng.uniform(-jitter, jitter, size=ll.shape)
    return SimpleNamespace(latitudes=ll[:, 0].copy(), longitudes=ll[:, 1].copy())


def test_match_permutation_recovers_exact_reorder():
    mod = _load_module()
    grid = _toy_grid()
    rng = np.random.default_rng(1)
    shuffle = rng.permutation(grid.shape[0])
    sample = grid  # sample row order
    ds = _dataset(grid[shuffle], jitter=1e-6)  # ERA5: same points, shuffled + jitter

    perm = mod._match_permutation(ds, sample)
    assert np.array_equal(np.sort(perm), np.arange(grid.shape[0]))  # bijection
    # ERA5 rows reordered by perm must coincide with the sample rows.
    era5_ll = np.column_stack([ds.latitudes, ds.longitudes])
    assert np.max(np.abs(era5_ll[perm, 0] - sample[:, 0])) < 1e-3
    assert np.max(np.abs(era5_ll[perm, 1] - sample[:, 1])) < 1e-3
    # A field carried on the ERA5 grid lands in sample order under perm.
    field = np.arange(grid.shape[0], dtype=float)[shuffle]  # value == sample index
    assert np.array_equal(field[perm], np.arange(grid.shape[0], dtype=float))


def test_match_permutation_rejects_different_grid():
    mod = _load_module()
    grid = _toy_grid()
    moved = grid.copy()
    moved[0, 0] += 5.0  # one point off any sample ring -> genuinely different grid
    ds = _dataset(moved)
    with pytest.raises(mod.Era5ReferenceUnavailable):
        mod._match_permutation(ds, grid)


def test_match_permutation_rejects_point_count_mismatch():
    mod = _load_module()
    grid = _toy_grid()
    ds = _dataset(grid[:-1])  # one fewer point
    with pytest.raises(mod.Era5ReferenceUnavailable):
        mod._match_permutation(ds, grid)
