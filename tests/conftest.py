"""Shared test path constants and artifact-presence skip markers (audit E3).

Centralises the repo-root / `outputs` path constants and the
`needs_real`/`needs_toy`/`needs_phase4_run`/`needs_ducc0` skip markers that were
previously re-declared in each test module. This is a pure test-support refactor: the
values here are character-identical to the per-file declarations they replace,
so collection and skip behaviour are unchanged.

Sibling test modules import directly, e.g. `from conftest import REPO_ROOT,
needs_real`. pytest's default ("prepend") import mode puts this directory on
`sys.path`, and there is no `tests/__init__.py`, so the bare `conftest` import
resolves. `REPO_ROOT` is `parents[1]` of this file, which lives in the same
directory as the test modules, so it matches their previous local definition.
"""

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "outputs" / "data"
RUNS_DIR = REPO_ROOT / "outputs" / "runs"

REAL_NPZ = DATA_DIR / "phase_4_real_2t.npz"
TOY_NPZ = DATA_DIR / "phase_1_homoscedastic.npz"
PHASE4_RUN_DIR = RUNS_DIR / "phase_4_real"

needs_real = pytest.mark.skipif(not REAL_NPZ.exists(), reason="real Phase 4 npz absent")
needs_toy = pytest.mark.skipif(not TOY_NPZ.exists(), reason="phase 1 toy npz absent")
needs_phase4_run = pytest.mark.skipif(
    not (PHASE4_RUN_DIR / "method1_sweep.npz").exists(),
    reason="persisted phase_4_real run artifacts absent",
)
needs_ducc0 = pytest.mark.skipif(
    importlib.util.find_spec("ducc0") is None,
    reason="ducc0 not installed (optional 'spectral' extra)",
)
