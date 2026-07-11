"""Regression checks for the Chapter-1 hook figure render."""

from __future__ import annotations

import importlib.util
import sys

import pytest

from conftest import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "make_hook_figure.py"


def _load_hook_figures():
    spec = importlib.util.spec_from_file_location("make_hook_figure", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_hook_figure_refuses_to_render_without_coastlines(monkeypatch):
    hook = _load_hook_figures()
    monkeypatch.setattr(hook, "_natural_earth_coastline_segments", lambda: ())

    with pytest.raises(RuntimeError, match="Natural Earth coastline layer unavailable"):
        hook._required_coastline_segments()


@pytest.mark.skipif(
    importlib.util.find_spec("shapefile") is None,
    reason="pyshp is required to read the bundled Natural Earth coastlines",
)
def test_hook_figure_loads_bundled_coastlines():
    hook = _load_hook_figures()

    assert len(hook._required_coastline_segments()) > 0
