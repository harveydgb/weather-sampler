"""Guard: the method/baseline DISPLAY-name rename cannot silently regress.

(1) `thesis.tex` carries no human-facing "Method 1/4/5" in RENDERED text (`%`
comments and the internal artifact keys are exempt). (2) the two method macros in
`macros.tex` expand to exactly the strings `sampler_research.plotting.DISPLAY_NAME`
uses, so the figure legends and the report prose cannot drift apart. The canonical
key -> display map is `sampler_research.plotting.DISPLAY_NAME`.
"""

from __future__ import annotations

import re

import pytest

from sampler_research.plotting import DISPLAY_NAME

from conftest import REPO_ROOT

THESIS = REPO_ROOT / "report" / "thesis.tex"
MACROS = REPO_ROOT / "report" / "construction" / "macros.tex"
TABLES_DIR = REPO_ROOT / "report" / "construction" / "tables"
TABLE_FILES = ["pi_softening.tex", "lambda_calibration.tex",
               "faithfulness.tex", "toy_baselines.tex"]

# Strip a LaTeX line comment (first unescaped %), leaving only rendered text.
_COMMENT = re.compile(r"(?<!\\)%.*$")
# Human-facing method labels the rename removed (Method 1 / Method~4 / Method-1 …).
_METHOD_N = re.compile(r"Method[~\s\-]*[145]\b")
# …and their bare abbreviations (M1 / M4 / M5) that the guard above misses; these
# leaked into the auto-generated table headers before the 29 Jun table review.
_METHOD_ABBR = re.compile(r"\bM[145]\b")


@pytest.mark.skipif(not THESIS.exists(), reason="report/thesis.tex absent")
def test_thesis_has_no_human_facing_method_numbers():
    hits = []
    for lineno, raw in enumerate(THESIS.read_text().splitlines(), 1):
        rendered = _COMMENT.sub("", raw)
        if _METHOD_N.search(rendered):
            hits.append(f"  {lineno}: {rendered.strip()}")
    assert not hits, (
        "human-facing 'Method 1/4/5' still in thesis.tex (use \\JointMAP/\\ModeMRF):\n"
        + "\n".join(hits))


@pytest.mark.parametrize("name", TABLE_FILES)
def test_table_files_have_no_human_facing_method_numbers(name):
    """The auto-generated result tables (\\input into thesis.tex, so missed by the
    thesis scan above) must carry no `Method n` / bare `Mn` in rendered text."""
    path = TABLES_DIR / name
    if not path.exists():
        pytest.skip(f"{name} absent")
    hits = []
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        rendered = _COMMENT.sub("", raw)
        if _METHOD_N.search(rendered) or _METHOD_ABBR.search(rendered):
            hits.append(f"  {lineno}: {rendered.strip()}")
    assert not hits, (
        f"human-facing method number/abbreviation in tables/{name} "
        "(use the locked display name, e.g. Joint MAP):\n" + "\n".join(hits))


@pytest.mark.skipif(not MACROS.exists(), reason="macros.tex absent")
@pytest.mark.parametrize("macro,key", [("JointMAP", "m1_star"), ("ModeMRF", "m4_beta1")])
def test_macro_matches_plotting_display_name(macro, key):
    """\\JointMAP / \\ModeMRF must expand to the same string the figures use."""
    text = MACROS.read_text()
    m = re.search(r"\\newcommand\{\\" + macro + r"\}\{([^}]*)\}", text)
    assert m, f"\\{macro} not defined in macros.tex"
    assert m.group(1) == DISPLAY_NAME[key], (
        f"\\{macro}={m.group(1)!r} != DISPLAY_NAME[{key!r}]={DISPLAY_NAME[key]!r} "
        "(macros.tex and plotting.DISPLAY_NAME have drifted)")
