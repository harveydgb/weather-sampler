"""Executive-summary figure trio (report/summary.tex needs larger-type figures).

Three NEW figures sized for the executive summary's two-column layout, where
each is displayed at roughly 3.4 in column width -- so every title, tick, and
colourbar label here is set noticeably larger than in the main-text Chapter
4/5 figures. Reuses the same persisted +48 h, 14-epoch Phase-4 run artifacts
as the main-text maps (no new sampling, no network, deterministic):

  summary_hook.png       Two-panel regional close-up: an independent draw at
                          each location vs. the ERA5 field it should
                          resemble. Same fields/region/colour-scale
                          convention as `scripts/make_hook_figure.py`'s
                          `make_region` (re-implemented here rather than
                          imported -- scripts are entry points, not an
                          importable library -- see that function for the
                          original).
  summary_fields.png     2x2 regional grid, one shared colour scale: the
                          independent draw, the Joint MAP field at the run's
                          selected weight, the Smoothed MAP (n=10) baseline,
                          and ERA5. These are 4 of the 6 panels
                          `scripts/make_real_figures.py`'s
                          `_method_map_fields`/`_plot_region_fields` build
                          for `phase_4_region_maps.png`; re-implemented here
                          (not imported, same reasoning as above) with larger
                          fonts and only the 4 panels this summary needs.
  summary_softening.png  One-hot fraction (share of locations whose largest
                          mixture weight exceeds 0.9) vs. lead time, 14-epoch
                          model only, +6 h .. +48 h. Recomputes the same
                          per-lead metric
                          (`sampler_research.forecast_diag.softening_metrics`)
                          over the same per-lead npz files that
                          `scripts/run_forecast_softening.py`'s
                          `softening_rows` reads for its "single" (canonical
                          init A) 14-epoch rows -- recomputed from the npz
                          rather than parsed from that script's persisted CSV
                          so this script has no ordering dependency on it.
                          Cross-checked at the end against the report macros
                          \\oneHotSixHourConverged / \\oneHotFortyEightConverged
                          (67.9% / 9.3%); a mismatch prints a warning rather
                          than failing silently.

Data sources (all pre-existing, read-only):
  outputs/data/phase_4_real_2t.npz                              latlons
  outputs/runs/phase_4_fc48_14ep_step8/anchors.npz               iid_seed0,
                                                                  smoothed_map_n10
  outputs/runs/phase_4_fc48_14ep_step8/method1_sensitivity.npz   field_star
                                                                  (Joint MAP
                                                                  @ lambda*)
  outputs/runs/phase_4_fc48_14ep_step8/era5_reference.npz        era5
  outputs/data/phase_4_fc48_14ep_step{1..8}_2t.npz(_meta.json)   per-lead
                                                                  GMMs for the
                                                                  softening
                                                                  curve

Run:
    .venv/bin/python scripts/make_summary_figures.py

Writes outputs/figures/summary/{summary_hook,summary_fields,summary_softening}.png.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sampler_research.forecast_diag import softening_metrics
from sampler_research.io import load_real_marginal
from sampler_research.plotting import (
    _latitude_marker_sizes,
    _natural_earth_coastline_segments,
    _wrap_longitudes,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_NPZ = REPO_ROOT / "outputs" / "data" / "phase_4_real_2t.npz"
RUN_DIR = REPO_ROOT / "outputs" / "runs" / "phase_4_fc48_14ep_step8"
SOFTENING_DATA_DIR = REPO_ROOT / "outputs" / "data"
SOFTENING_PREFIX = "phase_4_fc48_14ep"
SOFTENING_STEPS = range(1, 9)  # +6h .. +48h, one npz per lead step
OUT_DIR = REPO_ROOT / "outputs" / "figures" / "summary"

# Report macro values these figures must reproduce (see
# report/construction/macros-results.tex \oneHotSixHourConverged /
# \oneHotFortyEightConverged).
EXPECTED_ONE_HOT_SIX_HOUR = 67.9
EXPECTED_ONE_HOT_FORTY_EIGHT_HOUR = 9.3

# Executive-summary panels render at ~3.4 in column width. What matters for
# legibility is not the point size alone but its ratio to the figure's own
# (native) width, since the whole raster gets scaled down to fit the column:
# effective on-page size = fontsize_pt * (3.4 / native_figure_width_in). The
# full-page report figures pair ~10-16pt fonts (make_real_figures.py
# FIG_TITLE_FONTSIZE=16 etc.) with a ~10.8in native width shown at roughly a
# report page's text width (little shrink). Here the `figsize=` values below
# are instead kept close to (roughly 2x) the 3.4in target -- much narrower
# than the report figures -- *and* paired with larger point sizes, so the
# text-to-figure-width ratio, and hence the shrunk-to-column-width result,
# comes out much larger than the report figures' effective on-page size.
FIG_TITLE_FONTSIZE = 18
PANEL_TITLE_FONTSIZE = 15
AXIS_LABEL_FONTSIZE = 13
TICK_LABEL_FONTSIZE = 12

# Same North Atlantic / Europe / North Africa crop as Figure 1.1
# (make_hook_figure.py LON_MIN/LAT_MIN.. and make_real_figures.py
# REGION_LON_MIN/REGION_LAT_MIN..).
REGION_LON_MIN, REGION_LON_MAX = -60.0, 40.0
REGION_LAT_MIN, REGION_LAT_MAX = 20.0, 75.0

# Regional point size, tuned for the smaller (column-scale) panels below --
# larger than make_hook_figure.py's REGION_MARKER_BASE_SIZE (14.0 * 1.418)
# would give on a full-page panel, but not so large the dots merge into a
# solid mass at this figure's much smaller natural panel size.
REGION_MARKER_BASE_SIZE = 22.0

COLORBAR_LABEL = "2-metre temperature (standardised)"
TITLE_INDEPENDENT = "Sampling each location separately"
TITLE_ERA5 = "ERA5, what actually occurred"
TITLE_JOINT_MAP = "Joint MAP, the sampler built here"
TITLE_SMOOTHED = "Smoothed baseline, same smoothness"


def _load_latlons():
    return load_real_marginal(DATA_NPZ)["latlons"]


def _region_mask(latlons):
    """Boolean mask + wrapped lon/lat for the Figure-1.1 region.

    Mirrors `_region_coords` in scripts/make_real_figures.py.
    """

    lats = latlons[:, 0]
    lons = _wrap_longitudes(latlons[:, 1])
    box = (
        (lons >= REGION_LON_MIN) & (lons <= REGION_LON_MAX)
        & (lats >= REGION_LAT_MIN) & (lats <= REGION_LAT_MAX)
    )
    if not np.any(box):
        raise ValueError("executive-summary region contains no grid cells")
    return box, lons[box], lats[box]


def _check_shape(name, arr, n):
    if arr.shape != (n,):
        raise ValueError(f"{name} shape {arr.shape} != ({n},)")


def _shared_scale(field_arrays, box):
    """Symmetric diverging (2/98 percentile) colour limits over several fields.

    Mirrors the vmin/vmax construction in make_hook_figure.py's `make_region`
    and make_real_figures.py's `_plot_region_fields`.
    """

    values = np.concatenate(
        [np.asarray(field, dtype=float).reshape(-1)[box] for field in field_arrays]
    )
    p2, p98 = np.percentile(values, [2, 98])
    vspan = float(max(abs(p2), abs(p98)))
    if not np.isfinite(vspan) or vspan == 0.0:
        vspan = 1.0
    return -vspan, vspan


def _draw_panel(ax, lon_b, lat_b, values_b, marker_sizes, vmin, vmax, title):
    sc = ax.scatter(
        lon_b, lat_b, c=values_b, s=marker_sizes, cmap="RdBu_r",
        vmin=vmin, vmax=vmax, rasterized=True, zorder=2,
    )
    for seg_lon, seg_lat in _natural_earth_coastline_segments():
        ax.plot(seg_lon, seg_lat, color="0.08", lw=1.1, alpha=0.85, zorder=3)
    ax.set_xlim(REGION_LON_MIN, REGION_LON_MAX)
    ax.set_ylim(REGION_LAT_MIN, REGION_LAT_MAX)
    aspect = 1.0 / np.cos(np.deg2rad(0.5 * (REGION_LAT_MIN + REGION_LAT_MAX)))
    ax.set_aspect(aspect)
    ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE, pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return sc


def fig_hook(box, lon_b, lat_b):
    """summary_hook.png: independent draw vs. ERA5, two panels, shared colourbar."""

    with np.load(RUN_DIR / "anchors.npz") as f:
        iid = f["iid_seed0"]
    with np.load(RUN_DIR / "era5_reference.npz") as f:
        era5 = f["era5"]
    _check_shape("iid_seed0", iid, box.shape[0])
    _check_shape("era5", era5, box.shape[0])

    vmin, vmax = _shared_scale([iid, era5], box)
    marker_sizes = _latitude_marker_sizes(lat_b, base_size=REGION_MARKER_BASE_SIZE)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
    sc = None
    for ax, values, title in zip(
        axes, (iid[box], era5[box]), (TITLE_INDEPENDENT, TITLE_ERA5)
    ):
        sc = _draw_panel(ax, lon_b, lat_b, values, marker_sizes, vmin, vmax, title)
    fig.subplots_adjust(left=0.02, right=0.86, bottom=0.03, top=0.87, wspace=0.12)
    # Height-matched colorbar (as in make_hook_figure.py): realise the
    # aspect-constrained geometry with a draw, then size the bar to the right
    # panel's actual box; bbox_inches="tight" keeps the rotated label unclipped.
    fig.canvas.draw()
    panel = axes[1].get_position()
    cbar_ax = fig.add_axes([0.875, panel.y0, 0.018, panel.height])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label(COLORBAR_LABEL, fontsize=AXIS_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)

    out = OUT_DIR / "summary_hook.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO_ROOT)}")


def fig_fields(box, lon_b, lat_b):
    """summary_fields.png: 2x2 method grid, one shared colourbar."""

    with np.load(RUN_DIR / "anchors.npz") as f:
        iid = f["iid_seed0"]
        smoothed = f["smoothed_map_n10"]
    with np.load(RUN_DIR / "method1_sensitivity.npz") as f:
        joint_map = f["field_star"]
    with np.load(RUN_DIR / "era5_reference.npz") as f:
        era5 = f["era5"]
    for name, arr in (
        ("iid_seed0", iid), ("smoothed_map_n10", smoothed),
        ("field_star", joint_map), ("era5", era5),
    ):
        _check_shape(name, arr, box.shape[0])

    panels = [
        (TITLE_INDEPENDENT, iid),
        (TITLE_JOINT_MAP, joint_map),
        (TITLE_SMOOTHED, smoothed),
        (TITLE_ERA5, era5),
    ]
    vmin, vmax = _shared_scale([values for _, values in panels], box)
    marker_sizes = _latitude_marker_sizes(lat_b, base_size=REGION_MARKER_BASE_SIZE)

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.6))
    sc = None
    for ax, (title, values) in zip(axes.reshape(-1), panels):
        sc = _draw_panel(ax, lon_b, lat_b, values[box], marker_sizes, vmin, vmax, title)
    # Aspect-locked panels centre inside their grid boxes, opening a wide blank
    # band between the rows; anchor the top row down and the bottom row up so
    # the rows meet at a gap set by hspace alone (sized for the row-2 titles).
    for ax in axes[0]:
        ax.set_anchor("S")
    for ax in axes[1]:
        ax.set_anchor("N")
    fig.subplots_adjust(left=0.02, right=0.87, bottom=0.02, top=0.94, wspace=0.16, hspace=0.2)
    # Height-matched colorbar spanning both diagram rows (see fig_hook):
    # realise the aspect-constrained geometry with a draw, then size the bar
    # from the bottom-right panel's base to the top-right panel's top.
    fig.canvas.draw()
    top_panel = axes[0][1].get_position()
    bottom_panel = axes[1][1].get_position()
    cbar_ax = fig.add_axes(
        [0.885, bottom_panel.y0, 0.018, top_panel.y1 - bottom_panel.y0]
    )
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label(COLORBAR_LABEL, fontsize=AXIS_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)

    out = OUT_DIR / "summary_fields.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO_ROOT)}")


def _softening_row(step):
    """(lead_hours, one_hot_fraction_pct) for one +Nh 14-epoch lead step.

    Re-implements the minimal slice of `softening_rows` in
    scripts/run_forecast_softening.py needed for the canonical (init A)
    14-epoch single-init rows: load the per-lead npz + its meta.json, compute
    `sampler_research.forecast_diag.softening_metrics`.
    """

    stem = f"{SOFTENING_PREFIX}_step{step}_2t"
    data = load_real_marginal(SOFTENING_DATA_DIR / f"{stem}.npz")
    meta = json.loads((SOFTENING_DATA_DIR / f"{stem}_meta.json").read_text())
    lead_hours = float(meta["lead_hours"])
    metrics = softening_metrics(data["pi"], data["mu"], data["sigma"])
    return lead_hours, 100.0 * metrics["one_hot_fraction"]


def fig_softening():
    """summary_softening.png: one-hot fraction (%) vs. lead time, 14-epoch only."""

    rows = sorted(_softening_row(step) for step in SOFTENING_STEPS)
    leads = [lead for lead, _ in rows]
    pct = [value for _, value in rows]

    six_hour = dict(rows)[6.0]
    forty_eight_hour = dict(rows)[48.0]
    if (
        round(six_hour, 1) != EXPECTED_ONE_HOT_SIX_HOUR
        or round(forty_eight_hour, 1) != EXPECTED_ONE_HOT_FORTY_EIGHT_HOUR
    ):
        print(
            "WARNING: recomputed one-hot fraction "
            f"({six_hour:.2f}% @6h, {forty_eight_hour:.2f}% @48h) does NOT match "
            f"report macros \\oneHotSixHourConverged/\\oneHotFortyEightConverged "
            f"({EXPECTED_ONE_HOT_SIX_HOUR}% / {EXPECTED_ONE_HOT_FORTY_EIGHT_HOUR}%)"
        )
    else:
        print(
            f"verified: one-hot fraction {six_hour:.2f}% @6h / {forty_eight_hour:.2f}% @48h "
            "matches \\oneHotSixHourConverged / \\oneHotFortyEightConverged "
            f"({EXPECTED_ONE_HOT_SIX_HOUR}% / {EXPECTED_ONE_HOT_FORTY_EIGHT_HOUR}%)"
        )

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.plot(leads, pct, "o-", color="tab:red", markersize=11, linewidth=2.8)
    ax.set_xlabel("Hours ahead", fontsize=AXIS_LABEL_FONTSIZE)
    # Wrapped across two lines: at this figure's narrow natural width, the
    # unwrapped label is wider than the whole canvas and gets clipped.
    ax.set_ylabel(
        "Locations effectively certain\nof one value (%)", fontsize=AXIS_LABEL_FONTSIZE
    )
    ax.set_title("Forecast certainty falls with range", fontsize=FIG_TITLE_FONTSIZE)
    ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    ax.set_xlim(0, 51)
    ax.set_ylim(0, max(pct) * 1.15)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = OUT_DIR / "summary_softening.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO_ROOT)}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    latlons = _load_latlons()
    box, lon_b, lat_b = _region_mask(latlons)
    fig_hook(box, lon_b, lat_b)
    fig_fields(box, lon_b, lat_b)
    fig_softening()


if __name__ == "__main__":
    main()
