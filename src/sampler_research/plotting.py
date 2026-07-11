"""Plotting helpers for the existing research notebooks."""

from functools import lru_cache
from pathlib import Path

import numpy as np


def plot_field_pair(
    field_a,
    field_b,
    titles,
    *,
    cmap: str = "viridis",
):
    import matplotlib.pyplot as plt

    vmin = min(np.nanmin(field_a), np.nanmin(field_b))
    vmax = max(np.nanmax(field_a), np.nanmax(field_b))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, field, title in zip(axes, [field_a, field_b], titles):
        im = ax.imshow(field, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig, axes


def plot_component_fields(component_fields, peak_offsets, extent):
    import matplotlib.pyplot as plt

    k = component_fields.shape[-1]
    vmin, vmax = component_fields.min(), component_fields.max()
    fig, axes = plt.subplots(1, k, figsize=(3 * k, 3.4))
    for idx, ax in enumerate(np.ravel(axes)):
        im = ax.imshow(
            component_fields[:, :, idx],
            origin="lower",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        a_k, b_k = peak_offsets[idx]
        ax.plot(-a_k, -b_k, "r+", markersize=12, markeredgewidth=2)
        ax.set_title(f"component {idx}\n(a, b) = ({a_k:g}, {b_k:g})")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.025)
    fig.suptitle("Four component fields (peak location varied), peak marked +", y=1.02)
    return fig, axes


def plot_mu_slices(mu):
    import matplotlib.pyplot as plt

    k = mu.shape[-1]
    vmin, vmax = mu.min(), mu.max()
    fig, axes = plt.subplots(1, k, figsize=(3 * k, 3.2))
    for idx, ax in enumerate(np.ravel(axes)):
        im = ax.imshow(mu[:, :, idx], origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(f"mu[:, :, {idx}]")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.025)
    return fig, axes


# Shared field -> (colour, marker, linestyle) map so the variogram (real space)
# and the spectrum (spectral space) read as the same diagnostic two ways: the iid
# white floor and the mixture-mean over-smooth field are the bracket; the sampler
# siblings sit between; the ERA5 reference is a thin grey dashed line, drawn last
# so it reads as a reference and not a series (direction-of-realism, NOT a target).
_FIELD_STYLE = {
    "iid_seed0": ("black", "^", "-"),            # over-noisy white floor (upper)
    "iid_seed1": ("black", "^", "-"),            # same baseline, extra seed
    "iid_seed2": ("black", "^", "-"),            # same baseline, extra seed
    "mode_map": ("black", "s", "--"),
    "smoothed_map_n5": ("black", "o", ":"),
    "smoothed_map_n10": ("black", "o", "--"),
    "smoothed_map_n20": ("black", "o", ":"),
    "mixture_mean": ("black", "D", "-"),         # over-smooth extreme (lower)
    "a_star": ("black", "*", "--"),              # smoothest-faithful bracket
    "m1_star": ("C0", "o", "-"),                 # Joint MAP @ lambda* (solid)
    "m1_budget": ("C0", "o", "--"),              # Joint MAP @ faithfulness budget
    #                                              (same blue as lambda*; DASHED
    #                                              distinguishes the operating point)
    "m1_star_half": ("C9", "o", "-"),            # Joint MAP @ half lambda*
    "m1_star_double": ("C4", "o", "-"),          # Joint MAP @ 2x lambda*
    "m4_beta1": ("C2", "o", "-"),                # Mode-selection MRF @ beta=1
    "tv_adam": ("C1", "o", "-"),                 # toy TV(Adam) Huber arm (Phase 2)
    "tv_cut": ("C3", "o", "-"),                  # toy TV(exact) min-cut frontier
}
ERA5_KEY = "era5"
_ERA5_STYLE = ("0.35", None, ":")
_FALLBACK_CYCLE = [
    ("cyan", "o", "-"), ("brown", "s", "--"),
    ("olive", "^", "-"), ("pink", "d", "--"),
]
_SPECTRUM_COLOUR = {
    # Independent draw in black -- the house-style iid colour used in the
    # variogram/Pareto -- NOT red: red next to the Pareto figure's blue Joint MAP
    # made the same hue mean two different things across the report (DEC-11.4).
    "iid_seed0": "k",
    "iid_seed1": "k",
    "iid_seed2": "k",
    "mode_map": "C4",
    "smoothed_map_n5": "C8",
    "smoothed_map_n10": "C8",
    "smoothed_map_n20": "C8",
    "mixture_mean": "C1",
    "a_star": "C9",
}
# The mid-band trio (Per-cell MAP / Mixture mean / Mode-selection MRF) is visually
# coincident on the spectrum; distinct dash patterns let a reader see all three
# curves are present even where they overprint (the caption also states this).
_SPECTRUM_LINESTYLE = {
    "mode_map": "--",
    "mixture_mean": "-.",
    "m4_beta1": ":",
    "smoothed_map_n10": "-",
}

# Canonical DISPLAY names: ONE source of truth for every legend label AND the
# matching term used in the report prose, so figures and text can never drift.
# The internal artifact keys (iid_seed0, m1_star, ...) stay unchanged across the
# CSV / macro pipeline; only the human-facing label is mapped here.
DISPLAY_NAME = {
    "iid_seed0": "Independent draw",
    "iid_seed1": "Independent draw (seed 1)",
    "iid_seed2": "Independent draw (seed 2)",
    "mode_map": "Per-cell MAP",
    "smoothed_map_n5": "Smoothed MAP (n=5)",
    "smoothed_map_n10": "Smoothed MAP",
    "smoothed_map_n20": "Smoothed MAP (n=20)",
    "mixture_mean": "Mixture mean",
    "a_star": "Smoothest faithful ($a^\\star$)",
    "m1_star": "Joint MAP",
    "m1_star_half": "Joint MAP ($\\tfrac12\\lambda^\\star$)",
    "m1_star_double": "Joint MAP ($2\\lambda^\\star$)",
    "m4_beta1": "Mode-selection MRF",
    "tv_adam": "TV (Adam)",
    "tv_cut": "TV (exact)",
    ERA5_KEY: "ERA5 (reference)",
}


def display_name(name):
    """Human-facing label for a field key (legend + report).

    Falls back to a humanised form of the key so an unmapped series never breaks
    a figure. Strips a trailing ``_seedN`` so per-seed iid draws collapse onto
    the one ``Independent draw`` label when several are plotted together.
    """
    if name in DISPLAY_NAME:
        return DISPLAY_NAME[name]
    return name.replace("_", " ")


def field_style(name, idx=0):
    """(colour, marker, linestyle) for a field, stable across variogram/spectrum."""
    if name == ERA5_KEY:
        return _ERA5_STYLE
    if name in _FIELD_STYLE:
        return _FIELD_STYLE[name]
    return _FALLBACK_CYCLE[idx % len(_FALLBACK_CYCLE)]


def _ordered_items(series):
    """Series in a stable order with the ERA5 reference drawn LAST (on top)."""
    keys = [k for k in series if k != ERA5_KEY]
    if ERA5_KEY in series:
        keys.append(ERA5_KEY)
    return [(k, series[k]) for k in keys]


def plot_variograms(
    centres,
    variograms,
    *,
    xlabel: str,
    title: str,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for idx, (label, values) in enumerate(_ordered_items(variograms)):
        colour, marker, ls = field_style(label, idx)
        if label == ERA5_KEY:
            ax.plot(centres, values, ls, color=colour, lw=1.0, label="ERA5 (reference)")
        else:
            ax.plot(centres, values, marker=marker, ls=ls, color=colour,
                    ms=3.5, lw=1.5, label=display_name(label))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("semivariance")
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig, ax


def plot_spectra(ell, spectra, *, title, era5_key=ERA5_KEY, labels=None,
                 figsize=(6.5, 3.8)):
    """Linear-x/log-y angular power spectrum C_l vs degree l for the headline fields.

    RUNG-3 BRACKET DIAGNOSTIC (not a skill metric, not an optimisation target):
    the iid draw is the over-noisy white floor (upper at high l), the mixture-mean
    field is the over-smooth extreme (lower), the sampler siblings sit between and
    below iid at high l. The ERA5 series (if present) is drawn as a THIN grey
    dashed reference line LAST -- direction-of-realism only, never a target. There
    is no GMM-derivable target spectrum (open problem). Returns
    ``(fig, ax)``.

    ``labels`` optionally overrides the legend text for specific keys (e.g. a
    parametrised ``Joint MAP (lambda=19)`` built at call time from the artifact),
    so a series can carry the same dynamic label as its panel in another figure
    without hard-coding it into ``DISPLAY_NAME``.
    """
    import matplotlib.pyplot as plt

    labels = labels or {}
    fig, ax = plt.subplots(figsize=figsize)
    for idx, (label, values) in enumerate(_ordered_items(spectra)):
        colour, _marker, ls = field_style(label, idx)
        colour = _SPECTRUM_COLOUR.get(label, colour)
        ls = _SPECTRUM_LINESTYLE.get(label, ls)
        # Plain lines (no per-point markers): colour encodes method, dash encodes
        # operating point; markers only crowd the ~191-point resolved band.
        if label == era5_key:
            ax.plot(ell, values, ls=ls, color=colour, lw=2.6,
                    label=labels.get(label, "ERA5 (reference)"))
        else:
            ax.plot(ell, values, ls=ls, color=colour, lw=2.2,
                    label=labels.get(label, display_name(label)))
    ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.set_xlabel("angular degree $\\ell$")
    ax.set_ylabel("$C_\\ell$ (resolved band)")
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig, ax


def plot_single_point_marginal(
    samples,
    x,
    total_pdf,
    component_pdfs,
    pi,
    *,
    point_index: int,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(samples, bins=100, density=True, alpha=0.55, color="steelblue", label=f"{len(samples):,} samples")
    ax.plot(x, total_pdf, "r-", lw=2, label="GMM PDF")
    for idx, component_pdf in enumerate(component_pdfs):
        ax.plot(x, component_pdf, "--", lw=1, alpha=0.55, label=f"comp {idx} (pi={pi[idx]:.2f})")
    ax.set_xlabel("2t (normalised)")
    ax.set_ylabel("density")
    ax.set_title(f"Single-point marginal (point {point_index})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig, ax


def plot_pooled_distribution(sample_iid, mean_field):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(sample_iid, bins=120, density=True, alpha=0.5, color="indianred", label="IID samples")
    ax.hist(mean_field, bins=120, density=True, alpha=0.5, color="steelblue", label="Mean field")
    ax.set_xlabel("2t (normalised)")
    ax.set_ylabel("density")
    ax.set_title("Pooled distribution across all grid points")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def _wrap_longitudes(lons):
    return ((np.asarray(lons, dtype=float) + 180.0) % 360.0) - 180.0


def _latitude_marker_sizes(lats_deg, base_size=14.0):
    """Latitude-dependent scatter marker areas for regional O96 point maps."""

    return float(base_size) * (0.65 + 0.7 * (np.abs(np.asarray(lats_deg)) / 90.0) ** 2)


@lru_cache(maxsize=1)
def _natural_earth_coastline_segments():
    """Natural Earth 110m coastline segments in wrapped lon/lat degrees."""
    coastline = (
        Path(__file__).resolve().parents[3]
        / "outputs" / "data" / "natural_earth" / "ne_110m_coastline.shp"
    )
    if not coastline.exists():
        return ()
    try:
        import shapefile
    except ImportError:
        return ()

    segments = []
    reader = shapefile.Reader(str(coastline))
    for shape in reader.shapes():
        points = np.asarray(shape.points, dtype=float)
        if points.size == 0:
            continue
        parts = list(shape.parts) + [len(points)]
        for start, stop in zip(parts[:-1], parts[1:]):
            part = points[start:stop]
            if len(part) < 2:
                continue
            lons = _wrap_longitudes(part[:, 0])
            lats = part[:, 1]
            breaks = np.where(np.abs(np.diff(lons)) > 180.0)[0] + 1
            for piece in np.split(np.column_stack([lons, lats]), breaks):
                if len(piece) >= 2:
                    segments.append((piece[:, 0], piece[:, 1]))
    return tuple(segments)


_ROBINSON_LAT = np.arange(0, 95, 5, dtype=float)
_ROBINSON_X = np.array([
    1.0000, 0.9986, 0.9954, 0.9900, 0.9822, 0.9730, 0.9600, 0.9427, 0.9216,
    0.8962, 0.8679, 0.8350, 0.7986, 0.7597, 0.7186, 0.6732, 0.6213, 0.5722,
    0.5322,
])
_ROBINSON_Y = np.array([
    0.0000, 0.0620, 0.1240, 0.1860, 0.2480, 0.3100, 0.3720, 0.4340, 0.4958,
    0.5571, 0.6176, 0.6769, 0.7346, 0.7903, 0.8435, 0.8936, 0.9394, 0.9761,
    1.0000,
])
_ROBINSON_X_SCALE = 0.8487
_ROBINSON_Y_SCALE = 1.3523


def _robinson_project(lons_deg, lats_deg):
    lons = _wrap_longitudes(lons_deg)
    lats = np.asarray(lats_deg, dtype=float)
    abs_lat = np.clip(np.abs(lats), 0.0, 90.0)
    x_coef = np.interp(abs_lat, _ROBINSON_LAT, _ROBINSON_X)
    y_coef = np.interp(abs_lat, _ROBINSON_LAT, _ROBINSON_Y)
    x = _ROBINSON_X_SCALE * x_coef * np.radians(lons)
    y = _ROBINSON_Y_SCALE * y_coef * np.sign(lats)
    return x, y


def _draw_robinson_frame(ax):
    grid_lons = np.linspace(-180.0, 180.0, 361)
    grid_lats = np.linspace(-90.0, 90.0, 181)
    for lat in np.arange(-60.0, 90.0, 30.0):
        x, y = _robinson_project(grid_lons, np.full_like(grid_lons, lat))
        ax.plot(x, y, color="0.55", lw=0.45, alpha=0.5, zorder=1)
    for lon in np.arange(-120.0, 180.0, 60.0):
        x, y = _robinson_project(np.full_like(grid_lats, lon), grid_lats)
        ax.plot(x, y, color="0.55", lw=0.45, alpha=0.5, zorder=1)

    top_x, top_y = _robinson_project(grid_lons, np.full_like(grid_lons, 90.0))
    bottom_x, bottom_y = _robinson_project(grid_lons, np.full_like(grid_lons, -90.0))
    left_x, left_y = _robinson_project(np.full_like(grid_lats, -180.0), grid_lats)
    right_x, right_y = _robinson_project(np.full_like(grid_lats, 180.0), grid_lats)
    for x, y in ((top_x, top_y), (right_x, right_y), (bottom_x, bottom_y), (left_x, left_y)):
        ax.plot(x, y, color="0.15", lw=0.85, alpha=0.85, zorder=4)


def _draw_robinson_coastlines(ax):
    for lons_deg, lats_deg in _natural_earth_coastline_segments():
        x, y = _robinson_project(lons_deg, lats_deg)
        ax.plot(x, y, color="0.08", lw=1.05, alpha=0.82, zorder=3)


def plot_mollweide_fields(
    latlons_deg,
    fields,
    *,
    suptitle: str,
    cmap: str = "RdBu_r",
    ncols: int = 3,
    suptitle_fontsize: float = 13,
    panel_title_fontsize: float | None = None,
    colorbar_label: str = "2-metre temperature (standardised)",
    colorbar_label_fontsize: float | None = None,
    colorbar_tick_fontsize: float | None = None,
    colorbar_shrink: float = 1.0,
    row_colorbars: bool = False,
    figsize: tuple[float, float] | None = None,
    suptitle_y: float = 0.98,
    panel_title_pad: float = 10,
    top: float = 0.89,
    hspace: float = 0.08,
):
    import matplotlib.pyplot as plt

    latlons = np.asarray(latlons_deg, dtype=float)
    lons_deg = _wrap_longitudes(latlons[:, 1])
    x_proj, y_proj = _robinson_project(lons_deg, latlons[:, 0])
    values_for_scale = np.concatenate([np.asarray(values).reshape(-1) for values in fields.values()])
    # Symmetric diverging norm: white sits at 0 on the standardised scale, so the
    # RdBu_r midpoint is meaningful rather than landing at an arbitrary value
    # (DEC-11.5). The 2/98 clip is preserved, then mirrored about zero.
    p2, p98 = np.percentile(values_for_scale, [2, 98])
    vspan = float(max(abs(p2), abs(p98)))
    vmin, vmax = -vspan, vspan
    # Larger markers toward the poles, where the reduced-Gaussian rings thin out;
    # fills the polar "corduroy" gaps without interpolating the field (DEC-11.5).
    marker_sizes = 0.5 + 2.0 * (np.abs(latlons[:, 0]) / 90.0) ** 2
    field_items = list(fields.items())
    ncols = min(max(1, int(ncols)), len(field_items))
    nrows = int(np.ceil(len(field_items) / ncols))

    fig = plt.figure(figsize=figsize or (4.7 * ncols + 0.55, 2.85 * nrows + 0.25))
    grid = fig.add_gridspec(
        nrows,
        ncols + 1,
        width_ratios=[1.0] * ncols + [0.055],
        wspace=0.08,
        hspace=hspace,
    )
    axes_grid = np.empty((nrows, ncols), dtype=object)
    for row in range(nrows):
        for col in range(ncols):
            axes_grid[row, col] = fig.add_subplot(grid[row, col])
    cbar_axes = [
        fig.add_subplot(grid[row, -1])
        for row in range(nrows if row_colorbars else 1)
    ]
    sc = None
    for idx, (title, values) in enumerate(field_items):
        row, col = divmod(idx, ncols)
        ax = axes_grid[row, col]
        _draw_robinson_frame(ax)
        sc = ax.scatter(
            x_proj,
            y_proj,
            c=np.asarray(values).reshape(-1),
            s=marker_sizes,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
            zorder=2,
        )
        _draw_robinson_coastlines(ax)
        if panel_title_fontsize is None:
            ax.set_title(title, pad=panel_title_pad)
        else:
            ax.set_title(title, pad=panel_title_pad, fontsize=panel_title_fontsize)
        ax.set_facecolor("0.96")
        ax.set_aspect("equal")
        ax.set_xlim(-2.75, 2.75)
        ax.set_ylim(-1.42, 1.42)
        ax.axis("off")
    for ax in axes_grid.reshape(-1)[len(field_items):]:
        ax.set_visible(False)
    fig.suptitle(suptitle, y=suptitle_y, fontsize=suptitle_fontsize)
    fig.subplots_adjust(left=0.035, right=0.965, bottom=0.06, top=top)
    if colorbar_shrink < 1.0:
        for cbar_ax in cbar_axes:
            pos = cbar_ax.get_position()
            new_height = pos.height * colorbar_shrink
            cbar_ax.set_position([
                pos.x0,
                pos.y0 + 0.5 * (pos.height - new_height),
                pos.width,
                new_height,
            ])
    for cbar_ax in cbar_axes:
        cbar = fig.colorbar(sc, cax=cbar_ax, orientation="vertical",
                            label=colorbar_label)
        if colorbar_label_fontsize is not None:
            cbar.set_label(colorbar_label, fontsize=colorbar_label_fontsize)
        if colorbar_tick_fontsize is not None:
            cbar.ax.tick_params(labelsize=colorbar_tick_fontsize)
    return fig, axes_grid


# --- Sweep-direction arrows and parameter-value labels -----------------------
# Shared by the toy Phase-2 figures (scripts/make_toy_figures.py) and the real
# Phase-4 figures (scripts/make_real_figures.py) so a parameter sweep reads in
# the same house style in both: outline-chevron direction arrows along the curve
# and first / best / last parameter-value labels on the sweep circles.

SWEEP_ARROW_SIZE = 50

# Open right-pointing chevron (">") used for the sweep-direction arrows. The
# apex sits at (1, 0) and the two arms reach back to (-1, +/-CHEVRON_HALF_WIDTH).
# Tip-to-tail LENGTH is the x-extent (fixed at +/-1, scaled by `s`); WIDTH is
# 2*CHEVRON_HALF_WIDTH -- LARGER = wider, SMALLER = narrower/pointier.
CHEVRON_HALF_WIDTH = 0.7


@lru_cache(maxsize=1)
def _chevron_path():
    from matplotlib.path import Path as MplPath

    return MplPath(
        [[-1.0, CHEVRON_HALF_WIDTH], [1.0, 0.0], [-1.0, -CHEVRON_HALF_WIDTH]]
    )


def _segment_arrows(
    ax, x, y, colour, *, label=None, label_segment=None, text_xy=(0, 8),
    mutation_scale=12, n_arrows=None, arrow_segments=None, outline=False,
    arrow_size=None, arrow_linewidth=1.0,
):
    """Overlay direction arrowheads along a sweep.

    By default an arrowhead is drawn on every sweep point (pointing to the next
    point). If ``n_arrows`` is given, only that many arrowheads are placed --
    spaced evenly along the valid segments and sitting at each segment's
    midpoint -- so the per-point circle markers from the underlying ``o-`` line
    stay visible. ``arrow_segments`` can pin arrowheads to specific zero-based
    segment indices for hand-tuned report figures. ``outline=True`` renders the
    arrowheads unfilled (coloured outline only) instead of solid blocks.
    ``arrow_linewidth`` controls the outline stroke width.
    """
    from matplotlib.markers import MarkerStyle
    from matplotlib.transforms import Affine2D

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # Per-call arrowhead size; defaults to the module SWEEP_ARROW_SIZE so existing
    # callers (e.g. the Ch5 phase-4 figures) are unchanged, while the toy figures
    # can request a slightly larger head.
    base_size = SWEEP_ARROW_SIZE if arrow_size is None else arrow_size

    valid_segments = []
    ax.figure.canvas.draw()
    for i in range(len(x) - 1):
        if not np.all(np.isfinite([x[i], x[i + 1], y[i], y[i + 1]])):
            continue
        valid_segments.append(i)

    if arrow_segments is not None:
        draw_segments = [int(i) for i in arrow_segments if int(i) in valid_segments]
        at_midpoint = True
    elif n_arrows is None:
        draw_segments = valid_segments
        at_midpoint = False
    elif not valid_segments or n_arrows >= len(valid_segments):
        draw_segments = valid_segments
        at_midpoint = True
    else:
        picks = np.linspace(0, len(valid_segments) - 1, n_arrows + 2)[1:-1]
        draw_segments = [valid_segments[int(round(j))] for j in picks]
        at_midpoint = True

    for i in draw_segments:
        p0 = ax.transData.transform((x[i], y[i]))
        p1 = ax.transData.transform((x[i + 1], y[i + 1]))
        delta = p1 - p0
        if np.allclose(delta, 0):
            continue
        angle = float(np.degrees(np.arctan2(delta[1], delta[0])))
        # Outline arrows use a custom open chevron path (two strokes meeting at
        # a point, like an inequality ">", no closing base line) whose apex
        # angle is tunable via CHEVRON_HALF_WIDTH. The filled ">" triangle is
        # kept for the solid per-point arrows.
        glyph = _chevron_path() if outline else ">"
        marker = MarkerStyle(glyph, transform=Affine2D().rotate_deg(angle))
        px = 0.5 * (x[i] + x[i + 1]) if at_midpoint else x[i]
        py = 0.5 * (y[i] + y[i + 1]) if at_midpoint else y[i]
        if outline:
            # facecolors="none" leaves the open path unfilled, so only the two
            # arms of the chevron are stroked (in `edgecolors`).
            ax.scatter(
                [px], [py], marker=marker, s=base_size * 1.5,
                facecolors="none", edgecolors=colour, linewidths=arrow_linewidth,
                zorder=10, clip_on=True,
            )
        else:
            ax.scatter(
                [px], [py], marker=marker, s=base_size,
                color=colour, edgecolors="white", linewidths=0.35, zorder=10,
                clip_on=True,
            )

    if label and valid_segments:
        i = label_segment if label_segment is not None else valid_segments[len(valid_segments) // 2]
        if i not in valid_segments:
            i = valid_segments[len(valid_segments) // 2]
        mid_x = float(np.sqrt(x[i] * x[i + 1])) if x[i] > 0 and x[i + 1] > 0 else float(0.5 * (x[i] + x[i + 1]))
        mid_y = float(0.5 * (y[i] + y[i + 1]))
        ax.annotate(
            label, (mid_x, mid_y), fontsize=7, color=colour,
            xytext=text_xy, textcoords="offset points", ha="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.8),
        )


def _knee_index(r_tilde, nll):
    """Index of the Pareto 'best' point: the sweep entry closest to the ideal
    lower-left corner (low roughness AND low NLL) in min-max-normalised
    (R~, NLL/N) space. Computed from the faithfulness-coherence objective so
    the same parameter value is flagged on whichever panel it is drawn."""
    r = np.asarray(r_tilde, dtype=float)
    n = np.asarray(nll, dtype=float)

    def _unit(a):
        span = float(a.max() - a.min())
        return (a - a.min()) / span if span > 0 else np.zeros_like(a)

    return int(np.argmin(_unit(r) ** 2 + _unit(n) ** 2))


def _sweep_param_labels(ax, x, y, params, best_idx, symbol, colour, placements):
    """Annotate the first, 'best' and last sweep circles with their parameter
    value. The best point is bolded. `placements` maps 'first'/'best'/'last' ->
    dict(xytext=(dx, dy), ha=..., va=...) of point-offset text placements,
    hand-tuned to dodge the curves, arrows and anchors."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    roles = {"first": 0, "best": int(best_idx), "last": len(x) - 1}
    for role, i in roles.items():
        txt = f"${symbol}={params[i]:g}$"
        p = placements[role]
        ax.annotate(
            txt, (x[i], y[i]), fontsize=7, color=colour,
            fontweight="bold" if role == "best" else "normal",
            xytext=p["xytext"], textcoords="offset points",
            ha=p.get("ha", "center"), va=p.get("va", "center"),
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.8),
            zorder=11,
        )
