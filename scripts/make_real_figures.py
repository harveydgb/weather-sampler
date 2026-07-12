"""Phase 4 report figures from the persisted runner artifacts.

Run after `scripts/run_real_eval.py` (all stages + --lambda-star). The
reconstruction-regime renders go under a `recon/` subdir so their short base
names never collide with the per-lead forecast renders, which the forecast
driver writes to `outputs/figures/forecast_<ep>_step<k>/` with the same base
names (regime is disambiguated by directory, not filename):

    .venv/bin/python scripts/make_real_figures.py --fig-dir outputs/figures/recon

Writes to the chosen --fig-dir (base names below):
  maps.png               4/5-panel global Mollweide
  pareto_smear.png       (NLL/N, R~) plane + smear-fraction panel
  variogram.png          sampled spherical variograms (descriptive;
                                 + ERA5 reference line if available)
  spectrum.png           native O96 angular power spectrum C_l vs l
                                 (rung-3 bracket; main-text coherence figure;
                                 + ERA5 direction reference if available)
  lambda_sweep_maps.png  Joint MAP maps for lambda=0 plus the first
                                 six positive lambda-sweep points, alongside
                                 the matched Smoothed MAP target when present
  region_maps.png        Figure-5.2 method panels over the Figure-1.1
                                 North Atlantic / Europe region
  region_lambda_sweep_maps.png
                                 Lambda-sweep panels over the Figure-1.1 region
  bimodal_enrichment.png W1 MUST: dNLL>0.125 enrichment in the audit
                                 S4 bimodal masks + Mollweide dNLL map
  robustness.png         unary-gap histogram (why the Mode-selection MRF beta
                                 sweep is pinned) + unit-vs-weighted-graph
                                 matched-coherence points (needs
                                 robustness_probes.json from
                                 scripts/run_real_probes.py)

Figures are generated here (not in notebook 04) so the MUST figure task does
not depend on notebook execution; the notebook displays these files.
`--only name [name ...]` regenerates a subset (maps / pareto / variogram /
spectrum / lambda_maps / region_maps / region_lambda_maps / enrichment /
robustness) without rewriting the other committed PNGs.
the other committed PNGs.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sampler_research.faithfulness import bootstrap_cell_statistic
from sampler_research.io import load_real_marginal
from sampler_research.plotting import (
    _draw_robinson_coastlines,
    _draw_robinson_frame,
    _latitude_marker_sizes,
    _natural_earth_coastline_segments,
    _robinson_project,
    _segment_arrows,
    _wrap_longitudes,
    display_name,
    field_style,
    plot_mollweide_fields,
    plot_spectra,
    plot_variograms,
)

# Within-field spatial-bootstrap settings, kept identical to the macro pipeline
# (scripts/emit_report_results.py BOOT_*) so the figure's bimodal smear-tail CIs
# match the \fcBimodalFrac* macros draw-for-draw.
BOOT_THRESHOLD = 0.125
BOOT_N_DRAWS = 2000
BOOT_SEED = 0
BOOT_CI = 0.95

FIG_TITLE_FONTSIZE = 16
PANEL_TITLE_FONTSIZE = 12.5
AXIS_LABEL_FONTSIZE = 12
TICK_LABEL_FONTSIZE = 10
LEGEND_FONTSIZE = 9
DENSE_LEGEND_FONTSIZE = 8
ANNOTATION_FONTSIZE = 8

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "outputs" / "runs" / "phase_4_real"
FIG_DIR = REPO_ROOT / "outputs" / "figures"
DATA_NPZ = REPO_ROOT / "outputs" / "data" / "phase_4_real_2t.npz"

# Same North Atlantic / Europe / North Africa crop as Figure 1.1
# (`scripts/make_hook_figure.py`).
REGION_LON_MIN, REGION_LON_MAX = -60.0, 40.0
REGION_LAT_MIN, REGION_LAT_MAX = 20.0, 75.0


def _load_rows():
    with open(RUN_DIR / "scores.csv") as fh:
        return list(csv.DictReader(fh))


def _bimodal_frac_ci(name):
    """Within-field spatial-bootstrap CI of frac(dNLL > 0.125) over the bimodal
    stratum for one field key, from this run's delta_per_cell.npz + masks.npz.

    Returns a ``bootstrap_cell_statistic`` record (``point``/``lo``/``hi``) or
    ``None`` if the artifacts/mask are absent. Uses the same draws/seed/threshold
    as the macro pipeline, so the figure and the \\fcBimodalFrac* macros agree.
    """

    delta_path = RUN_DIR / "delta_per_cell.npz"
    masks_path = RUN_DIR / "masks.npz"
    if not (delta_path.exists() and masks_path.exists()):
        return None
    with np.load(delta_path) as f:
        if name not in f.files:
            return None
        d = np.asarray(f[name], dtype=float)
    with np.load(masks_path) as f:
        if "bimodal" not in f.files:
            return None
        mask = np.asarray(f["bimodal"], dtype=bool)
    vals = d[mask]
    if vals.size == 0:
        return None
    return bootstrap_cell_statistic(
        vals, lambda v: float(np.mean(v > BOOT_THRESHOLD)),
        n_boot=BOOT_N_DRAWS, ci=BOOT_CI, rng=np.random.default_rng(BOOT_SEED),
    )


def _regime_label():
    """Human-readable regime/lead tag for figure suptitles, from the artifacts.

    Prefers provenance.json (written by the runner); falls back to lambda_star.json
    or the reconstruction-regime default for the committed step-0 run dir.
    """

    regime = None
    prov = RUN_DIR / "provenance.json"
    if prov.exists():
        regime = json.loads(prov.read_text()).get("regime")
    if regime is None:
        star_path = RUN_DIR / "lambda_star.json"
        if star_path.exists():
            regime = json.loads(star_path.read_text()).get("regime")
    if not isinstance(regime, dict):
        return "reconstruction regime (step 0)" if regime is None else str(regime)
    if regime.get("regime") == "forecast":
        return f"forecast regime (+{regime.get('lead_hours', '?')} h lead)"
    return "reconstruction regime (step 0)"


def _display_regime_label(*, compact=False):
    raw = _regime_label()
    if raw.startswith("forecast regime"):
        try:
            lead = raw.split("(+", 1)[1].split(" h", 1)[0]
            lead_txt = f"+{float(lead):g} h"
        except (IndexError, ValueError):
            lead_txt = raw
        return f"Forecast {lead_txt}" if compact else f"Forecast Regime ({lead_txt} Lead)"
    if raw.startswith("reconstruction regime"):
        return "Reconstruction" if compact else "Reconstruction Regime (Step 0)"
    return raw


def _mollweide_scatter(ax, latlons, values, title, cmap="viridis", vmin=None, vmax=None):
    sc = ax.scatter(
        np.radians(latlons[:, 1]), np.radians(latlons[:, 0]),
        c=values, s=0.5, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True,
    )
    ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE)
    ax.grid(True, lw=0.3, alpha=0.4)
    return sc


def _robinson_scatter(ax, latlons, values, title, cmap="viridis", vmin=None, vmax=None):
    x, y = _robinson_project(latlons[:, 1], latlons[:, 0])
    _draw_robinson_frame(ax)
    # Larger markers toward the poles (matches plot_mollweide_fields): fills the
    # thinning reduced-Gaussian rings without interpolating the field.
    marker_sizes = 0.5 + 2.0 * (np.abs(latlons[:, 0]) / 90.0) ** 2
    sc = ax.scatter(
        x, y, c=values, s=marker_sizes, cmap=cmap, vmin=vmin, vmax=vmax,
        rasterized=True, zorder=2,
    )
    _draw_robinson_coastlines(ax)
    ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE, pad=10)
    ax.set_facecolor("0.96")
    ax.set_aspect("equal")
    ax.set_xlim(-2.75, 2.75)
    ax.set_ylim(-1.42, 1.42)
    ax.axis("off")
    return sc


def _method_map_fields(star):
    fields = {}
    with np.load(RUN_DIR / "anchors.npz") as f:
        fields[display_name("mode_map")] = f["mode_map"]
        fields[display_name("iid_seed0")] = f["iid_seed0"]
        smoothed_map_n10 = f["smoothed_map_n10"]
    with np.load(RUN_DIR / "method1_sensitivity.npz") as f:
        fields[f"Joint MAP ($\\lambda^\\star$={star['lambda_star']:.0f})"] = f["field_star"]
    fields["Smoothed MAP (n=10)"] = smoothed_map_n10
    # The faithfulness-budget operating point: when its solve exists for this run,
    # its panel REPLACES the Mode-selection MRF panel -- at +48h the MRF is
    # degenerate (every beta returns essentially the Per-cell MAP field, so the
    # panel duplicates information; the MRF keeps its pareto-smear point and
    # spectrum curve). Regimes without budget_point.npz keep the MRF panel.
    budget_path = RUN_DIR / "budget_point.npz"
    m4_path = RUN_DIR / "method4_sweep.npz"
    if budget_path.exists():
        with np.load(budget_path) as f:
            fields[f"Joint MAP ($\\lambda$={float(f['lambda_budget']):.0f})"] = f["field"]
    elif m4_path.exists():
        rows = _load_rows()
        star_rt = float(next(r for r in rows if r["name"] == "m1_star")["r_tilde"])
        m4_rows = [r for r in rows if r["kind"] == "method4"]
        if m4_rows:
            nearest = min(m4_rows, key=lambda r: abs(float(r["r_tilde"]) - star_rt))
            with np.load(m4_path) as f:
                idx = int(np.argmin(np.abs(f["betas"] - float(nearest["param"]))))
                fields[f"Mode-selection MRF ($\\beta$={nearest['param']})"] = f["fields"][idx]
    era5_path = RUN_DIR / "era5_reference.npz"
    if era5_path.exists():
        with np.load(era5_path) as f:
            fields["ERA5 reference"] = f["era5"]
    return fields


def _region_coords(latlons):
    latlons = np.asarray(latlons, dtype=float)
    lats = latlons[:, 0]
    lons = _wrap_longitudes(latlons[:, 1])
    box = (
        (lons >= REGION_LON_MIN)
        & (lons <= REGION_LON_MAX)
        & (lats >= REGION_LAT_MIN)
        & (lats <= REGION_LAT_MAX)
    )
    if not np.any(box):
        raise ValueError("Figure-1.1 region contains no grid cells")
    return box, lons[box], lats[box]


def _plot_region_fields(
    latlons,
    fields,
    *,
    suptitle,
    figsize,
    ncols=2,
    marker_size=14.0,
    suptitle_y=0.985,
    top=0.92,
    hspace=0.13,
):
    box, lon_b, lat_b = _region_coords(latlons)
    field_items = list(fields.items())
    ncols = min(max(1, int(ncols)), len(field_items))
    nrows = int(np.ceil(len(field_items) / ncols))

    values_for_scale = np.concatenate([
        np.asarray(values, dtype=float).reshape(-1)[box] for values in fields.values()
    ])
    p2, p98 = np.percentile(values_for_scale, [2, 98])
    vspan = float(max(abs(p2), abs(p98)))
    if not np.isfinite(vspan) or vspan == 0.0:
        vspan = 1.0
    vmin, vmax = -vspan, vspan
    aspect = 1.0 / np.cos(np.deg2rad(0.5 * (REGION_LAT_MIN + REGION_LAT_MAX)))
    marker_sizes = _latitude_marker_sizes(lat_b, base_size=marker_size)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    sc = None
    for idx, (title, values) in enumerate(field_items):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        values_b = np.asarray(values, dtype=float).reshape(-1)[box]
        sc = ax.scatter(
            lon_b,
            lat_b,
            c=values_b,
            s=marker_sizes,
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
            zorder=2,
        )
        for seg_lon, seg_lat in _natural_earth_coastline_segments():
            ax.plot(seg_lon, seg_lat, color="0.08", lw=0.9, alpha=0.85, zorder=3)
        ax.set_xlim(REGION_LON_MIN, REGION_LON_MAX)
        ax.set_ylim(REGION_LAT_MIN, REGION_LAT_MAX)
        ax.set_aspect(aspect)
        ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE, pad=7)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.reshape(-1)[len(field_items):]:
        ax.set_visible(False)

    fig.suptitle(suptitle, y=suptitle_y, fontsize=FIG_TITLE_FONTSIZE)
    fig.subplots_adjust(left=0.02, right=0.90, bottom=0.035, top=top,
                        wspace=0.06, hspace=hspace)
    fig.canvas.draw()
    for row in range(nrows):
        visible = [ax for ax in axes[row, :] if ax.get_visible()]
        if not visible:
            continue
        panel = visible[-1].get_position()
        cbar_ax = fig.add_axes([0.915, panel.y0, 0.017, panel.height])
        cbar = fig.colorbar(sc, cax=cbar_ax, orientation="vertical",
                            label="2 m temperature (standardised)")
        cbar.set_label("2 m temperature (standardised)", fontsize=AXIS_LABEL_FONTSIZE)
        cbar.ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    return fig, axes


def fig_maps(latlons, star):
    fields = _method_map_fields(star)
    regime_title = _display_regime_label(compact=True)
    fig, _ = plot_mollweide_fields(
        latlons, fields,
        suptitle=f"{regime_title}: 2-metre temperature (standardised)",
        suptitle_fontsize=FIG_TITLE_FONTSIZE,
        panel_title_fontsize=PANEL_TITLE_FONTSIZE,
        colorbar_label="2 m temperature (standardised)",
        colorbar_label_fontsize=AXIS_LABEL_FONTSIZE,
        colorbar_tick_fontsize=TICK_LABEL_FONTSIZE,
        colorbar_shrink=0.78,
        ncols=2,
        row_colorbars=True,
        figsize=(10.8, 12.0),
        suptitle_y=0.965,
        panel_title_pad=4,
        top=0.93,
        hspace=0.02,
    )
    fig.savefig(FIG_DIR / "maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote maps.png")


def fig_region_maps(latlons, star):
    fields = _method_map_fields(star)
    regime_title = _display_regime_label(compact=True)
    fig, _ = _plot_region_fields(
        latlons,
        fields,
        suptitle=f"{regime_title}: 2-metre temperature, North Atlantic / Europe",
        figsize=(10.8, 10.8),
        top=0.91,
        hspace=0.18,
    )
    fig.savefig(FIG_DIR / "region_maps.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote region_maps.png")


def _format_lambda(lam):
    return f"{float(lam):g}"


def _select_lambda_map_indices(lambdas, n_positive=6):
    """Indices for lambda=0 plus the first positive sweep values, in plot order."""

    lambdas = np.asarray(lambdas, dtype=float)
    order = np.argsort(lambdas)
    zero = [idx for idx in order if np.isclose(lambdas[idx], 0.0)]
    positive = [idx for idx in order if lambdas[idx] > 0.0]
    selected = zero[:1] + positive[:n_positive]
    if not selected:
        raise ValueError("method1_sweep.npz contains no lambda fields")
    return selected


def _lambda_map_fields():
    with np.load(RUN_DIR / "method1_sweep.npz") as f:
        lambdas = np.asarray(f["lambdas"], dtype=float)
        sweep_fields = np.asarray(f["fields"])
    selected = _select_lambda_map_indices(lambdas, n_positive=6)
    fields = {
        f"Joint MAP ($\\lambda={_format_lambda(lambdas[idx])}$)": sweep_fields[idx]
        for idx in selected
    }
    anchors_path = RUN_DIR / "anchors.npz"
    if anchors_path.exists():
        with np.load(anchors_path) as f:
            if "smoothed_map_n10" in f.files:
                fields["Smoothed MAP ($n=10$ target)"] = f["smoothed_map_n10"]
    return fields, lambdas, selected


def fig_lambda_maps(latlons):
    fields, lambdas, selected = _lambda_map_fields()
    regime_title = _display_regime_label(compact=True)
    fig, _ = plot_mollweide_fields(
        latlons,
        fields,
        suptitle=f"{regime_title}: Joint MAP Lambda Sweep",
        suptitle_fontsize=FIG_TITLE_FONTSIZE,
        panel_title_fontsize=PANEL_TITLE_FONTSIZE,
        colorbar_label="2 m temperature (standardised)",
        colorbar_label_fontsize=AXIS_LABEL_FONTSIZE,
        colorbar_tick_fontsize=TICK_LABEL_FONTSIZE,
        colorbar_shrink=0.78,
        ncols=2,
        row_colorbars=True,
        figsize=(10.8, 15.4),
        suptitle_y=0.97,
        panel_title_pad=4,
        top=0.94,
        hspace=0.02,
    )
    fig.savefig(FIG_DIR / "lambda_sweep_maps.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    selected_lambdas = ", ".join(_format_lambda(lambdas[idx]) for idx in selected)
    target_suffix = " + smoothed_map_n10 target" if len(fields) > len(selected) else ""
    print(f"wrote lambda_sweep_maps.png (lambdas: {selected_lambdas}{target_suffix})")


def fig_region_lambda_maps(latlons):
    fields, lambdas, selected = _lambda_map_fields()
    regime_title = _display_regime_label(compact=True)
    fig, _ = _plot_region_fields(
        latlons,
        fields,
        suptitle=f"{regime_title}: Joint MAP Lambda Sweep, North Atlantic / Europe",
        figsize=(10.8, 14.2),
        top=0.93,
        hspace=0.16,
    )
    fig.savefig(FIG_DIR / "region_lambda_sweep_maps.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    selected_lambdas = ", ".join(_format_lambda(lambdas[idx]) for idx in selected)
    target_suffix = " + smoothed_map_n10 target" if len(fields) > len(selected) else ""
    print(
        "wrote region_lambda_sweep_maps.png "
        f"(lambdas: {selected_lambdas}{target_suffix})"
    )


def fig_pareto_smear(star):
    rows = _load_rows()
    anchors = [r for r in rows if r["kind"] == "anchor"]
    m1 = sorted((r for r in rows if r["kind"] == "method1"), key=lambda r: float(r["param"]))
    m4 = sorted((r for r in rows if r["kind"] == "method4"), key=lambda r: float(r["param"]))
    star_row = next((r for r in rows if r["name"] == "m1_star"), None)

    m1_colour = field_style("m1_star")[0]
    m4_colour = field_style("m4_beta1")[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.95))

    # Left: Pareto plane; R~ is trend-compressed on the real grid.
    ax1.plot(
        [float(r["r_tilde"]) for r in m1], [float(r["nll_over_n"]) for r in m1],
        "o-", color=m1_colour, markersize=7.5, label="Joint MAP ($\\lambda$ Sweep)",
    )
    # Anchors keyed by the shared field_style; omit the iid draw here so the
    # linear axis resolves the method/blur regime instead of the noise-floor tail.
    for r in anchors:
        name = r["name"]
        if name.startswith("iid_seed"):
            continue
        colour, marker, _ = field_style(name)
        x, y = float(r["r_tilde"]), float(r["nll_over_n"])
        ax1.scatter(x, y, marker=marker, s=80, color=colour, zorder=5)
        _lkw = ({"xytext": (-4, 7), "ha": "right"} if name == "mixture_mean"
                else {"xytext": (10, 8), "ha": "right"} if name == "mode_map"
                else {"xytext": (4, 4), "ha": "left"})
        ax1.annotate(display_name(name), (x, y),
                     fontsize=ANNOTATION_FONTSIZE, textcoords="offset points", **_lkw)
    if m4:
        r0 = m4[-1]
        ax1.scatter(float(r0["r_tilde"]), float(r0["nll_over_n"]),
                    marker="o", s=80, color=m4_colour, zorder=5,
                    label="Mode-selection MRF")
    if star_row is not None:
        colour, marker, _ = field_style("m1_star")
        ax1.scatter(float(star_row["r_tilde"]), float(star_row["nll_over_n"]),
                    marker=marker, s=120, color=colour, edgecolor="k", linewidth=0.8,
                    zorder=6, label="Joint MAP @ $\\lambda^\\star$")
    ax1.axvline(star["target_r_tilde"], color="grey", lw=0.8, ls=":",
                label="Smoothed MAP $\\tilde{R}$ Target")
    ax1.set_xscale("linear")
    ax1.set_xlabel("Scale-free roughness ($\\tilde{R}$)", fontsize=AXIS_LABEL_FONTSIZE)
    ax1.set_ylabel("NLL/N (nats)", fontsize=AXIS_LABEL_FONTSIZE)
    ax1.set_title("NLL vs Roughness", fontsize=PANEL_TITLE_FONTSIZE)
    ax1.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    ax1.legend(fontsize=LEGEND_FONTSIZE, framealpha=0.9)

    # Right: smear fraction vs R~ -- the panel the bare Pareto cannot show.
    for suffix, ls, tag in (("", "-", "Global"), ("__bimodal", "--", "Bimodal $>2\\sigma$")):
        ax2.plot(
            [float(r["r_tilde"]) for r in m1],
            [float(r[f"dnll_frac_gt_0p125{suffix}"]) for r in m1],
            "o" + ls, color=m1_colour, markersize=7.5, label=f"Joint MAP - {tag}",
        )
    if m4:
        ax2.plot(
            [float(r["r_tilde"]) for r in m4],
            [float(r["dnll_frac_gt_0p125"]) for r in m4],
            "o-", color=m4_colour, markersize=7.5, label="Mode-selection MRF",
        )
    # Global anchors (one star each, shared field_style colour); omit iid as above.
    for r in anchors:
        name = r["name"]
        if name not in ("smoothed_map_n10", "mode_map"):
            continue
        colour, marker, _ = field_style(name)
        x, y = float(r["r_tilde"]), float(r["dnll_frac_gt_0p125"])
        ax2.scatter(x, y, marker=marker, s=80, color=colour, zorder=5)
        _lkw2 = ({"xytext": (-4, 7), "ha": "right"}
                 if name == "mode_map"
                 else {"xytext": (4, 4), "ha": "left"})
        ax2.annotate(display_name(name), (x, y),
                     fontsize=ANNOTATION_FONTSIZE, textcoords="offset points", **_lkw2)

    # Headline matched-R~ comparison IN the bimodal stratum, with within-field
    # spatial-bootstrap CIs (same draws as the \fcBimodalFrac* macros). This is
    # the comparison the chapter leads with; without the blur's bimodal point and
    # the CIs the panel cannot actually show it.
    blur_row = next((r for r in anchors if r["name"] == "smoothed_map_n10"), None)
    m1_ci = _bimodal_frac_ci("m1_star")
    blur_ci = _bimodal_frac_ci("smoothed_map_n10")
    if star_row is not None and m1_ci is not None:
        x = float(star_row["r_tilde"])
        ax2.errorbar(
            x, m1_ci["point"],
            yerr=[[m1_ci["point"] - m1_ci["lo"]], [m1_ci["hi"] - m1_ci["point"]]],
            fmt="o", ms=12, color=m1_colour, ecolor=m1_colour, capsize=5,
            elinewidth=2.0, markeredgecolor="k", markeredgewidth=1.2, zorder=8,
            label="Joint MAP @ $\\lambda^\\star$ (Bimodal $>2\\sigma$, 95% CI)",
        )
    if blur_row is not None and blur_ci is not None:
        x = float(blur_row["r_tilde"])
        colour, marker, _ = field_style("smoothed_map_n10")
        ax2.errorbar(
            x, blur_ci["point"],
            yerr=[[blur_ci["point"] - blur_ci["lo"]], [blur_ci["hi"] - blur_ci["point"]]],
            fmt=marker, ms=12, color=colour, ecolor=colour, capsize=5,
            elinewidth=2.0, markeredgecolor="k", markeredgewidth=1.2, zorder=8,
            label="Smoothed MAP (Bimodal $>2\\sigma$, 95% CI)",
        )
        ax2.annotate("Smoothed MAP\n(Bimodal)", (x, blur_ci["point"]),
                     fontsize=ANNOTATION_FONTSIZE, textcoords="offset points",
                     xytext=(4, 4), ha="left")
    ax2.axvline(star["target_r_tilde"], color="grey", lw=0.8, ls=":")
    ax2.set_xscale("linear")
    ax2.set_xlabel("Scale-free roughness ($\\tilde{R}$)", fontsize=AXIS_LABEL_FONTSIZE)
    ax2.set_ylabel("Off-mode fraction", fontsize=AXIS_LABEL_FONTSIZE)
    ax2.set_title("Smear Tail vs Roughness", fontsize=PANEL_TITLE_FONTSIZE)
    ax2.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    ax2.legend(fontsize=DENSE_LEGEND_FONTSIZE, framealpha=0.9)

    fig.suptitle(
        _display_regime_label(),
        fontsize=FIG_TITLE_FONTSIZE,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))

    # House-style sweep treatment (shared with the toy Phase-2 figures): outline
    # direction arrows + first / best / last lambda labels on Method 1 only. The
    # Mode-selection MRF beta sweep is omitted here -- it is degenerate at +48h
    # (constant R~, see caption), so arrows/labels would be meaningless. Added
    # after tight_layout so the transforms (and hence arrow angles) are final.
    m1_r = [float(r["r_tilde"]) for r in m1]
    m1_nll = [float(r["nll_over_n"]) for r in m1]
    m1_global = [float(r["dnll_frac_gt_0p125"]) for r in m1]
    m1_bimodal = [float(r["dnll_frac_gt_0p125__bimodal"]) for r in m1]
    lam_first = float(m1[0]["param"])
    lam_last = float(m1[-1]["param"])

    def _plabel(ax, x, y, text, *, xytext, ha="left", va="center", bold=False):
        ax.annotate(
            text, (float(x), float(y)), fontsize=ANNOTATION_FONTSIZE, color=m1_colour,
            fontweight="bold" if bold else "normal",
            xytext=xytext, textcoords="offset points", ha=ha, va=va,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.8),
            zorder=11,
        )

    star_lbl = (f"$\\lambda^\\star\\!\\approx\\!{float(star_row['param']):.0f}$"
                if star_row is not None else None)

    # Left panel: Pareto plane.
    _segment_arrows(ax1, m1_r, m1_nll, m1_colour, n_arrows=2, outline=True)
    _plabel(ax1, m1_r[0], m1_nll[0], f"$\\lambda={lam_first:g}$",
            xytext=(6, -1), ha="left", va="top")
    _plabel(ax1, m1_r[-1], m1_nll[-1], f"$\\lambda={lam_last:g}$",
            xytext=(8, 0), ha="left", va="center")
    if star_row is not None:
        _plabel(ax1, float(star_row["r_tilde"]), float(star_row["nll_over_n"]),
                star_lbl, xytext=(0, -10), ha="center", va="top", bold=True)

    # Right panel: smear tail, both Method-1 curves (global solid, bimodal dashed).
    _segment_arrows(ax2, m1_r, m1_global, m1_colour, n_arrows=2, outline=True)
    _segment_arrows(ax2, m1_r, m1_bimodal, m1_colour, n_arrows=2, outline=True)
    _plabel(ax2, m1_r[0], m1_global[0], f"$\\lambda={lam_first:g}$",
            xytext=(15, 0), ha="center", va="center")
    _plabel(ax2, m1_r[-1], m1_global[-1], f"$\\lambda={lam_last:g}$",
            xytext=(23, 0), ha="center", va="center")
    _plabel(ax2, m1_r[-1], m1_bimodal[-1], f"$\\lambda={lam_last:g}$",
            xytext=(3, 10), ha="center", va="center")
    if star_row is not None and m1_ci is not None:
        _plabel(ax2, float(star_row["r_tilde"]), m1_ci["point"], star_lbl,
                xytext=(0, -10), ha="center", va="top", bold=True)

    fig.savefig(FIG_DIR / "pareto_smear.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote pareto_smear.png")


def fig_variogram():
    with np.load(RUN_DIR / "variograms.npz") as f:
        centres = f["centres"]
        variograms = {k: f[k] for k in f.files if k != "centres"}
    # An `era5` series (if the runner could load it) is the same rung-3
    # direction-of-realism reference as on the spectrum -- descriptive, not a
    # target. It is picked up automatically from variograms.npz and drawn last.
    has_era5 = "era5" in variograms
    title = "Sampled spherical variograms (descriptive, bracket-anchored)"
    if has_era5:
        title += " + ERA5 reference"
    fig, ax = plot_variograms(
        centres, variograms, xlabel="angular distance (deg)", title=title,
    )
    ax.set_xlim(0, 60)
    fig.savefig(FIG_DIR / "variogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote variogram.png" + (" (with ERA5 reference)" if has_era5 else ""))


# The main-text +48h spectrum shows exactly the methods that appear as panels in
# the field-maps figure (the faithfulness-budget operating point included), in
# the same panel order. The two mid-band duplicates
# (Mixture mean and Mode-selection MRF) are dropped -- at +48h the MRF is
# degenerate onto Per-cell MAP and neither has a maps panel -- so the coherence
# figure and the field-maps figure carry an identical method set. Colour encodes
# method, dash encodes operating point: the two Joint MAP curves (lambda* solid,
# faithfulness-budget dashed) share the same blue.
SPECTRUM_CURVES = ("mode_map", "iid_seed0", "m1_star", "smoothed_map_n10",
                   "m1_budget", "era5")


def fig_spectrum():
    """Native O96 angular power spectrum C_l -- main-text coherence figure (S5.4).

    RUNG-3 bracket diagnostic, never a target/validation. Loads spectra.npz
    (written by run_real_eval.py stage_scores / stage_spectrum), plots the
    SPECTRUM_CURVES set over the Parseval-resolved band, and draws the `era5`
    series as a direction-of-realism reference line. The faithfulness-budget Joint
    MAP curve is labelled with its solved lambda, single-sourced from
    budget_point.npz so it matches the maps figure's budget panel exactly. Skips
    gracefully if spectra.npz is absent (older run dirs).
    """
    spec_path = RUN_DIR / "spectra.npz"
    if not spec_path.exists():
        print(f"skip spectrum.png ({spec_path.name} absent; "
              "rerun the scores stage to produce it)")
        return
    with np.load(spec_path) as f:
        ell = f["ell"]
        lmax_resolved = int(f["lmax_resolved"]) if "lmax_resolved" in f.files else int(ell[-1])
        available = {k: f[k] for k in f.files if k not in ("ell", "lmax_resolved")}
    # Select the fig:fc-maps method set, preserving the panel order.
    spectra = {k: available[k] for k in SPECTRUM_CURVES if k in available}
    has_era5 = "era5" in spectra

    # Match the maps figure's budget-panel label (same lambda, same rounding).
    labels = {}
    budget_path = RUN_DIR / "budget_point.npz"
    if "m1_budget" in spectra and budget_path.exists():
        with np.load(budget_path) as f:
            labels["m1_budget"] = f"Joint MAP ($\\lambda$={float(f['lambda_budget']):.0f})"

    fig, ax = plot_spectra(ell, spectra,
                           title="Angular power spectrum ($C_\\ell$)",
                           labels=labels,
                           figsize=(8.0, 4.8))
    ax.set_xlim(1, lmax_resolved)
    ax.set_xlabel("Angular degree ($\\ell$)")
    ax.set_ylabel("$C_\\ell$ (normalised power)")
    ax.legend(fontsize=LEGEND_FONTSIZE)
    fig.savefig(FIG_DIR / "spectrum.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote spectrum.png" + (" (with ERA5 reference)" if has_era5 else ""))


def fig_bimodal_enrichment(latlons, star):
    with np.load(RUN_DIR / "masks.npz") as f:
        bimodal_2s = f["bimodal"]
        bimodal_1s = f["bimodal_1sigma"]
    with np.load(RUN_DIR / "delta_per_cell.npz") as f:
        deltas = {k: f[k] for k in f.files}

    names = ["iid_seed0", "smoothed_map_n10", "m1_star"]
    names = [n for n in names if n in deltas]

    def bar_tick_label(name):
        return {
            "iid_seed0": "Independent\nDraw",
            "smoothed_map_n10": "Smoothed\nMAP",
            "m1_star": "Joint MAP\n($\\lambda^\\star$)",
            "m4_beta1": "Mode-selection\nMRF",
        }.get(name, display_name(name).replace(" ", "\n"))

    panel_title_size = PANEL_TITLE_FONTSIZE + 1.5
    fig = plt.figure(figsize=(13.6, 5.7))
    grid = fig.add_gridspec(1, 3, width_ratios=[0.82, 1.72, 0.045], wspace=0.10)
    ax1 = fig.add_subplot(grid[0, 0])
    width = 0.35
    xs = np.arange(len(names))
    zero_smear = {n: float(np.mean(deltas[n] > 0.125)) == 0 for n in names}
    for offset, (mask, label) in enumerate(
        (
            (bimodal_1s, "Bimodal $>1\\sigma$"),
            (bimodal_2s, "Bimodal $>2\\sigma$"),
        )
    ):
        enrich = []
        for n in names:
            d = deltas[n]
            global_frac = float(np.mean(d > 0.125))
            mask_frac = float(np.mean(d[mask] > 0.125))
            enrich.append(mask_frac / global_frac if global_frac > 0 else 0.0)
        ax1.bar(xs + (offset - 0.5) * width, enrich, width, label=label)
    ax1.axhline(1.0, color="grey", lw=0.8, ls=":")
    for i, n in enumerate(names):
        if zero_smear[n]:
            ax1.text(xs[i], 0.05, "No off-mode\ncells", ha="center", va="bottom",
                     fontsize=8, color="grey", style="italic")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([bar_tick_label(n) for n in names], rotation=0,
                        ha="center", fontsize=12)
    ax1.tick_params(axis="x", pad=4)
    ax1.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)
    ax1.set_ylabel("Enrichment ratio", fontsize=AXIS_LABEL_FONTSIZE)
    ax1.set_title("Off-Mode Enrichment by Bimodal Class", fontsize=panel_title_size)
    ax1.legend(fontsize=LEGEND_FONTSIZE)

    ax2 = fig.add_subplot(grid[0, 1])
    cbar_ax = fig.add_subplot(grid[0, 2])
    d = deltas["m1_star"]
    sc = _robinson_scatter(
        ax2, latlons, np.log10(np.maximum(d, 1e-6)),
        f"$\\log_{{10}}$ Off-Mode Cost, Joint MAP @ $\\lambda^\\star$={star['lambda_star']:.0f}",
        cmap="viridis", vmin=-4, vmax=1,
    )
    ax2.set_title(ax2.get_title(), fontsize=panel_title_size, pad=10)
    cbar = fig.colorbar(sc, cax=cbar_ax, orientation="vertical",
                        label="$\\log_{10}$ $\\Delta$NLL")
    cbar.set_label("$\\log_{10}$ $\\Delta$NLL", fontsize=AXIS_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)

    regime_title = _display_regime_label()
    fig.suptitle(regime_title, fontsize=FIG_TITLE_FONTSIZE, y=0.98)
    fig.subplots_adjust(left=0.055, right=0.965, bottom=0.10, top=0.86)
    fig.savefig(FIG_DIR / "bimodal_enrichment.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote bimodal_enrichment.png")


def fig_robustness(star):
    probes_path = RUN_DIR / "robustness_probes.json"
    if not probes_path.exists():
        print(f"skip robustness.png ({probes_path.name} absent; "
              "robustness probes not run for this regime)")
        return
    with np.load(RUN_DIR / "modes.npz") as f:
        mode_unary = f["mode_unary"]
        mode_counts = f["mode_counts"]
    probes = json.loads(probes_path.read_text())

    # mode_unary is +inf on padded slots, so columns 0/1 of the sort are the
    # best and second-best real modes wherever mode_counts >= 2.
    multi = mode_counts >= 2
    sorted_unary = np.sort(mode_unary, axis=1)
    gaps = (sorted_unary[:, 1] - sorted_unary[:, 0])[multi]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))

    ax1.hist(gaps, bins=60, color="tab:blue")
    med = float(np.median(gaps))
    ax1.axvline(med, color="tab:red", ls="--", lw=1, label=f"median {med:.1f} nats")
    beta100 = probes["m4_beta_scale"]["betas"]["100"]
    ax1.set_xlabel("unary gap, 2nd-best minus best mode (nats)")
    ax1.set_ylabel(f"multi-mode cells (n = {int(multi.sum()):,})")
    ax1.set_title(
        "Near-one-hot unary gaps pin the Mode-selection MRF sweep\n"
        f"(even beta=100 moves only {beta100['cells_moved_off_unary_best']} of "
        f"{probes['m4_beta_scale']['n_cells']:,} cells)",
        fontsize=10,
    )
    ax1.legend(fontsize=8)

    rows = _load_rows()
    unit_blur = next(r for r in rows if r["name"] == "smoothed_map_n10")
    unit_star = next(r for r in rows if r["name"] == "m1_star")
    w = probes["weighted_graph"]
    points = [
        ("Smoothed MAP", float(unit_blur["r_tilde"]), float(unit_blur["nll_over_n"]),
         float(unit_blur["dnll_frac_gt_0p125"]), "tab:orange", "o"),
        (f"Joint MAP @ $\\lambda^\\star$={star['lambda_star']:.0f}", float(unit_star["r_tilde"]),
         float(unit_star["nll_over_n"]),
         float(unit_star["dnll_frac_gt_0p125"]), "tab:blue", "o"),
        ("Smoothed MAP (weighted)", w["smoothed_map_n10_weighted"]["r_tilde"],
         w["smoothed_map_n10_weighted"]["nll_over_n"],
         w["smoothed_map_n10_weighted"]["dnll_frac_gt_0p125"], "tab:orange", "s"),
        (f"Joint MAP @ $\\lambda^\\star$={w['lambda_star_weighted']:.0f} (weighted)",
         w["m1_at_lambda_star_weighted"]["r_tilde"],
         w["m1_at_lambda_star_weighted"]["nll_over_n"],
         w["m1_at_lambda_star_weighted"]["dnll_frac_gt_0p125"], "tab:blue", "s"),
    ]
    for label, rt, nll, tail, color, marker in points:
        ax2.scatter(rt, nll, color=color, marker=marker, s=70, zorder=5)
        ax2.annotate(f"{label}\ntail>0.125: {tail:.1%}", (rt, nll), fontsize=7,
                     xytext=(6, -10), textcoords="offset points")
    ax2.set_xscale("log")
    # Auto-derive the log-axis window from the plotted R~ values (was hard-coded
    # to the step-0 compressed range); half-decade pad on each side.
    rts = [rt for _, rt, _, _, _, _ in points]
    ax2.set_xlim(min(rts) / np.sqrt(10), max(rts) * np.sqrt(10))
    ax2.set_xlabel("R~ (unit-weight metric, log axis)")
    ax2.set_ylabel("NLL/N (nats)")
    ax2.set_title(
        "Edge-weight convention robustness: Joint MAP beats the blur\n"
        "at matched coherence under both conventions (circle = unit, square = weighted)",
        fontsize=10,
    )

    fig.suptitle(_regime_label(), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / "robustness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote robustness.png")


def main():
    global RUN_DIR, FIG_DIR, DATA_NPZ
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None,
                        help=("subset of {maps, pareto, variogram, spectrum, lambda_maps, "
                              "region_maps, region_lambda_maps, enrichment, robustness}"))
    parser.add_argument("--data", type=Path, default=DATA_NPZ,
                        help="real-marginal npz (mirrors run_real_eval.py --data)")
    parser.add_argument("--out-dir", type=Path, default=RUN_DIR,
                        help="run artifacts dir (mirrors run_real_eval.py --out-dir)")
    parser.add_argument("--fig-dir", type=Path, default=FIG_DIR,
                        help="figure output dir (default outputs/figures)")
    args = parser.parse_args()

    RUN_DIR = args.out_dir
    FIG_DIR = args.fig_dir
    DATA_NPZ = args.data

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = load_real_marginal(DATA_NPZ)
    star = json.loads((RUN_DIR / "lambda_star.json").read_text())
    figures = {
        "maps": lambda: fig_maps(data["latlons"], star),
        "pareto": lambda: fig_pareto_smear(star),
        "variogram": fig_variogram,
        "spectrum": fig_spectrum,
        "lambda_maps": lambda: fig_lambda_maps(data["latlons"]),
        "region_maps": lambda: fig_region_maps(data["latlons"], star),
        "region_lambda_maps": lambda: fig_region_lambda_maps(data["latlons"]),
        "enrichment": lambda: fig_bimodal_enrichment(data["latlons"], star),
        "robustness": lambda: fig_robustness(star),
    }
    requested = args.only if args.only else list(figures)
    unknown = set(requested) - set(figures)
    if unknown:
        raise SystemExit(f"unknown figures: {sorted(unknown)}")
    for name in figures:
        if name in requested:
            figures[name]()


if __name__ == "__main__":
    main()
