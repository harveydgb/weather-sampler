"""Phase 4 report figures from the persisted runner artifacts (phase_4_plan S5).

Run after `scripts/run_phase4_real.py` (all stages + --lambda-star):

    .venv/bin/python scripts/make_phase4_figures.py

Writes to outputs/figures/:
  phase_4_maps.png               4/5-panel global Mollweide
  phase_4_pareto_smear.png       (NLL/N, R~) plane + smear-fraction panel
  phase_4_variogram.png          sampled spherical variograms (descriptive;
                                 + ERA5 reference line if available)
  phase_4_spectrum.png           native O96 angular power spectrum C_l vs l
                                 (rung-3 bracket; main-text coherence figure;
                                 + ERA5 direction reference if available)
  phase_4_bimodal_enrichment.png W1 MUST: dNLL>0.125 enrichment in the audit
                                 S4 bimodal masks + Mollweide dNLL map
  phase_4_robustness.png         unary-gap histogram (why the Mode-selection MRF beta
                                 sweep is pinned) + unit-vs-weighted-graph
                                 matched-coherence points (needs
                                 robustness_probes.json from
                                 scripts/run_phase4_probes.py)

Figures are generated here (not in notebook 04) so the MUST figure task does
not depend on notebook execution; the notebook displays these files.
`--only name [name ...]` regenerates a subset (maps / pareto / variogram /
enrichment / robustness) without rewriting the other committed PNGs.
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
    _robinson_project,
    _segment_arrows,
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

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "outputs" / "runs" / "phase_4_real"
FIG_DIR = REPO_ROOT / "outputs" / "figures"
DATA_NPZ = REPO_ROOT / "outputs" / "data" / "phase_4_real_2t.npz"


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


def _mollweide_scatter(ax, latlons, values, title, cmap="viridis", vmin=None, vmax=None):
    sc = ax.scatter(
        np.radians(latlons[:, 1]), np.radians(latlons[:, 0]),
        c=values, s=0.5, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True,
    )
    ax.set_title(title, fontsize=10)
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
    ax.set_title(title, fontsize=10, pad=10)
    ax.set_facecolor("0.96")
    ax.set_aspect("equal")
    ax.set_xlim(-2.75, 2.75)
    ax.set_ylim(-1.42, 1.42)
    ax.axis("off")
    return sc


def fig_maps(latlons, star):
    fields = {}
    with np.load(RUN_DIR / "anchors.npz") as f:
        fields[display_name("mode_map")] = f["mode_map"]
        fields[display_name("iid_seed0")] = f["iid_seed0"]
    with np.load(RUN_DIR / "method1_sensitivity.npz") as f:
        fields[f"Joint MAP ($\\lambda^\\star$={star['lambda_star']:.0f})"] = f["field_star"]
    with np.load(RUN_DIR / "anchors.npz") as f:
        fields["Smoothed MAP (n=10)"] = f["smoothed_map_n10"]
    m4_path = RUN_DIR / "method4_sweep.npz"
    if m4_path.exists():
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
    fig, _ = plot_mollweide_fields(
        latlons, fields,
        suptitle=(
            "Real O96 2-metre temperature (standardised), single snapshot\n"
            f"{_regime_label()}"
        ),
    )
    fig.savefig(FIG_DIR / "phase_4_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote phase_4_maps.png")


def fig_pareto_smear(star):
    rows = _load_rows()
    anchors = [r for r in rows if r["kind"] == "anchor"]
    m1 = sorted((r for r in rows if r["kind"] == "method1"), key=lambda r: float(r["param"]))
    m4 = sorted((r for r in rows if r["kind"] == "method4"), key=lambda r: float(r["param"]))
    star_row = next((r for r in rows if r["name"] == "m1_star"), None)

    m1_colour = field_style("m1_star")[0]
    m4_colour = field_style("m4_beta1")[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # Left: Pareto plane; R~ is trend-compressed on the real grid.
    ax1.plot(
        [float(r["r_tilde"]) for r in m1], [float(r["nll_over_n"]) for r in m1],
        "o-", color=m1_colour, label="Joint MAP ($\\lambda$ sweep)",
    )
    # Anchors keyed by the shared field_style; omit the iid draw here so the
    # linear axis resolves the method/blur regime instead of the noise-floor tail.
    for r in anchors:
        name = r["name"]
        if name.startswith("iid_seed"):
            continue
        colour, marker, _ = field_style(name)
        x, y = float(r["r_tilde"]), float(r["nll_over_n"])
        ax1.scatter(x, y, marker=marker, s=70, color=colour, zorder=5)
        _lkw = ({"xytext": (-4, 7), "ha": "right"} if name == "mixture_mean"
                else {"xytext": (10, 8), "ha": "right"} if name == "mode_map"
                else {"xytext": (4, 4), "ha": "left"})
        ax1.annotate(display_name(name), (x, y),
                     fontsize=7, textcoords="offset points", **_lkw)
    if m4:
        r0 = m4[-1]
        ax1.scatter(float(r0["r_tilde"]), float(r0["nll_over_n"]),
                    marker="o", s=70, color=m4_colour, zorder=5,
                    label="Mode-selection MRF")
    if star_row is not None:
        colour, marker, _ = field_style("m1_star")
        ax1.scatter(float(star_row["r_tilde"]), float(star_row["nll_over_n"]),
                    marker=marker, s=90, color=colour, edgecolor="k", linewidth=0.6,
                    zorder=6, label="Joint MAP @ $\\lambda^\\star$")
    ax1.axvline(star["target_r_tilde"], color="grey", lw=0.8, ls=":",
                label="Smoothed MAP $\\tilde{R}$ target")
    ax1.set_xscale("linear")
    ax1.set_xlabel("Normalised roughness ($\\tilde{R}$)")
    ax1.set_ylabel("NLL/N (nats)")
    ax1.set_title("Faithfulness–coherence plane (lower-left is better)")
    ax1.legend(fontsize=7)

    # Right: smear fraction vs R~ -- the panel the bare Pareto cannot show.
    for suffix, ls, tag in (("", "-", "global"), ("__bimodal", "--", "bimodal stratum")):
        ax2.plot(
            [float(r["r_tilde"]) for r in m1],
            [float(r[f"dnll_frac_gt_0p125{suffix}"]) for r in m1],
            "o" + ls, color=m1_colour, label=f"Joint MAP — {tag}",
        )
    if m4:
        ax2.plot(
            [float(r["r_tilde"]) for r in m4],
            [float(r["dnll_frac_gt_0p125"]) for r in m4],
            "o-", color=m4_colour, label="Mode-selection MRF",
        )
    # Global anchors (one star each, shared field_style colour); omit iid as above.
    for r in anchors:
        name = r["name"]
        if name not in ("smoothed_map_n10", "mode_map"):
            continue
        colour, marker, _ = field_style(name)
        x, y = float(r["r_tilde"]), float(r["dnll_frac_gt_0p125"])
        ax2.scatter(x, y, marker=marker, s=70, color=colour, zorder=5)
        _lkw2 = ({"xytext": (-4, 7), "ha": "right"}
                 if name == "mode_map"
                 else {"xytext": (4, 4), "ha": "left"})
        ax2.annotate(display_name(name), (x, y),
                     fontsize=7, textcoords="offset points", **_lkw2)

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
            label="Joint MAP @ $\\lambda^\\star$ (bimodal, 95% CI)",
        )
    if blur_row is not None and blur_ci is not None:
        x = float(blur_row["r_tilde"])
        colour, marker, _ = field_style("smoothed_map_n10")
        ax2.errorbar(
            x, blur_ci["point"],
            yerr=[[blur_ci["point"] - blur_ci["lo"]], [blur_ci["hi"] - blur_ci["point"]]],
            fmt=marker, ms=12, color=colour, ecolor=colour, capsize=5,
            elinewidth=2.0, markeredgecolor="k", markeredgewidth=1.2, zorder=8,
            label="Smoothed MAP (bimodal, 95% CI)",
        )
        ax2.annotate("Smoothed MAP\n(bimodal)", (x, blur_ci["point"]),
                     fontsize=7, textcoords="offset points", xytext=(4, 4), ha="left")
    ax2.axvline(star["target_r_tilde"], color="grey", lw=0.8, ls=":")
    ax2.set_xscale("linear")
    ax2.set_xlabel("Normalised roughness ($\\tilde{R}$)")
    ax2.set_ylabel("Fraction of cells with $\\Delta$NLL > 0.125 nats (= 0.5$\\sigma$ off mode)")
    ax2.set_title("Smear tail vs coherence")
    ax2.legend(fontsize=6.5)

    fig.suptitle(
        _regime_label()
        .replace("forecast regime", "Forecast regime")
        .replace("reconstruction regime", "Reconstruction regime"),
        fontsize=12,
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
            text, (float(x), float(y)), fontsize=7, color=m1_colour,
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

    fig.savefig(FIG_DIR / "phase_4_pareto_smear.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote phase_4_pareto_smear.png")


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
    fig.savefig(FIG_DIR / "phase_4_variogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote phase_4_variogram.png" + (" (with ERA5 reference)" if has_era5 else ""))


def fig_spectrum():
    """Native O96 angular power spectrum C_l -- main-text coherence figure (S5.4).

    RUNG-3 bracket diagnostic, never a target/validation. Mirrors fig_variogram:
    loads spectra.npz (written by run_phase4_real.py stage_scores), plots over the
    Parseval-resolved band, draws any `era5` series as a direction-of-realism
    reference line. Skips gracefully if spectra.npz is absent (older run dirs).
    """
    spec_path = RUN_DIR / "spectra.npz"
    if not spec_path.exists():
        print(f"skip phase_4_spectrum.png ({spec_path.name} absent; "
              "rerun the scores stage to produce it)")
        return
    with np.load(spec_path) as f:
        ell = f["ell"]
        lmax_resolved = int(f["lmax_resolved"]) if "lmax_resolved" in f.files else int(ell[-1])
        spectra = {k: f[k] for k in f.files if k not in ("ell", "lmax_resolved")}
    has_era5 = "era5" in spectra
    fig, ax = plot_spectra(ell, spectra,
                           title="Angular power spectrum ($C_\\ell$)",
                           figsize=(8.0, 4.8))
    ax.set_xlim(1, lmax_resolved)
    ax.set_xlabel("Angular degree ($\\ell$)")
    ax.set_ylabel("$C_\\ell$ (normalised power)")
    fig.savefig(FIG_DIR / "phase_4_spectrum.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote phase_4_spectrum.png" + (" (with ERA5 reference)" if has_era5 else ""))


def fig_bimodal_enrichment(latlons, star):
    with np.load(RUN_DIR / "masks.npz") as f:
        bimodal_2s = f["bimodal"]
        bimodal_1s = f["bimodal_1sigma"]
    with np.load(RUN_DIR / "delta_per_cell.npz") as f:
        deltas = {k: f[k] for k in f.files}

    names = ["iid_seed0", "smoothed_map_n10", "m1_star", "m1_star_double"]
    names = [n for n in names if n in deltas]

    def bar_tick_label(name):
        return {
            "iid_seed0": "Independent\nDraw",
            "smoothed_map_n10": "Smoothed\nMAP",
            "m1_star": "Joint\nMAP",
            "m1_star_double": "Joint MAP\n(2$\\lambda$)",
            "m4_beta1": "Mode-selection\nMRF",
        }.get(name, display_name(name).replace(" ", "\n"))

    fig = plt.figure(figsize=(13, 4.6))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.18, 0.045], wspace=0.28)
    ax1 = fig.add_subplot(grid[0, 0])
    width = 0.35
    xs = np.arange(len(names))
    zero_smear = {n: float(np.mean(deltas[n] > 0.125)) == 0 for n in names}
    for offset, (mask, label) in enumerate(
        (
            (bimodal_1s, f"bimodal $>1\\sigma$ ({int(bimodal_1s.sum()):,} cells)"),
            (bimodal_2s, f"bimodal $>2\\sigma$ ({int(bimodal_2s.sum()):,} cells)"),
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
            ax1.text(xs[i], 0.05, "no smear\ncells", ha="center", va="bottom",
                     fontsize=6, color="grey", style="italic")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([bar_tick_label(n) for n in names], rotation=0,
                        ha="center", fontsize=8)
    ax1.tick_params(axis="x", pad=4)
    ax1.set_ylabel("enrichment of $\\Delta$NLL > 0.125 cells\n"
                   "(within-stratum fraction / global fraction)")
    ax1.set_title("Enrichment of $\\Delta$NLL > 0.125 cells by stratum")
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(grid[0, 1])
    cbar_ax = fig.add_subplot(grid[0, 2])
    d = deltas["m1_star"]
    sc = _robinson_scatter(
        ax2, latlons, np.log10(np.maximum(d, 1e-6)),
        "$\\log_{10}$ $\\Delta$NLL to best mode "
        f"(Joint MAP, $\\lambda^\\star$={star['lambda_star']:.0f})",
        cmap="viridis", vmin=-4, vmax=1,
    )
    fig.colorbar(sc, cax=cbar_ax, orientation="vertical",
                 label="$\\log_{10}$ $\\Delta$NLL (nats)")

    regime_title = _regime_label().replace("forecast regime", "Forecast regime")
    fig.suptitle(regime_title, fontsize=10, y=0.97)
    fig.subplots_adjust(left=0.06, right=0.965, bottom=0.16, top=0.84)
    fig.savefig(FIG_DIR / "phase_4_bimodal_enrichment.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote phase_4_bimodal_enrichment.png")


def fig_robustness(star):
    probes_path = RUN_DIR / "robustness_probes.json"
    if not probes_path.exists():
        print(f"skip phase_4_robustness.png ({probes_path.name} absent; "
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
    fig.savefig(FIG_DIR / "phase_4_robustness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote phase_4_robustness.png")


def main():
    global RUN_DIR, FIG_DIR, DATA_NPZ
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None,
                        help="subset of {maps, pareto, variogram, spectrum, enrichment, robustness}")
    parser.add_argument("--data", type=Path, default=DATA_NPZ,
                        help="real-marginal npz (mirrors run_phase4_real.py --data)")
    parser.add_argument("--out-dir", type=Path, default=RUN_DIR,
                        help="run artifacts dir (mirrors run_phase4_real.py --out-dir)")
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
