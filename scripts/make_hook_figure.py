"""Chapter 1 motivation hook figure.

Two O96 Mollweide panels on a shared colour scale, built from the same
persisted +48 h artifacts as the main-text maps figure (no new sampling):

  left  — one field drawn *independently* from each location's emitted GMM
          (anchors.npz `iid_seed0`): spatially incoherent, salt-and-pepper.
  right — an ERA5 reanalysis 2-metre-temperature field over the same grid
          (era5_reference.npz `era5`): spatially coherent.

The contrast motivates the whole report: the model emits each location's
distribution on its own, so sampling them independently discards the spatial
structure real weather exhibits. ERA5 is used here only as an illustration of
coherence, never as a scoring target (claim ladder rung 4 stays disavowed).

Run:
    .venv/bin/python scripts/make_hook_figure.py

Writes outputs/figures/fig_hook_noise_vs_structure.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sampler_research.io import load_real_marginal
from sampler_research.plotting import (
    _latitude_marker_sizes,
    _natural_earth_coastline_segments,
    _wrap_longitudes,
    plot_mollweide_fields,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "outputs" / "runs" / "phase_4_fc48_14ep_step8"
DATA_NPZ = REPO_ROOT / "outputs" / "data" / "phase_4_real_2t.npz"
OUT_GLOBAL = REPO_ROOT / "outputs" / "figures" / "fig_hook_global.png"
OUT_REGION = REPO_ROOT / "outputs" / "figures" / "fig_hook_noise_vs_structure.png"

# Plain-language panel titles for the introduction (no project jargon).
TITLE_NOISE = "Independent draw at each location"
TITLE_STRUCTURE = "ERA5 reanalysis 2 m temperature"

# Regional close-up: North Atlantic / Europe / North Africa, where the
# land-sea contrast and mid-latitude gradients give real structure, so the
# independent-draw speckle is unmistakable.
LON_MIN, LON_MAX = -60.0, 40.0
LAT_MIN, LAT_MAX = 20.0, 75.0

# Match the Phase-4 regional point-size treatment after accounting for panel
# size. Matplotlib scatter `s` is marker area; the Figure 1.1 regional panels
# are 1.418x the area of a panel in `region_maps.png`.
REGION_MARKER_BASE_SIZE = 14.0 * 1.4181399176954732


def _load():
    latlons = load_real_marginal(DATA_NPZ)["latlons"]
    with np.load(RUN_DIR / "anchors.npz") as f:
        iid = f["iid_seed0"]
    with np.load(RUN_DIR / "era5_reference.npz") as f:
        era5 = f["era5"]
    if not (iid.shape == era5.shape == (latlons.shape[0],)):
        raise ValueError(
            f"shape mismatch: iid {iid.shape}, era5 {era5.shape}, latlons {latlons.shape}"
        )
    return latlons, iid, era5


def _required_coastline_segments():
    segments = _natural_earth_coastline_segments()
    if not segments:
        raise RuntimeError(
            "Natural Earth coastline layer unavailable; install the plotting "
            "requirements, including pyshp, and check that "
            "outputs/data/natural_earth/ne_110m_coastline.shp exists before "
            "rendering Figure 1.1."
        )
    return segments


def make_global(latlons, iid, era5):
    fields = {TITLE_NOISE: iid, TITLE_STRUCTURE: era5}
    fig, _ = plot_mollweide_fields(latlons, fields, suptitle="", ncols=2)
    fig.subplots_adjust(top=0.97)
    fig.savefig(OUT_GLOBAL, dpi=200)
    print(f"wrote {OUT_GLOBAL.relative_to(REPO_ROOT)}")


def make_region(latlons, iid, era5, coastline_segments):
    lats = latlons[:, 0]
    lons = _wrap_longitudes(latlons[:, 1])
    box = (lons >= LON_MIN) & (lons <= LON_MAX) & (lats >= LAT_MIN) & (lats <= LAT_MAX)
    lon_b, lat_b = lons[box], lats[box]
    # Symmetric diverging norm: white sits at 0 on the standardised scale so the
    # RdBu_r midpoint is meaningful (2/98 clip preserved, mirrored about zero).
    p2, p98 = np.percentile(np.concatenate([iid[box], era5[box]]), [2, 98])
    vspan = float(max(abs(p2), abs(p98)))
    vmin, vmax = -vspan, vspan
    aspect = 1.0 / np.cos(np.deg2rad(0.5 * (LAT_MIN + LAT_MAX)))
    marker_sizes = _latitude_marker_sizes(lat_b, base_size=REGION_MARKER_BASE_SIZE)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for ax, (title, field) in zip(
        axes, ((TITLE_NOISE, iid[box]), (TITLE_STRUCTURE, era5[box]))
    ):
        sc = ax.scatter(
            lon_b, lat_b, c=field, s=marker_sizes, cmap="RdBu_r", vmin=vmin, vmax=vmax,
            rasterized=True, zorder=2,
        )
        for seg_lon, seg_lat in coastline_segments:
            ax.plot(seg_lon, seg_lat, color="0.08", lw=0.9, alpha=0.85, zorder=3)
        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)
        ax.set_aspect(aspect)
        ax.set_title(title, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.subplots_adjust(left=0.02, right=0.9, bottom=0.04, top=0.93, wspace=0.06)
    # Match the colour bar to the aspect-constrained panel height: realise the
    # geometry with a draw, then take the right panel's actual (post-aspect) box.
    fig.canvas.draw()
    panel = axes[1].get_position()
    cbar_ax = fig.add_axes([0.915, panel.y0, 0.017, panel.height])
    fig.colorbar(sc, cax=cbar_ax, label="2-metre temperature (standardised)")
    fig.savefig(OUT_REGION, dpi=200)
    print(f"wrote {OUT_REGION.relative_to(REPO_ROOT)}")


def main():
    OUT_REGION.parent.mkdir(parents=True, exist_ok=True)
    latlons, iid, era5 = _load()
    coastline_segments = _required_coastline_segments()
    make_global(latlons, iid, era5)
    make_region(latlons, iid, era5, coastline_segments)


if __name__ == "__main__":
    main()
