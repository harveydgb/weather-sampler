"""Phase 4 report figures from the persisted runner artifacts (phase_4_plan S5).

Run after `scripts/run_phase4_real.py` (all stages + --lambda-star):

    .venv/bin/python scripts/make_phase4_figures.py

Writes to outputs/figures/:
  phase_4_maps.png               4/5-panel global Mollweide
  phase_4_pareto_smear.png       (NLL/N, R~) plane + smear-fraction panel
  phase_4_variogram.png          sampled spherical variograms (descriptive)
  phase_4_bimodal_enrichment.png W1 MUST: dNLL>0.125 enrichment in the audit
                                 S4 bimodal masks + Mollweide dNLL map

Figures are generated here (not in notebook 04) so the MUST figure task does
not depend on notebook execution; the notebook displays these files.
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sampler_research.io import load_real_marginal
from sampler_research.plotting import plot_mollweide_fields, plot_variograms

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "outputs" / "runs" / "phase_4_real"
FIG_DIR = REPO_ROOT / "outputs" / "figures"
DATA_NPZ = REPO_ROOT / "outputs" / "data" / "phase_4_real_2t.npz"


def _load_rows():
    with open(RUN_DIR / "scores.csv") as fh:
        return list(csv.DictReader(fh))


def _mollweide_scatter(ax, latlons, values, title, cmap="viridis", vmin=None, vmax=None):
    sc = ax.scatter(
        np.radians(latlons[:, 1]), np.radians(latlons[:, 0]),
        c=values, s=0.5, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True,
    )
    ax.set_title(title, fontsize=10)
    ax.grid(True, lw=0.3, alpha=0.4)
    return sc


def fig_maps(latlons, star):
    fields = {}
    with np.load(RUN_DIR / "anchors.npz") as f:
        fields["MAP anchor"] = f["mode_map"]
        fields["iid draw (seed 0)"] = f["iid_seed0"]
    with np.load(RUN_DIR / "method1_sensitivity.npz") as f:
        fields[f"Method 1 @ lambda*={star['lambda_star']:.0f}"] = f["field_star"]
    with np.load(RUN_DIR / "anchors.npz") as f:
        fields["smoothed-MAP (n=10)"] = f["smoothed_map_n10"]
    m4_path = RUN_DIR / "method4_sweep.npz"
    if m4_path.exists():
        rows = _load_rows()
        star_rt = float(next(r for r in rows if r["name"] == "m1_star")["r_tilde"])
        m4_rows = [r for r in rows if r["kind"] == "method4"]
        if m4_rows:
            nearest = min(m4_rows, key=lambda r: abs(float(r["r_tilde"]) - star_rt))
            with np.load(m4_path) as f:
                idx = int(np.argmin(np.abs(f["betas"] - float(nearest["param"]))))
                fields[f"Method 4 @ beta={nearest['param']}"] = f["fields"][idx]
    fig, _ = plot_mollweide_fields(
        latlons, fields,
        suptitle="Phase 4: real O96 2t fields (standardised units, single snapshot)",
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # Left: Pareto plane, log R~ axis (R~ is trend-compressed on the real grid).
    ax1.plot(
        [float(r["r_tilde"]) for r in m1], [float(r["nll_over_n"]) for r in m1],
        "o-", color="tab:blue", label="Method 1 (lambda sweep)",
    )
    if m4:
        ax1.plot(
            [float(r["r_tilde"]) for r in m4], [float(r["nll_over_n"]) for r in m4],
            "s--", color="tab:green", label="Method 4 (beta sweep)",
        )
    for r in anchors:
        ax1.scatter(float(r["r_tilde"]), float(r["nll_over_n"]), marker="*", s=110, zorder=5)
        ax1.annotate(r["name"], (float(r["r_tilde"]), float(r["nll_over_n"])),
                     fontsize=7, xytext=(4, 4), textcoords="offset points")
    if star_row is not None:
        ax1.scatter(float(star_row["r_tilde"]), float(star_row["nll_over_n"]),
                    marker="D", s=70, color="tab:red", zorder=6, label="M1 @ lambda*")
    ax1.axvline(star["target_r_tilde"], color="grey", lw=0.8, ls=":",
                label="smoothed-MAP n10 R~ (target)")
    ax1.set_xscale("log")
    ax1.set_xlabel("R~ = S_edge / Var_V (log axis)")
    ax1.set_ylabel("NLL/N (nats; negative on standardised data)")
    ax1.set_title("Faithfulness vs declared coherence")
    ax1.legend(fontsize=7)

    # Right: smear fraction vs R~ -- the panel the bare Pareto cannot show.
    for series, suffix, ls in ((m1, "", "-"), (m1, "__bimodal", "--")):
        ax2.plot(
            [float(r["r_tilde"]) for r in series],
            [float(r[f"dnll_frac_gt_0p125{suffix}"]) for r in series],
            "o" + ls, color="tab:blue",
            label=f"Method 1 {'bimodal stratum' if suffix else 'global'}",
        )
    if m4:
        for suffix, ls in (("", "-"), ("__bimodal", "--")):
            ax2.plot(
                [float(r["r_tilde"]) for r in m4],
                [float(r[f"dnll_frac_gt_0p125{suffix}"]) for r in m4],
                "s" + ls, color="tab:green",
                label=f"Method 4 {'bimodal stratum' if suffix else 'global'}",
            )
    for r in anchors:
        if r["name"] in ("smoothed_map_n10", "iid_seed0", "mode_map"):
            ax2.scatter(float(r["r_tilde"]), float(r["dnll_frac_gt_0p125"]),
                        marker="*", s=110, zorder=5)
            ax2.annotate(r["name"], (float(r["r_tilde"]), float(r["dnll_frac_gt_0p125"])),
                         fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax2.axvline(star["target_r_tilde"], color="grey", lw=0.8, ls=":")
    ax2.set_xscale("log")
    ax2.set_xlabel("R~ (log axis)")
    ax2.set_ylabel("fraction of cells with dNLL > 0.125 nats")
    ax2.set_title("Smear tail vs coherence (global and bimodal)")
    ax2.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase_4_pareto_smear.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote phase_4_pareto_smear.png")


def fig_variogram():
    with np.load(RUN_DIR / "variograms.npz") as f:
        centres = f["centres"]
        variograms = {k: f[k] for k in f.files if k != "centres"}
    fig, ax = plot_variograms(
        centres, variograms,
        xlabel="angular distance (deg)",
        title="Sampled spherical variograms (descriptive, bracket-anchored)",
    )
    ax.set_xlim(0, 60)
    fig.savefig(FIG_DIR / "phase_4_variogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote phase_4_variogram.png")


def fig_bimodal_enrichment(latlons, star):
    with np.load(RUN_DIR / "masks.npz") as f:
        bimodal_2s = f["bimodal"]
        bimodal_1s = f["bimodal_1sigma"]
    with np.load(RUN_DIR / "delta_per_cell.npz") as f:
        deltas = {k: f[k] for k in f.files}

    names = ["iid_seed0", "smoothed_map_n10", "m1_star", "m1_star_double"]
    names += [n for n in deltas if n.startswith("m4_beta")][-1:]
    names = [n for n in names if n in deltas]

    fig = plt.figure(figsize=(13, 4.6))
    ax1 = fig.add_subplot(1, 2, 1)
    width = 0.35
    xs = np.arange(len(names))
    for offset, (mask, label) in enumerate(
        ((bimodal_1s, "bimodal >1 sigma (5,576 cells)"), (bimodal_2s, "bimodal >2 sigma (604 cells)"))
    ):
        enrich = []
        for n in names:
            d = deltas[n]
            global_frac = float(np.mean(d > 0.125))
            mask_frac = float(np.mean(d[mask] > 0.125))
            enrich.append(mask_frac / global_frac if global_frac > 0 else np.nan)
        ax1.bar(xs + (offset - 0.5) * width, enrich, width, label=label)
    ax1.axhline(1.0, color="grey", lw=0.8, ls=":")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax1.set_ylabel("enrichment of dNLL > 0.125 cells\n(in-mask frac / global frac)")
    ax1.set_title("Smear concentrates on the bimodal subset")
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(1, 2, 2, projection="mollweide")
    d = deltas["m1_star"]
    sc = _mollweide_scatter(
        ax2, latlons, np.log10(np.maximum(d, 1e-6)),
        f"log10 dNLL-to-best-mode, Method 1 @ lambda*={star['lambda_star']:.0f}",
        cmap="magma", vmin=-4, vmax=1,
    )
    fig.colorbar(sc, ax=ax2, orientation="horizontal", pad=0.05, shrink=0.8,
                 label="log10 dNLL (nats)")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase_4_bimodal_enrichment.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote phase_4_bimodal_enrichment.png")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = load_real_marginal(DATA_NPZ)
    star = json.loads((RUN_DIR / "lambda_star.json").read_text())
    fig_maps(data["latlons"], star)
    fig_pareto_smear(star)
    fig_variogram()
    fig_bimodal_enrichment(data["latlons"], star)


if __name__ == "__main__":
    main()
