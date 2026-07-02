"""Phase 1/2 TOY report figures from the persisted stage_* artifacts.

Generates the synthetic-testbed figures the report includes, replacing the
old notebook exports so the report no longer depends on notebook execution
(notebooks 02/03 are now exploratory only):

  phase_2_stage_bc_nonsmearing_homoscedastic.png
      Non-smearing smear tail: Joint MAP (Method 1, lambda sweep) vs
      Mode-selection MRF (Method 4, beta sweep) vs the per-cell-MAP /
      smoothed-MAP / a* reference fields -- the >0.5 sigma (= dNLL > 0.125 nats)
      smear tail with within-field spatial-bootstrap CIs, vs scale-free
      roughness R~. (A former left panel showing mean dNLL to best mode was
      dropped: mean dNLL = NLL/N minus a fixed per-cell constant, i.e. the
      pareto figure's y-axis, so it duplicated phase_2_tv_pareto_plane.)
  phase_2_tv_pareto_plane_homoscedastic.png
      Faithfulness-coherence plane (NLL/N vs R~) for Joint MAP,
      Mode-selection MRF and the exact min-cut TV frontier, bracketed by the
      independent-draw / mixture-mean / smoothed-MAP / a* anchors. The confounded
      TV (Adam) arm is omitted from this main-text figure (it fails its own
      min-cut optimality certificate below lambda=0.2 and is dominated); it
      remains a pipeline quantity available for Appendix F.
  phase_2_stage_bc_pareto_homoscedastic.png
      Appendix full-sweep plane: every swept Joint MAP (lambda) and
      Mode-selection MRF (beta) configuration against the Stage A anchors,
      including the off-scale Independent draw the main-text plane omits.
      A display-name re-render of the retired notebook export of the same
      name, read from the same persisted stage_* artifacts.

House style -- colours, markers and display names -- is shared with the real
Phase-4 figures via `sampler_research.plotting`, so a reader can map toy -> real.
The ANALYSIS is ported verbatim from the notebooks: per-cell drift off mode is
the same `method4_mrf.delta_nll_to_best_mode` against Stage C's extracted modes,
and every (NLL/N, R~) point is read straight from the persisted stage_*
artifacts (no sweep is re-run here). The within-field bootstrap reuses
`faithfulness.bootstrap_cell_statistic` with the same draws/seed/threshold as the
real smear-tail CIs.

    .venv/bin/python scripts/make_toy_figures.py
    .venv/bin/python scripts/make_toy_figures.py --only nonsmearing
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sampler_research import method4_mrf as rm4
from sampler_research.faithfulness import bootstrap_cell_statistic, frac_exceeds
from sampler_research.io import load_sampler_arrays
from sampler_research.plotting import (
    _knee_index,
    _segment_arrows,
    _sweep_param_labels,
    display_name,
    field_style,
)

# Within-field spatial-bootstrap settings, kept identical to the real macro
# pipeline (scripts/make_phase4_figures.py BOOT_* / emit_report_results.py) so
# the toy and real smear-tail CIs are computed the same way, draw-for-draw.
BOOT_THRESHOLD = 0.125
BOOT_N_DRAWS = 2000
BOOT_SEED = 0
BOOT_CI = 0.95

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "outputs" / "data"
RUNS_DIR = REPO_ROOT / "outputs" / "runs"
FIG_DIR = REPO_ROOT / "outputs" / "figures"
DATASET = "phase_1_homoscedastic"

# Toy anchor key (stage_a / .npz) -> shared plotting style key. The toy keys are
# the bare baseline names; the plotting house style is keyed on the canonical
# real-figure keys, so iid -> iid_seed0 and smoothed_map -> smoothed_map_n10.
ANCHOR_STYLE_KEY = {
    "iid": "iid_seed0",
    "mode_map": "mode_map",
    "mixture_mean": "mixture_mean",
    "smoothed_map": "smoothed_map_n10",
    "a_star": "a_star",
}

# Method-sweep colours come from the shared style of the canonical field key, so
# the toy curves read in the same blue/green as Joint MAP / Mode-selection MRF in
# the real figures. Full-method sweep markers are circles throughout.
M1_COLOUR = field_style("m1_star")[0]
M4_COLOUR = field_style("m4_beta1")[0]
SWEEP_MARKER_SIZE = 6.0
# Toy-figure arrowhead size, passed to _segment_arrows so only these two figures
# grow; the shared default (SWEEP_ARROW_SIZE=50) still drives the Ch5 phase-4 figures.
TOY_ARROW_SIZE = 70

# Sweep-direction arrows (_segment_arrows) and first/best/last parameter labels
# (_sweep_param_labels, _knee_index) now live in sampler_research.plotting so the
# real Phase-4 figures share the exact same house style.


def _read_scores(path):
    """Read a persisted *_scores.csv into list-of-dict rows (float where possible)."""
    rows = []
    with open(path, newline="") as fh:
        for raw in csv.DictReader(fh):
            row = {}
            for key, value in raw.items():
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    row[key] = value
            rows.append(row)
    return rows


def load_artifacts(runs_dir=RUNS_DIR, data_dir=DATA_DIR, dataset=DATASET):
    """Load the toy GMM params + the persisted stage_a/b/c/tv arrays and tables.

    Pure read of the committed artifacts -- no sweep is re-run. Returns a dict
    holding the GMM (`pi`/`mu`/`sigma`), the four stage arrays (`A`/`B`/`C`/`T`),
    Stage A's per-anchor score rows, and the TV-cut score rows (for the
    variance-collapsed mask).
    """

    runs_dir, data_dir = Path(runs_dir), Path(data_dir)
    d = load_sampler_arrays(data_dir / f"{dataset}.npz")
    A = dict(np.load(runs_dir / "stage_a_baselines" / f"{dataset}_baselines.npz"))
    B = dict(np.load(runs_dir / "stage_b_regularised_map" / f"{dataset}_regularised_map.npz"))
    C = dict(np.load(runs_dir / "stage_c_method4_mrf" / f"{dataset}_method4_mrf.npz"))
    T = dict(np.load(runs_dir / "stage_b_tv_ablation" / f"{dataset}_tv_ablation.npz"))
    scores_a = {r["baseline"]: r
                for r in _read_scores(runs_dir / "stage_a_baselines" / "stage_a_scores.csv")}
    scores_t = _read_scores(runs_dir / "stage_b_tv_ablation" / "stage_b_tv_scores.csv")
    rows_cut = [r for r in scores_t if r["arm"] == "cut-tv"]
    return {
        "pi": d["pi"], "mu": d["mu"], "sigma": d["sigma"],
        "A": A, "B": B, "C": C, "T": T,
        "scores_a": scores_a, "rows_cut": rows_cut,
        # Stage C's extracted modes are the shared reference for every field's
        # drift-off-mode score (cross-method, exactly as in the notebooks).
        "mode_values": C["mode_values"], "valid_mask": C["valid_mask"],
    }


def smear(field, art):
    """Per-cell drift of `field` off its best Stage-C mode (notebook-verbatim).

    Thin wrapper over `method4_mrf.delta_nll_to_best_mode` against the shared
    extracted modes; returns its dict (`mean`, `frac_over`, `per_cell`, ...).
    """

    return rm4.delta_nll_to_best_mode(
        field, art["pi"], art["mu"], art["sigma"],
        art["mode_values"], art["valid_mask"],
    )


def smear_tail_ci(per_cell):
    """Within-field spatial-bootstrap CI of frac(dNLL > 0.125) over a field's cells.

    Same statistic/draws/seed/threshold as the real `_bimodal_frac_ci`. With only
    64 cells the toy CI is legitimately wide -- that width is honest, not a bug.
    """

    return bootstrap_cell_statistic(
        np.asarray(per_cell, dtype=float).reshape(-1),
        lambda v: float(frac_exceeds(v, BOOT_THRESHOLD)),
        n_boot=BOOT_N_DRAWS, ci=BOOT_CI, rng=np.random.default_rng(BOOT_SEED),
    )


def _anchor(ax, art, key, y, *, ci=None, annotate_xy=(5, 4), annotate_ha="left"):
    """Scatter (or error-bar) one Stage A anchor in the shared house style."""
    style_key = ANCHOR_STYLE_KEY[key]
    colour, marker, _ = field_style(style_key)
    x = float(art["scores_a"][key]["r_tilde"])
    if ci is None:
        ax.scatter(x, y, marker=marker, s=90, color=colour,
                   edgecolor="k", linewidth=0.6, zorder=5)
    else:
        ax.errorbar(x, y, yerr=[[y - ci["lo"]], [ci["hi"] - y]],
                    fmt=marker, ms=9, color=colour, ecolor=colour, capsize=3,
                    markeredgecolor="k", markeredgewidth=0.6, zorder=5)
    ax.annotate(
        display_name(style_key), (x, y), fontsize=7,
        xytext=annotate_xy, textcoords="offset points", ha=annotate_ha,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.8),
    )


def fig_nonsmearing(art, fig_dir=FIG_DIR):
    """Smear-tail panel -> phase_2_stage_bc_nonsmearing_homoscedastic.png."""
    B, C = art["B"], art["C"]
    b_delta = [smear(f, art) for f in B["fields"]]
    c_delta = [smear(f, art) for f in C["fields"]]

    fig, ax2 = plt.subplots(figsize=(8, 5.5))

    # The >0.5 sigma smear tail (frac dNLL > 0.125 nats) with within-field
    # spatial-bootstrap CIs -- the panel the bare (NLL/N, R~) plane cannot show.
    def tail(ax, r_tilde, deltas, fmt, colour, label):
        fracs = [d["frac_over"][0] for d in deltas]
        cis = [smear_tail_ci(d["per_cell"]) for d in deltas]
        lo = [f - c["lo"] for f, c in zip(fracs, cis)]
        hi = [c["hi"] - f for f, c in zip(fracs, cis)]
        ax.errorbar(r_tilde, fracs, yerr=[lo, hi], fmt=fmt, color=colour, lw=2,
                    ms=SWEEP_MARKER_SIZE, capsize=2, elinewidth=0.8,
                    label=label)

    tail(ax2, B["r_tilde"], b_delta, "o-", M1_COLOUR, "Joint MAP ($\\lambda:0\\to2$)")
    tail(ax2, C["r_tilde"], c_delta, "o-", M4_COLOUR, "Mode-selection MRF ($\\beta:0\\to0.1$)")
    # The Independent draw (iid) anchor is omitted from this plot: at R~ = 2.077 it sits
    # far right of the working region and stretches the axis, hiding the methods. Its
    # scores stay in Table 4.1 (tab:toy-baselines); the omission is noted in the report
    # prose (sec:toy-behaviour) and this figure's caption.
    anchor_labels_right = {
        "mixture_mean": dict(annotate_xy=(6, 4)),
        "mode_map": dict(annotate_xy=(-8, 4), annotate_ha="right"),
        "a_star": dict(annotate_xy=(6, 4)),
        "smoothed_map": dict(annotate_xy=(6, 4)),
    }
    for key in ("mixture_mean", "mode_map", "a_star", "smoothed_map"):
        d = smear(art["A"][key], art)
        _anchor(ax2, art, key, d["frac_over"][0], ci=smear_tail_ci(d["per_cell"]),
                **anchor_labels_right[key])
    ax2.set_xscale("linear")
    ax2.set_xlabel("Normalised roughness ($\\tilde{R}$)")
    ax2.set_ylabel("fraction of cells $>0.5\\sigma$ off mode")
    ax2.set_title("Smear tail past $0.5\\sigma$ (95% within-field CI)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7, loc="upper center")

    fig.tight_layout()
    _segment_arrows(ax2, B["r_tilde"], [d["frac_over"][0] for d in b_delta],
                    M1_COLOUR, n_arrows=2, outline=True, arrow_size=TOY_ARROW_SIZE)
    _segment_arrows(ax2, C["r_tilde"], [d["frac_over"][0] for d in c_delta],
                    M4_COLOUR, n_arrows=2, outline=True, arrow_size=TOY_ARROW_SIZE)

    # Parameter-value labels on the first / best (knee) / last sweep circles.
    # The best index is found once from each method's faithfulness-coherence
    # objective so both panels flag the same lambda*/beta*.
    b_best = _knee_index(B["r_tilde"], B["nll_over_n"])  # lambda* = 0.2
    c_best = _knee_index(C["r_tilde"], C["nll_over_n"])  # beta*  = 0.05
    _sweep_param_labels(
        ax2, B["r_tilde"], [d["frac_over"][0] for d in b_delta], B["lambdas"],
        b_best, "\\lambda", M1_COLOUR, placements={
            "first": dict(xytext=(6, -2), ha="left", va="center"),
            "best": dict(xytext=(-2, -4), ha="right", va="center"),
            "last": dict(xytext=(-7, -5), ha="right", va="bottom"),
        },
    )
    _sweep_param_labels(
        ax2, C["r_tilde"], [d["frac_over"][0] for d in c_delta], C["betas"],
        c_best, "\\beta", M4_COLOUR, placements={
            "first": dict(xytext=(3, 1), ha="left", va="bottom"),
            "best": dict(xytext=(-17, -5), ha="center", va="center"),
            "last": dict(xytext=(3, -5), ha="left", va="center"),
        },
    )
    out = Path(fig_dir) / "phase_2_stage_bc_nonsmearing_homoscedastic.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")
    return out


def fig_pareto_plane(art, fig_dir=FIG_DIR):
    """Faithfulness-coherence plane -> phase_2_tv_pareto_plane_homoscedastic.png."""
    B, C, T = art["B"], art["C"], art["T"]
    cut_colour = field_style("tv_cut")[0]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Joint MAP (quadratic lambda sweep, Stage B). Sweep direction and the
    # first / best (knee) / last lambda values are added below, in the same
    # house style as the non-smearing figure.
    ax.plot(B["r_tilde"], B["nll_over_n"], "o-", color=M1_COLOUR, lw=2,
            ms=SWEEP_MARKER_SIZE,
            label="Joint MAP (quad $\\lambda:0\\to2$)")

    # Mode-selection MRF (discrete beta sweep, Stage C).
    ax.plot(C["r_tilde"], C["nll_over_n"], "o-", color=M4_COLOUR, lw=2,
            ms=SWEEP_MARKER_SIZE,
            label="Mode-selection MRF ($\\beta:0\\to0.1$)")

    # TV (Adam) is intentionally omitted from this main-text figure: it is
    # confounded (fails its own min-cut optimality certificate below lambda=0.2)
    # and dominated on this plane. The arm is still produced by the pipeline
    # (run_stage_b_tv_ablation.py) and may appear in Appendix F; see
    # research_notes/phase_2.md and report_plan.md for the rationale.

    # TV (exact) min-cut frontier; variance-collapsed points (lambda >= 2) have
    # an undefined R~ and are excluded, exactly as in the notebook. Drawn with
    # diamond markers on a dashed line so it never relies on colour alone to
    # separate from the Mode-selection MRF where the two overlap near the origin.
    collapsed = np.array([r["variance_collapsed"] == "True" for r in art["rows_cut"]])
    cut_r = T["cut_tv_r_tilde"][~collapsed]
    cut_nll = T["cut_tv_nll_over_n"][~collapsed]
    ax.plot(cut_r, cut_nll, "D--", color=cut_colour, lw=2,
            ms=SWEEP_MARKER_SIZE,
            label=display_name("tv_cut") + " ($\\lambda:0\\to1$ shown)")

    # Bracket anchors (mixture-mean over-smooth, smoothed-MAP low-roughness, a*). The
    # Independent draw (iid) is omitted here too: at R~ = 2.077 it lies far off-scale
    # (see fig_nonsmearing and Table 4.1); the report prose and caption note the omission.
    anchor_labels = {
        "mixture_mean": dict(annotate_xy=(6, 4)),
        "smoothed_map": dict(annotate_xy=(6, 4)),
        "a_star": dict(annotate_xy=(0, 14), annotate_ha="center"),
    }
    for key in ("mixture_mean", "smoothed_map", "a_star"):
        _anchor(
            ax, art, key, float(art["scores_a"][key]["nll_over_n"]),
            **anchor_labels[key],
        )

    ax.set_xscale("linear")
    ax.set_xlabel("Normalised roughness ($\\tilde{R} = S_{\\mathrm{edge}}/\\mathrm{Var}(V)$)")
    ax.set_ylabel("NLL/$N$ (nats)")
    # Title names the plane and the reading direction only; the interpretive
    # findings (TV collapse, the quadratic reaching R~ ~= 0.25) live in the
    # report caption (fig:toy-tv-pareto in report/thesis.tex).
    ax.set_title("Faithfulness–coherence plane (lower-left is better)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Sweep-direction arrows (outline chevrons) on all three curves, matching
    # the non-smearing figure.
    _segment_arrows(ax, B["r_tilde"], B["nll_over_n"], M1_COLOUR,
                    n_arrows=2, outline=True, arrow_size=TOY_ARROW_SIZE)
    _segment_arrows(ax, C["r_tilde"], C["nll_over_n"], M4_COLOUR,
                    n_arrows=2, outline=True, arrow_size=TOY_ARROW_SIZE)
    _segment_arrows(ax, cut_r, cut_nll, cut_colour, n_arrows=2, outline=True,
                    arrow_size=TOY_ARROW_SIZE)

    # First / best (knee) / last parameter-value labels on the two method
    # sweeps. Here best is the literal lower-left knee of this plane.
    b_best = _knee_index(B["r_tilde"], B["nll_over_n"])  # lambda* = 0.2
    c_best = _knee_index(C["r_tilde"], C["nll_over_n"])  # beta*  = 0.05
    _sweep_param_labels(
        ax, B["r_tilde"], B["nll_over_n"], B["lambdas"], b_best,
        "\\lambda", M1_COLOUR, placements={
            "first": dict(xytext=(8, -3), ha="left", va="center"),
            "best": dict(xytext=(-6, -2), ha="right", va="center"),
            "last": dict(xytext=(-6, 3), ha="right", va="center"),
        },
    )
    _sweep_param_labels(
        ax, C["r_tilde"], C["nll_over_n"], C["betas"], c_best,
        "\\beta", M4_COLOUR, placements={
            # beta=0 sits on TV's lambda_TV=0; stack them (beta above, TV below).
            "first": dict(xytext=(8, 5), ha="left", va="bottom"),
            "best": dict(xytext=(0, -7), ha="center", va="top"),
            "last": dict(xytext=(5, 8), ha="left", va="bottom"),
        },
    )
    cut_lambdas = T["cut_tv_lambdas"][~collapsed]
    cut_best = _knee_index(cut_r, cut_nll)  # lambda_TV = 0.1
    _sweep_param_labels(
        ax, cut_r, cut_nll, cut_lambdas, cut_best,
        "\\lambda_{\\mathrm{TV}}", cut_colour, placements={
            "first": dict(xytext=(6, -2), ha="left", va="top"),
            "best": dict(xytext=(0, -10), ha="center", va="center"),
            "last": dict(xytext=(3, 0), ha="left", va="bottom"),
        },
    )
    out = Path(fig_dir) / "phase_2_tv_pareto_plane_homoscedastic.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")
    return out


def fig_full_plane(art, fig_dir=FIG_DIR):
    """Appendix full-sweep plane -> phase_2_stage_bc_pareto_homoscedastic.png.

    The complete lambda+beta grids behind the Stage B/C selections: both full
    method sweeps on the (NLL/N, R~) plane against the Stage A anchors,
    including the Independent draw at R~ = 2.077 that the main-text figures
    omit as off-scale. Content matches the retired notebook export it
    replaces; only the display names and house style change (no sweep is
    re-run here).
    """
    B, C = art["B"], art["C"]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(B["r_tilde"], B["nll_over_n"], "o-", color=M1_COLOUR, lw=2,
            ms=SWEEP_MARKER_SIZE,
            label="Joint MAP (quad $\\lambda:0\\to2$)")
    ax.plot(C["r_tilde"], C["nll_over_n"], "o-", color=M4_COLOUR, lw=2,
            ms=SWEEP_MARKER_SIZE,
            label="Mode-selection MRF ($\\beta:0\\to0.1$)")

    # All four anchors of the notebook export, iid included: showing the full
    # bracket is the point of this appendix panel.
    anchor_labels = {
        "iid": dict(annotate_xy=(-8, 4), annotate_ha="right"),
        "mode_map": dict(annotate_xy=(-8, 4), annotate_ha="right"),
        "a_star": dict(annotate_xy=(0, 14), annotate_ha="center"),
        "smoothed_map": dict(annotate_xy=(6, 4)),
    }
    for key, placement in anchor_labels.items():
        _anchor(ax, art, key, float(art["scores_a"][key]["nll_over_n"]),
                **placement)

    ax.set_xscale("linear")
    ax.set_xlabel("Normalised roughness ($\\tilde{R} = S_{\\mathrm{edge}}/\\mathrm{Var}(V)$)")
    ax.set_ylabel("NLL/$N$ (nats)")
    ax.set_title("Full $\\lambda$ and $\\beta$ sweeps (lower-left is better)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    _segment_arrows(ax, B["r_tilde"], B["nll_over_n"], M1_COLOUR,
                    n_arrows=2, outline=True, arrow_size=TOY_ARROW_SIZE)
    _segment_arrows(ax, C["r_tilde"], C["nll_over_n"], M4_COLOUR,
                    n_arrows=2, outline=True, arrow_size=TOY_ARROW_SIZE)

    b_best = _knee_index(B["r_tilde"], B["nll_over_n"])  # lambda* = 0.2
    c_best = _knee_index(C["r_tilde"], C["nll_over_n"])  # beta*  = 0.05
    _sweep_param_labels(
        ax, B["r_tilde"], B["nll_over_n"], B["lambdas"], b_best,
        "\\lambda", M1_COLOUR, placements={
            "first": dict(xytext=(8, -3), ha="left", va="center"),
            "best": dict(xytext=(-6, -2), ha="right", va="center"),
            "last": dict(xytext=(-6, 3), ha="right", va="center"),
        },
    )
    _sweep_param_labels(
        ax, C["r_tilde"], C["nll_over_n"], C["betas"], c_best,
        "\\beta", M4_COLOUR, placements={
            "first": dict(xytext=(3, 8), ha="left", va="bottom"),
            "best": dict(xytext=(0, -8), ha="center", va="top"),
            "last": dict(xytext=(-3, 8), ha="right", va="bottom"),
        },
    )
    out = Path(fig_dir) / "phase_2_stage_bc_pareto_homoscedastic.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")
    return out


FIGURES = {"nonsmearing": fig_nonsmearing, "pareto": fig_pareto_plane,
           "fullplane": fig_full_plane}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    parser.add_argument("--only", nargs="+", choices=sorted(FIGURES),
                        help="regenerate only these figures (default: both)")
    args = parser.parse_args()

    art = load_artifacts(args.runs_dir, args.data_dir)
    for name in (args.only or sorted(FIGURES)):
        FIGURES[name](art, args.fig_dir)


if __name__ == "__main__":
    main()
