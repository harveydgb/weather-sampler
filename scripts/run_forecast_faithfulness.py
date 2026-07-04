"""Marginal faithfulness & do-no-harm per lead, both runs (plan Step 4 / report S5.5).

Builds the S5.5 numbers that did not previously exist in the tree. For each
forecast lead x run it reads the converted GMM npz + the persisted sampler
artifacts (anchors.npz, method1_sensitivity.npz, method1_sweep.npz,
lambda_star.json) and computes:

  * iid do-no-harm (stochastic baseline ONLY): delta CRPS = ensemble CRPS -
    analytic GMM CRPS (~0), and PIT-of-an-iid-draw KS (~0, the Uniform identity);
  * Method 1 @ lambda* MARGINAL POSITION (deterministic field): PIT KS, central
    coverage, mean |PIT - 0.5| -- a marginal-position/coverage diagnostic, never
    an ensemble-calibration claim (report non-claim #3);
  * faithfulness-vs-lambda: coverage + NLL/N along the M1 lambda sweep, for the
    S5.4 "faithfulness figure first".

Outputs: per-lead `faithfulness.npz` in each run dir; one combined
`faithfulness_by_lead.csv` (run column); one faithfulness-vs-lambda figure per run.

    .venv/bin/python scripts/run_forecast_faithfulness.py            # both runs
    .venv/bin/python scripts/run_forecast_faithfulness.py --prefix phase_4_fc48_14ep
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from sampler_research.baselines import gmm_nll_over_n
from sampler_research import faithfulness as fth
from sampler_research.gmm import sample_iid_gmm
from sampler_research.io import load_real_marginal

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "outputs" / "data"
RUNS_DIR = REPO_ROOT / "outputs" / "runs"
OUT_DIR = REPO_ROOT / "outputs" / "runs" / "phase_4_forecast_faithfulness"
FIG_DIR = REPO_ROOT / "outputs" / "figures"

SIXEP_SPEC = {"prefix": "phase_4_fc48_6ep", "label": "6ep", "column": "SixEp"}

# Converged = v2 me7 (14 ep). F1 track-2 replicates: the SAME trained model at the
# twelve first-of-month 2023 init dates (one synoptic case per month, spanning all
# four calendar seasons; a descriptive span only -- seasonal/global generality
# stays a non-claim, n=1/month, single year).
# Canonical init A (= 2023-11-01) gives the unchanged single-init headline rows;
# all present inits feed the across-init mean + range (`kind='init_mean'`).
CONVERGED_LABEL = "14ep"
CONVERGED_COLUMN = "Converged"
# Across-init set resolved by the SAME rule as scripts/emit_report_results.py
# (headline prefix + `init2023MM01` glob) so the faithfulness/softening aggregate
# and the emitted comparison gap can never desync. The glob excludes the deprecated
# pilot inits B (..._init20231010T12) and C (..._init20231215) -- not first-of-month.
CONVERGED_HEADLINE_PREFIX = "phase_4_fc48_14ep"       # init A = 2023-11-01 (canonical)
CONVERGED_REPLICATE_GLOB = "phase_4_fc48_v2_init2023??01"


def _discover_converged_prefixes(data_dir):
    """Converged-init prefixes with per-lead npz in `data_dir`, canonical-first.

    Headline init A first, then the first-of-month `init2023MM01` replicates
    (sorted). Mirrors `_discover_converged_prefixes` in emit_report_results.py.
    """
    data_dir = Path(data_dir)
    prefixes = []
    if any(data_dir.glob(f"{CONVERGED_HEADLINE_PREFIX}_step*_2t.npz")):
        prefixes.append(CONVERGED_HEADLINE_PREFIX)
    seen = set()
    for path in sorted(data_dir.glob(f"{CONVERGED_REPLICATE_GLOB}_step*_2t.npz")):
        prefix = path.name.rsplit("_step", 1)[0]
        if prefix not in seen:
            seen.add(prefix)
            prefixes.append(prefix)
    return prefixes
N_MEMBERS = 50

BASE_FIELDS = ("run", "column", "prefix", "kind", "n_inits", "init_datetime",
               "step", "lead_hours")
METRIC_FIELDS = (
    "lambda_star",
    "iid_delta_crps", "iid_ensemble_crps", "iid_analytic_crps", "iid_pit_ks", "iid_n_members",
    "m1_star_pit_ks", "m1_star_cov50", "m1_star_cov90", "m1_star_mean_abs_pit_centre",
    "m1_star_nll_over_n",
)
# Report-quoted metrics that get an across-init min-max range (lambda*, do-no-harm
# delta CRPS, M1 marginal-position PIT-KS, M1 central-90% coverage).
FAITH_SPREAD_KEYS = ("lambda_star", "iid_delta_crps", "m1_star_pit_ks", "m1_star_cov90")
CSV_FIELDS = (
    BASE_FIELDS + METRIC_FIELDS
    + tuple(f"{k}{s}" for k in FAITH_SPREAD_KEYS for s in ("_lo", "_hi"))
)


def _lead_files(data_dir, prefix):
    found = []
    for path in sorted(Path(data_dir).glob(f"{prefix}_step*_2t.npz")):
        try:
            step = int(path.stem.split("_step")[1].split("_2t")[0])
        except (IndexError, ValueError):
            continue
        found.append((step, path))
    return sorted(found, key=lambda item: item[0])


def faithfulness_for_lead(data_npz, run_dir, *, n_members, seed):
    """One CSV row + one npz payload for a single lead's run dir."""

    data = load_real_marginal(data_npz)
    pi, mu, sigma = data["pi"], data["mu"], data["sigma"]
    rng = np.random.default_rng(seed)

    # --- iid stochastic baseline: do-no-harm delta CRPS + PIT-uniformity identity
    crps = fth.delta_crps_iid(pi, mu, sigma, n_members=n_members, rng=rng)
    iid_draw, _ = sample_iid_gmm(pi, mu, sigma, rng)
    iid_pit = fth.pit_values(iid_draw, pi, mu, sigma)
    iid_pit_ks = fth.ks_uniform(iid_pit)

    # --- Method 1 @ lambda* marginal position (deterministic field)
    star_path = run_dir / "lambda_star.json"
    lambda_star = None
    if star_path.exists():
        lambda_star = json.loads(star_path.read_text()).get("lambda_star")
    m1_row = {"m1_star_pit_ks": np.nan, "m1_star_cov50": np.nan, "m1_star_cov90": np.nan,
              "m1_star_mean_abs_pit_centre": np.nan, "m1_star_nll_over_n": np.nan}
    pit_m1 = None
    sens_path = run_dir / "method1_sensitivity.npz"
    if sens_path.exists():
        with np.load(sens_path) as f:
            field_star = f["field_star"]
        ff = fth.field_faithfulness(field_star, pi, mu, sigma)
        pit_m1 = fth.pit_values(field_star, pi, mu, sigma)
        m1_row = {
            "m1_star_pit_ks": ff["pit_ks"], "m1_star_cov50": ff["cov50"],
            "m1_star_cov90": ff["cov90"], "m1_star_mean_abs_pit_centre": ff["mean_abs_pit_centre"],
            "m1_star_nll_over_n": gmm_nll_over_n(field_star, pi, mu, sigma),
        }

    # --- faithfulness-vs-lambda along the M1 sweep (for the S5.4 figure)
    sweep_lambdas = np.array([])
    sweep_cov50 = sweep_cov90 = sweep_mac = sweep_nll = np.array([])
    sweep_path = run_dir / "method1_sweep.npz"
    if sweep_path.exists():
        with np.load(sweep_path) as f:
            sweep_lambdas = np.asarray(f["lambdas"], dtype=float)
            fields = f["fields"]
        cov50, cov90, mac, nll = [], [], [], []
        for field in fields:
            pit = fth.pit_values(field, pi, mu, sigma)
            cov = fth.quantile_coverage(pit)
            cov50.append(cov[0.5]); cov90.append(cov[0.9])
            mac.append(fth.mean_abs_pit_centre(pit))
            nll.append(gmm_nll_over_n(field, pi, mu, sigma))
        order = np.argsort(sweep_lambdas)
        sweep_lambdas = sweep_lambdas[order]
        sweep_cov50 = np.array(cov50)[order]; sweep_cov90 = np.array(cov90)[order]
        sweep_mac = np.array(mac)[order]; sweep_nll = np.array(nll)[order]

    npz_payload = {
        "iid_pit": iid_pit,
        "lambdas": sweep_lambdas, "sweep_cov50": sweep_cov50, "sweep_cov90": sweep_cov90,
        "sweep_mean_abs_pit_centre": sweep_mac, "sweep_nll_over_n": sweep_nll,
        "lambda_star": np.float64(lambda_star if lambda_star is not None else np.nan),
    }
    if pit_m1 is not None:
        npz_payload["pit_m1_star"] = pit_m1

    row = {
        "lambda_star": lambda_star,
        "iid_delta_crps": crps["delta_crps"], "iid_ensemble_crps": crps["ensemble_crps"],
        "iid_analytic_crps": crps["analytic_crps"], "iid_pit_ks": iid_pit_ks,
        "iid_n_members": crps["n_members"],
        **m1_row,
    }
    return row, npz_payload


def _num(value):
    """Coerce a CSV/metric value to a finite float, else None (skipped in means)."""

    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def rows_for_prefix(prefix, label, column, kind, *, data_dir, runs_dir, n_members, seed):
    """Per-lead faithfulness rows + figure payloads for one init prefix.

    Reads each lead's converted npz + persisted sampler run dir
    (`{prefix}_step{k}`). Returns `(rows, fig_rows, payloads)`; `rows` is empty
    if no per-lead npz exist for the prefix (a not-yet-run init is skipped).
    """

    rows, fig_rows, payloads = [], [], {}
    for step, npz in _lead_files(data_dir, prefix):
        run_dir = runs_dir / f"{prefix}_step{step}"
        if not run_dir.exists():
            print(f"[faithfulness] missing run dir {run_dir}; run the per-lead audit first")
            continue
        meta = json.loads(npz.with_name(npz.stem + "_meta.json").read_text())
        row, payload = faithfulness_for_lead(npz, run_dir, n_members=n_members, seed=seed + step)
        row = {"run": label, "column": column, "prefix": prefix, "kind": kind,
               "n_inits": 1, "init_datetime": meta.get("init_datetime"),
               "step": step, "lead_hours": meta.get("lead_hours", 6.0 * step), **row}
        np.savez(run_dir / "faithfulness.npz", **payload)
        rows.append(row)
        fig_rows.append(row)
        payloads[step] = payload
        print(f"[faithfulness] {label} {prefix} +{row['lead_hours']:g}h: "
              f"iid dCRPS={row['iid_delta_crps']:+.4f} iid PIT-KS={row['iid_pit_ks']:.4f} "
              f"M1* cov90={row['m1_star_cov90']:.3f} PIT-KS={row['m1_star_pit_ks']:.3f}")
    return rows, fig_rows, payloads


def aggregate_converged_faith(per_init):
    """Across-init MEAN of every metric + min-max range for the report metrics.

    `per_init`: list of (prefix, rows). Emits one `kind='init_mean'` row per lead
    step: base metric columns hold the mean over the inits, and `{k}_lo`/`{k}_hi`
    (for `FAITH_SPREAD_KEYS`) the min/max -- a range over a few synoptic cases,
    not a sampling CI. Fewer than two inits -> [].
    """

    if len(per_init) < 2:
        return []
    by_step = {}
    for _prefix, rows in per_init:
        for r in rows:
            by_step.setdefault(int(r["step"]), []).append(r)
    out = []
    for step in sorted(by_step):
        group = by_step[step]
        inits = sorted({str(r.get("init_datetime")) for r in group})
        agg = {"run": CONVERGED_LABEL, "column": CONVERGED_COLUMN,
               "prefix": f"{CONVERGED_HEADLINE_PREFIX}+{len(per_init) - 1}",
               "kind": "init_mean", "n_inits": len(group),
               "init_datetime": ";".join(inits), "step": step,
               "lead_hours": group[0]["lead_hours"]}
        for key in METRIC_FIELDS:
            vals = [v for v in (_num(r.get(key)) for r in group) if v is not None]
            if not vals:
                agg[key] = ""
                continue
            agg[key] = sum(vals) / len(vals)
            if key in FAITH_SPREAD_KEYS:
                agg[f"{key}_lo"] = min(vals)
                agg[f"{key}_hi"] = max(vals)
        out.append(agg)
    return out


def make_figure(rows, payloads, label, fig_path, cov90_spread=None):
    """Two-panel faithfulness-vs-lambda figure for one run.

    ``cov90_spread`` (optional): ``{step: (mean, lo, hi)}`` across-init central-90%
    coverage summary (Option A), built from the ``kind='init_mean'`` aggregate
    rows. When given, one error bar per headline lead is drawn at
    ``(lambda*, mean)`` spanning the across-init ``[min, max]`` -- the real-only
    analogue of the softening figure's across-init band. Single-init runs (6ep)
    pass ``None`` so no bar is drawn.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Headline leads for the curve panels: longest (+48h) and shortest (+6h).
    by_step = {r["step"]: r for r in rows}
    steps_sorted = sorted(by_step)
    headline = [steps_sorted[-1]] + ([steps_sorted[0]] if len(steps_sorted) > 1 else [])

    title_fs = 16
    panel_title_fs = 12.5
    label_fs = 12
    tick_fs = 10
    legend_fs = 9
    marker_size = 7.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    styles = {headline[0]: ("tab:red", "-")}
    if len(headline) > 1:
        styles[headline[1]] = ("tab:blue", "--")
    for step in headline:
        pay = payloads[step]
        lams = pay["lambdas"]
        if lams.size == 0:
            continue
        c, ls = styles[step]
        lead = by_step[step]["lead_hours"]
        x = np.where(lams > 0, lams, lams[lams > 0].min() / 3 if np.any(lams > 0) else 0.1)
        ax1.plot(x, pay["sweep_cov90"], "o" + ls, color=c, markersize=marker_size,
                 label=f"Central 90% (+{lead:g} h)")
        ax1.plot(x, pay["sweep_cov50"], "s" + ls, color=c, alpha=0.6,
                 markersize=marker_size, label=f"Central 50% (+{lead:g} h)")
        ax2.plot(x, pay["sweep_nll_over_n"], "o" + ls, color=c,
                 markersize=marker_size, label=f"NLL/N (+{lead:g} h)")
        ls_star = by_step[step]["lambda_star"]
        if ls_star:
            ax1.axvline(ls_star, color=c, lw=0.8, ls=":")
            ax2.axvline(ls_star, color=c, lw=0.8, ls=":")
        # Across-init range (Option A): one error bar at (lambda*, mean cov90)
        # spanning the [min, max] over the 12 inits. Labelled once, on the
        # headline +48h lead (headline[0]); the +6h bar shares the colour code.
        if cov90_spread and ls_star and step in cov90_spread:
            mean, lo, hi = cov90_spread[step]
            if None not in (mean, lo, hi):
                ax1.errorbar(
                    [ls_star], [mean], yerr=[[mean - lo], [hi - mean]],
                    fmt="o", color=c, ms=7, mec="k", mew=0.7, capsize=4,
                    elinewidth=1.4, zorder=7,
                    label="Across-Init Range (12 Inits)" if step == headline[0] else None,
                )
    ax1.axhline(0.9, color="grey", lw=0.8, ls=":", label="90% Reference")
    ax1.axhline(0.5, color="grey", lw=0.8, ls=":")
    ax1.set_xscale("log")
    ax1.set_ylim(0.6, 1.03)
    ax1.set_xlabel("$\\lambda$ (log scale)", fontsize=label_fs)
    ax1.set_ylabel("Marginal-Position Coverage", fontsize=label_fs)
    ax1.set_title("Coverage vs $\\lambda$", fontsize=panel_title_fs)
    ax1.tick_params(labelsize=tick_fs)
    ax1.legend(fontsize=legend_fs)
    ax1.grid(alpha=0.3)
    ax2.set_xscale("log")
    ax2.set_xlabel("$\\lambda$ (log scale)", fontsize=label_fs)
    ax2.set_ylabel("NLL/N (nats)", fontsize=label_fs)
    ax2.set_title("Likelihood Cost vs $\\lambda$", fontsize=panel_title_fs)
    ax2.tick_params(labelsize=tick_fs)
    ax2.legend(fontsize=legend_fs)
    ax2.grid(alpha=0.3)
    display_label = label.replace("ep", "-Epoch").replace("6-Epoch", "6-Epoch")
    fig.suptitle(f"Forecast Marginal Faithfulness ({display_label})", fontsize=title_fs)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--out-csv", type=Path, default=OUT_DIR / "faithfulness_by_lead.csv")
    parser.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    parser.add_argument("--prefix", action="append", default=None,
                        help="restrict to one run prefix (repeatable); default: both")
    parser.add_argument("--n-members", type=int, default=N_MEMBERS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-fig", action="store_true")
    args = parser.parse_args()

    # (prefix, label, column, kind, make_fig). The 6ep run and the canonical 14ep
    # init A are the headline single-init rows (kind="single", figures drawn); the
    # remaining first-of-month replicates are extra inits (kind="init", no separate
    # figure) that only feed the across-init aggregate.
    plan = [(SIXEP_SPEC["prefix"], SIXEP_SPEC["label"], SIXEP_SPEC["column"], "single", True)]
    for i, prefix in enumerate(_discover_converged_prefixes(args.data_dir)):
        plan.append((prefix, CONVERGED_LABEL, CONVERGED_COLUMN,
                     "single" if i == 0 else "init", i == 0))
    if args.prefix is not None:
        plan = [p for p in plan if p[0] in args.prefix]

    all_rows = []
    converged_per_init = []
    pending_figs = []  # (label, column, fig_rows, payloads, fig_path); drawn post-agg
    for prefix, label, column, kind, make_fig in plan:
        rows, fig_rows, payloads = rows_for_prefix(
            prefix, label, column, kind,
            data_dir=args.data_dir, runs_dir=args.runs_dir,
            n_members=args.n_members, seed=args.seed,
        )
        if not rows:
            print(f"[faithfulness] no per-lead npz/run dirs for {prefix!r} in {args.data_dir}")
            continue
        all_rows.extend(rows)
        if column == CONVERGED_COLUMN:
            converged_per_init.append((prefix, rows))
        if make_fig and fig_rows and not args.no_fig:
            # Defer rendering: the converged figure's across-init band needs the
            # aggregate, which is only complete after every init is processed.
            fig_path = args.fig_dir / f"phase_4_forecast_faithfulness_{label}.png"
            pending_figs.append((label, column, fig_rows, payloads, fig_path))

    agg_rows = aggregate_converged_faith(converged_per_init)
    all_rows.extend(agg_rows)
    if agg_rows:
        print(f"[faithfulness] aggregated {len(converged_per_init)} converged inits "
              f"({', '.join(p for p, _ in converged_per_init)})")
    elif len(converged_per_init) == 1:
        print(f"[faithfulness] only one converged init ({converged_per_init[0][0]}); "
              "no across-init spread yet (rerun B/C to populate it)")

    # Across-init central-90% coverage range (Option A), step -> (mean, lo, hi),
    # from the kind='init_mean' rows -- only the Converged figure carries it; the
    # single-init 6ep run gets cov90_spread=None.
    cov90_spread = {
        int(r["step"]): (
            _num(r.get("m1_star_cov90")),
            _num(r.get("m1_star_cov90_lo")),
            _num(r.get("m1_star_cov90_hi")),
        )
        for r in agg_rows
    }
    for label, column, fig_rows, payloads, fig_path in pending_figs:
        spread = cov90_spread if column == CONVERGED_COLUMN else None
        make_figure(fig_rows, payloads, label, fig_path, cov90_spread=spread)
        print(f"[faithfulness] wrote {fig_path}")

    if not all_rows:
        raise SystemExit("no faithfulness rows produced; run conversion + the per-lead audit first")
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as fh:
        # restval="" leaves the spread columns blank on single/init rows (only
        # the kind='init_mean' aggregate rows carry _lo/_hi).
        writer = csv.DictWriter(fh, fieldnames=list(CSV_FIELDS), restval="")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[faithfulness] wrote {args.out_csv} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
