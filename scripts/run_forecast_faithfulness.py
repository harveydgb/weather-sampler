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

RUN_SPECS = (
    {"prefix": "phase_4_fc48_6ep", "label": "6ep", "column": "SixEp"},
    {"prefix": "phase_4_fc48_14ep", "label": "14ep", "column": "Converged"},
)
N_MEMBERS = 50
CSV_FIELDS = (
    "run", "column", "prefix", "step", "lead_hours", "lambda_star",
    "iid_delta_crps", "iid_ensemble_crps", "iid_analytic_crps", "iid_pit_ks", "iid_n_members",
    "m1_star_pit_ks", "m1_star_cov50", "m1_star_cov90", "m1_star_mean_abs_pit_centre",
    "m1_star_nll_over_n",
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


def make_figure(rows, payloads, label, fig_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Headline leads for the curve panels: longest (+48h) and shortest (+6h).
    by_step = {r["step"]: r for r in rows}
    steps_sorted = sorted(by_step)
    headline = [steps_sorted[-1]] + ([steps_sorted[0]] if len(steps_sorted) > 1 else [])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
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
        ax1.plot(x, pay["sweep_cov90"], "o" + ls, color=c, label=f"central-90% (+{lead:g}h)")
        ax1.plot(x, pay["sweep_cov50"], "s" + ls, color=c, alpha=0.6, label=f"central-50% (+{lead:g}h)")
        ax2.plot(x, pay["sweep_nll_over_n"], "o" + ls, color=c, label=f"NLL/N (+{lead:g}h)")
        ls_star = by_step[step]["lambda_star"]
        if ls_star:
            ax1.axvline(ls_star, color=c, lw=0.8, ls=":")
            ax2.axvline(ls_star, color=c, lw=0.8, ls=":")
    ax1.axhline(0.9, color="grey", lw=0.8, ls=":", label="iid do-no-harm (0.90)")
    ax1.axhline(0.5, color="grey", lw=0.8, ls=":")
    ax1.set_xscale("log")
    ax1.set_xlabel("$\\lambda$ (log axis; dotted = $\\lambda^\\star$)")
    ax1.set_ylabel("Method 1 marginal-position coverage")
    ax1.set_title("Faithfulness vs regularisation strength\n(marginal position, not calibration)")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)
    ax2.set_xscale("log")
    ax2.set_xlabel("$\\lambda$ (log axis)")
    ax2.set_ylabel("NLL/N under the emitted GMM (nats)")
    ax2.set_title("Likelihood cost of coherence along the sweep")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)
    fig.suptitle(f"Forecast regime ({label}): marginal faithfulness & do-no-harm", fontsize=11)
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

    specs = [s for s in RUN_SPECS if args.prefix is None or s["prefix"] in args.prefix]
    all_rows = []
    for spec in specs:
        leads = _lead_files(args.data_dir, spec["prefix"])
        if not leads:
            print(f"[faithfulness] no per-lead npz for {spec['prefix']!r} in {args.data_dir}")
            continue
        rows_for_fig, payloads_for_fig = [], {}
        for step, npz in leads:
            run_dir = args.runs_dir / f"{spec['prefix']}_step{step}"
            if not run_dir.exists():
                print(f"[faithfulness] missing run dir {run_dir}; run the per-lead audit first")
                continue
            meta = json.loads(npz.with_name(npz.stem + "_meta.json").read_text())
            row, payload = faithfulness_for_lead(
                npz, run_dir, n_members=args.n_members, seed=args.seed + step
            )
            row = {"run": spec["label"], "column": spec["column"], "prefix": spec["prefix"],
                   "step": step, "lead_hours": meta.get("lead_hours", 6.0 * step), **row}
            np.savez(run_dir / "faithfulness.npz", **payload)
            all_rows.append(row)
            rows_for_fig.append(row)
            payloads_for_fig[step] = payload
            print(f"[faithfulness] {spec['label']} +{row['lead_hours']:g}h: "
                  f"iid dCRPS={row['iid_delta_crps']:+.4f} iid PIT-KS={row['iid_pit_ks']:.4f} "
                  f"M1* cov90={row['m1_star_cov90']:.3f} PIT-KS={row['m1_star_pit_ks']:.3f}")
        if rows_for_fig and not args.no_fig:
            fig_path = args.fig_dir / f"phase_4_forecast_faithfulness_{spec['label']}.png"
            make_figure(rows_for_fig, payloads_for_fig, spec["label"], fig_path)
            print(f"[faithfulness] wrote {fig_path}")

    if not all_rows:
        raise SystemExit("no faithfulness rows produced; run conversion + the per-lead audit first")
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[faithfulness] wrote {args.out_csv} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
