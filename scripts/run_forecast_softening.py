"""RQ1 headline: per-lead mixture-weight softening, both forecast runs (plan Step 2).

Loops over forecast leads x runs (6-epoch gmm_fc48_v1 | 14-epoch gmm_fc48_v2),
computes the pinned softening metrics (`sampler_research.forecast_diag`), writes
one tidy CSV (`softening_by_lead.csv`, a `run` column distinguishes the two), and
draws the S5.2 two-curve figure (median max-pi, one-hot fraction, 2nd-mode mass
vs lead, both runs, AE step-0 anchor marked). Reads only the converted per-lead
npz; no sampler run needed.

    .venv/bin/python scripts/run_forecast_softening.py

Verification gate (the log used the same one): every CSV cell reproduces
research_notes/log.md to displayed precision.
"""

import argparse
import csv
import json
from pathlib import Path

from sampler_research.forecast_diag import SOFTENING_METRIC_KEYS, softening_metrics
from sampler_research.io import load_real_marginal

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "outputs" / "data"
RUN_DIR = REPO_ROOT / "outputs" / "runs" / "phase_4_forecast_softening"
FIG_DIR = REPO_ROOT / "outputs" / "figures"

# prefix -> report column/run identity. SixEp = v1 me5 (6 ep, single init).
SIXEP_SPEC = {"prefix": "phase_4_fc48_6ep", "label": "6ep", "column": "SixEp",
              "run_id": "gmm_fc48_v1"}

# Converged = v2 me7 (14 ep). F1 track-2: the SAME trained model inferred at three
# autumn-2023 init dates -- distinct synoptic cases within the Oct-Dec 2023
# validation window, NOT seasons (report: seasonal/global generality stays a
# non-claim). The canonical init (A, first below) supplies the *unchanged*
# single-init headline rows; all three are aggregated into the across-init
# mean + range (`kind='init_mean'`).
CONVERGED_LABEL = "14ep"
CONVERGED_COLUMN = "Converged"
CONVERGED_RUN_ID = "gmm_fc48_v2"
CONVERGED_INIT_PREFIXES = (
    "phase_4_fc48_14ep",                # init A: 2023-11-01T00:00 (canonical/headline)
    "phase_4_fc48_v2_init20231010T12",  # init B: 2023-10-10T12:00
    "phase_4_fc48_v2_init20231215",     # init C: 2023-12-15T00:00
)
# Min-max range columns appended per metric for the across-init aggregate rows.
SPREAD_SUFFIXES = ("_lo", "_hi")

# AE step-0 reconstruction anchor (gmm_era5_32ep_v3 me31; log.md): median max-pi.
AE_ANCHOR_MAX_PI = 0.965


def _lead_files(data_dir, prefix):
    found = []
    for path in sorted(Path(data_dir).glob(f"{prefix}_step*_2t.npz")):
        stem = path.stem  # phase_4_fc48_6ep_step3_2t
        try:
            step = int(stem.split("_step")[1].split("_2t")[0])
        except (IndexError, ValueError):
            continue
        found.append((step, path))
    return sorted(found, key=lambda item: item[0])


def softening_rows(data_dir, spec, kind="single"):
    """Per-lead metric rows for one run spec (empty if its npz are absent).

    `kind` labels the row's role: ``"single"`` for the headline single-init rows
    (6ep, and the canonical 14ep init A) and ``"init"`` for the extra converged
    replicates (inits B/C) that feed the across-init aggregate but are kept out
    of the headline point-macro keying downstream.
    """

    rows = []
    for step, path in _lead_files(data_dir, spec["prefix"]):
        data = load_real_marginal(path)
        meta_path = path.with_name(path.stem + "_meta.json")
        lead = init_dt = None
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            lead = meta.get("lead_hours")
            init_dt = meta.get("init_datetime")
        row = {"run": spec["label"], "column": spec["column"], "run_id": spec["run_id"],
               "prefix": spec["prefix"], "kind": kind, "n_inits": 1,
               "init_datetime": init_dt, "step": step,
               "lead_hours": lead if lead is not None else 6.0 * step}
        row.update(softening_metrics(data["pi"], data["mu"], data["sigma"]))
        rows.append(row)
    return rows


def converged_init_rows(data_dir, prefixes=CONVERGED_INIT_PREFIXES):
    """Per-init softening rows for every converged init prefix that has npz.

    Returns a list of (prefix, rows) for the inits present, canonical-first. The
    canonical init (A) is tagged ``kind='single'`` (the unchanged headline row);
    the others ``kind='init'``. Missing inits are skipped, so the script still
    runs (single-init behaviour) before the B/C reruns land.
    """

    found = []
    for i, prefix in enumerate(prefixes):
        spec = {"prefix": prefix, "label": CONVERGED_LABEL,
                "column": CONVERGED_COLUMN, "run_id": CONVERGED_RUN_ID}
        rows = softening_rows(data_dir, spec, kind="single" if i == 0 else "init")
        if rows:
            found.append((prefix, rows))
    return found


def aggregate_converged_rows(per_init):
    """Across-init MEAN + min-max range per lead, as `kind='init_mean'` rows.

    `per_init`: list of (prefix, rows) from `converged_init_rows`. For each lead
    step present, the base metric column holds the mean over the inits and
    `{metric}_lo`/`{metric}_hi` hold the min/max. The spread is a RANGE over a
    few distinct synoptic cases, not a sampling CI. Fewer than two inits -> [].
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
               "run_id": CONVERGED_RUN_ID,
               "prefix": f"{CONVERGED_INIT_PREFIXES[0]}+{len(per_init) - 1}",
               "kind": "init_mean", "n_inits": len(group),
               "init_datetime": ";".join(inits), "step": step,
               "lead_hours": group[0]["lead_hours"]}
        for key in SOFTENING_METRIC_KEYS:
            vals = [float(r[key]) for r in group]
            agg[key] = sum(vals) / len(vals)
            agg[f"{key}_lo"] = min(vals)
            agg[f"{key}_hi"] = max(vals)
        out.append(agg)
    return out


def write_csv(rows, out_csv):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        ["run", "column", "run_id", "prefix", "kind", "n_inits", "init_datetime",
         "step", "lead_hours", *SOFTENING_METRIC_KEYS]
        + [f"{k}{s}" for k in SOFTENING_METRIC_KEYS for s in SPREAD_SUFFIXES]
    )
    # restval="" leaves the spread columns blank on single/init rows (only the
    # `kind='init_mean'` aggregate rows carry _lo/_hi).
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return out_csv


def make_figure(rows, fig_path, ae_anchor=AE_ANCHOR_MAX_PI):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    colours = {"6ep": "tab:blue", "14ep": "tab:red"}
    # Headline curves use the single-init rows only (6ep, and the canonical
    # 14ep init A); the across-init range is shaded as a band where available.
    singles = [r for r in rows if r.get("kind", "single") == "single"]
    means = {int(r["step"]): r for r in rows if r.get("kind") == "init_mean"}
    for label in ("6ep", "14ep"):
        sub = sorted((r for r in singles if r["run"] == label), key=lambda r: r["lead_hours"])
        if not sub:
            continue
        if label == "14ep" and means:
            leads_b = [r["lead_hours"] for r in sub]
            lo = [means[int(r["step"])]["median_max_pi_lo"] for r in sub if int(r["step"]) in means]
            hi = [means[int(r["step"])]["median_max_pi_hi"] for r in sub if int(r["step"]) in means]
            if len(lo) == len(leads_b):
                ax1.fill_between(leads_b, lo, hi, color="tab:red", alpha=0.15,
                                 label="median max-$\\pi$ across-init range")
        leads = [r["lead_hours"] for r in sub]
        c = colours[label]
        ax1.plot(leads, [r["median_max_pi"] for r in sub], "o-", color=c,
                 label=f"median max-$\\pi$ ({label})")
        ax1.plot(leads, [r["median_second_mode"] for r in sub], "^--", color=c, alpha=0.7,
                 label=f"2nd-mode mass ({label})")
        ax2.plot(leads, [100 * r["one_hot_fraction"] for r in sub], "s-", color=c,
                 label=f"one-hot fraction ({label})")
    # AE step-0 reconstruction anchor (left edge of the continuous curve).
    ax1.scatter([0.0], [ae_anchor], marker="*", s=160, color="k", zorder=6,
                label=f"AE step-0 anchor ({ae_anchor:.3f})")
    ax1.set_xlabel("forecast lead time (h)")
    ax1.set_ylabel("mixture weight")
    ax1.set_title("Weights soften monotonically with lead time")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)
    ax2.set_xlabel("forecast lead time (h)")
    ax2.set_ylabel("one-hot fraction (max-$\\pi$ > 0.9), %")
    ax2.set_title("Near-deterministic cells vanish with lead and training")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)
    fig.suptitle("Forecast-regime softening: 6-epoch (v1) vs 14-epoch converged (v2)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _print_table(rows):
    hdr = ["run", "step", "lead", *SOFTENING_METRIC_KEYS]
    print("  ".join(f"{h:>18s}" if i else f"{h:>6s}" for i, h in enumerate(hdr)))
    for r in rows:
        cells = [f"{r['run']:>6s}", f"{r['step']:>18d}", f"{r['lead_hours']:>18g}"]
        cells += [f"{r[k]:>18.4f}" for k in SOFTENING_METRIC_KEYS]
        print("  ".join(cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-csv", type=Path, default=RUN_DIR / "softening_by_lead.csv")
    parser.add_argument("--fig", type=Path, default=FIG_DIR / "phase_4_forecast_softening.png")
    parser.add_argument("--no-fig", action="store_true", help="skip the figure (CSV only)")
    args = parser.parse_args()

    rows = []
    six_rows = softening_rows(args.data_dir, SIXEP_SPEC)
    if not six_rows:
        print(f"[softening] no per-lead npz for prefix {SIXEP_SPEC['prefix']!r} in {args.data_dir}")
    rows.extend(six_rows)

    per_init = converged_init_rows(args.data_dir)
    for prefix, init_rows in per_init:
        rows.extend(init_rows)
    if not per_init:
        print(f"[softening] no per-lead npz for any converged init in {args.data_dir}")
    agg_rows = aggregate_converged_rows(per_init)
    rows.extend(agg_rows)
    if agg_rows:
        print(f"[softening] aggregated {len(per_init)} converged inits "
              f"({', '.join(p for p, _ in per_init)})")
    elif len(per_init) == 1:
        print(f"[softening] only one converged init ({per_init[0][0]}); "
              "no across-init spread yet (rerun B/C to populate it)")

    if not rows:
        raise SystemExit("no per-lead npz found for any run; convert the forecast .pt first")

    out_csv = write_csv(rows, args.out_csv)
    print(f"[softening] wrote {out_csv} ({len(rows)} rows)")
    _print_table(rows)
    if not args.no_fig:
        make_figure(rows, args.fig)
        print(f"[softening] wrote {args.fig}")


if __name__ == "__main__":
    main()
