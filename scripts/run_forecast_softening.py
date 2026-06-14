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

# prefix -> report column/run identity. SixEp = v1 me5 (6 ep); Converged = v2 me7 (14 ep).
RUN_SPECS = (
    {"prefix": "phase_4_fc48_6ep", "label": "6ep", "column": "SixEp", "run_id": "gmm_fc48_v1"},
    {"prefix": "phase_4_fc48_14ep", "label": "14ep", "column": "Converged", "run_id": "gmm_fc48_v2"},
)
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


def softening_rows(data_dir, spec):
    """Per-lead metric rows for one run spec (empty if its npz are absent)."""

    rows = []
    for step, path in _lead_files(data_dir, spec["prefix"]):
        data = load_real_marginal(path)
        meta_path = path.with_name(path.stem + "_meta.json")
        lead = None
        if meta_path.exists():
            lead = json.loads(meta_path.read_text()).get("lead_hours")
        row = {"run": spec["label"], "column": spec["column"], "run_id": spec["run_id"],
               "prefix": spec["prefix"], "step": step,
               "lead_hours": lead if lead is not None else 6.0 * step}
        row.update(softening_metrics(data["pi"], data["mu"], data["sigma"]))
        rows.append(row)
    return rows


def write_csv(rows, out_csv):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run", "column", "run_id", "prefix", "step", "lead_hours", *SOFTENING_METRIC_KEYS]
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_csv


def make_figure(rows, fig_path, ae_anchor=AE_ANCHOR_MAX_PI):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    colours = {"6ep": "tab:blue", "14ep": "tab:red"}
    for label in ("6ep", "14ep"):
        sub = sorted((r for r in rows if r["run"] == label), key=lambda r: r["lead_hours"])
        if not sub:
            continue
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
    for spec in RUN_SPECS:
        spec_rows = softening_rows(args.data_dir, spec)
        if not spec_rows:
            print(f"[softening] no per-lead npz for prefix {spec['prefix']!r} in {args.data_dir}")
        rows.extend(spec_rows)
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
