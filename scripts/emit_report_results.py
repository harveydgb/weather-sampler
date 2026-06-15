r"""Emit Ch 5 result macros + table fragments from persisted artifacts (report S7).

Drop-in discipline: "no Ch 5 number is typed by hand". This script reads ONLY
persisted artifacts (softening CSV, per-lead lambda_star.json, faithfulness CSV,
method4_sweep.npz, the AE run dir) and overwrites:

  * report/construction/macros-results.tex   (\maxPi*, \oneHot*, \secondModeMass*,
    \forecastLambdaStar*, \forecastBeta*, \deltaCrps*, \reconstruction*)
  * report/construction/tables/pi_softening.tex
  * report/construction/tables/lambda_calibration.tex
  * report/construction/tables/faithfulness.tex

Every cell that has no artifact stays an em-dash (\ResultPlaceholder), so the
document always compiles and can never carry a stale hand-typed number. The
core builders are pure functions over plain dicts/lists (unit-tested over
fixtures); `main()` only does the artifact I/O.

    .venv/bin/python scripts/emit_report_results.py
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from sampler_research import faithfulness as fth

REPO_ROOT = Path(__file__).resolve().parents[1]
SOFT_CSV = REPO_ROOT / "outputs" / "runs" / "phase_4_forecast_softening" / "softening_by_lead.csv"
FAITH_CSV = REPO_ROOT / "outputs" / "runs" / "phase_4_forecast_faithfulness" / "faithfulness_by_lead.csv"
RUNS_DIR = REPO_ROOT / "outputs" / "runs"
AE_RUN_DIR = REPO_ROOT / "outputs" / "runs" / "phase_4_real"
REPORT_DIR = REPO_ROOT / "report" / "construction"

PLACEHOLDER = r"\ResultPlaceholder"
# Single-scalar macros (lambda*, beta, dCRPS) quote the +48h headline lead (step 8).
HEADLINE_STEP = 8
HOUR_STEP = {"SixHour": 1, "FortyEight": 8}
COL_TO_LABEL = {"SixEp": "6ep", "Converged": "14ep"}
# Documented, ultra-review-verified AE lambda* fallback (log.md 2026-06-11) when
# the local AE run dir is absent; physical value, not a guess.
AE_LAMBDA_STAR_FALLBACK = 93.74


# --------------------------------------------------------------- formatters
def _fmt(value, spec, transform=lambda v: v):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return PLACEHOLDER
    return format(transform(value), spec)


def fmt_pi(v):
    return _fmt(v, ".3f")


def fmt_pct(v):
    return PLACEHOLDER if v is None else format(100.0 * v, ".1f") + r"\%"


def fmt_lambda(v):
    return _fmt(v, ".1f")


def fmt_beta(v):
    return _fmt(v, ".3g")


def fmt_delta(v):
    return _fmt(v, "+.3f")


def fmt_cov(v):
    return _fmt(v, ".3f")


def fmt_sci(v):
    """Two-sig-fig LaTeX sci-notation, math-mode content (no surrounding $)."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return PLACEHOLDER
    if v == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(v))))
    mant = v / (10.0 ** exp)
    return rf"{mant:.1f}\times10^{{{exp}}}"


# --------------------------------------------------------------- pure core
def build_macros(soft_rows, lambda_stars, faith_rows, m4_ops, ae):
    """Map persisted artifacts to the fixed macro set. Pure over plain inputs.

    `soft_rows`/`faith_rows`: lists of dicts (CSV rows, numeric fields parsed).
    `lambda_stars`/`m4_ops`: dicts keyed `(run_label, step)`.
    `ae`: `{"lambda_star": float|None, "beta": float|None}`.
    """

    # Headline point macros use the single-init rows only (6ep, and the canonical
    # 14ep init A). The across-init `kind='init_mean'` aggregate rows are surfaced
    # separately by build_softening/faithfulness_spread_macros, so the published
    # single-init headline numbers never move when replicate inits are added.
    soft = {(r["run"], int(r["step"])): r for r in soft_rows
            if r.get("kind", "single") == "single"}
    faith = {(r["run"], int(r["step"])): r for r in faith_rows
             if r.get("kind", "single") == "single"}
    macros = {}

    for hour, step in HOUR_STEP.items():
        for col, label in COL_TO_LABEL.items():
            row = soft.get((label, step))
            macros[f"maxPi{hour}{col}"] = fmt_pi(row["median_max_pi"]) if row else PLACEHOLDER
            macros[f"oneHot{hour}{col}"] = fmt_pct(row["one_hot_fraction"]) if row else PLACEHOLDER
            macros[f"secondModeMass{hour}{col}"] = (
                fmt_pi(row["median_second_mode"]) if row else PLACEHOLDER
            )
            # Well-separated (>2sigma) bimodal fraction: the "emerges with training"
            # result. Rare/flat at 6ep, grows with lead in the converged column.
            # `.get` (not `[]`) keeps the pure-fixture tests placeholder-safe.
            macros[f"bimodalTwoSigma{hour}{col}"] = (
                fmt_pct(row.get("bimodal_frac_2sigma")) if row else PLACEHOLDER
            )

    macros["reconstructionLambdaStar"] = fmt_lambda(ae.get("lambda_star"))
    macros["reconstructionBeta"] = fmt_beta(ae.get("beta"))
    for col, label in COL_TO_LABEL.items():
        ls = lambda_stars.get((label, HEADLINE_STEP))
        macros[f"forecastLambdaStar{col}"] = fmt_lambda(ls["lambda_star"]) if ls else PLACEHOLDER
        macros[f"forecastBeta{col}"] = fmt_beta(m4_ops.get((label, HEADLINE_STEP)))
        fr = faith.get((label, HEADLINE_STEP))
        macros[f"deltaCrps{col}"] = fmt_delta(fr["iid_delta_crps"]) if fr else PLACEHOLDER
    return macros


def render_macros(macros):
    lines = [
        "% AUTO-GENERATED by scripts/emit_report_results.py -- do not edit by hand.",
        "% Unfilled cells are em-dashes; rerun the emitter after the pipeline lands.",
        "",
        r"\newcommand{\ResultPlaceholder}{\textemdash}",
        "",
    ]
    for name, value in macros.items():
        lines.append(rf"\newcommand{{\{name}}}{{{value}}}")
    return "\n".join(lines) + "\n"


def _steps_union(*dicts_or_rows):
    steps = set()
    for item in dicts_or_rows:
        if isinstance(item, dict):
            steps.update(s for _, s in item)
        else:
            steps.update(int(r["step"]) for r in item)
    return sorted(steps)


def _lead_label(step, soft_rows):
    for r in soft_rows:
        if int(r["step"]) == step:
            return f"+{int(round(float(r['lead_hours'])))}h"
    return f"+{6 * step}h"


def build_pi_softening_table(soft_rows):
    # Single-init rows only: the table's conv. column stays the canonical init A
    # point estimate (the across-init range is surfaced via the spread macros and
    # prose, not by silently swapping the table to the 3-init mean).
    soft = {(r["run"], int(r["step"])): r for r in soft_rows
            if r.get("kind", "single") == "single"}
    steps = _steps_union(soft_rows)
    head = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{median max-$\pi$} & \multicolumn{2}{c}{one-hot frac.} "
        r"& \multicolumn{2}{c}{2nd-mode mass} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Lead & 6\,ep & conv. & 6\,ep & conv. & 6\,ep & conv. \\",
        r"\midrule",
    ]
    body = []
    for step in steps:
        s6, s14 = soft.get(("6ep", step)), soft.get(("14ep", step))
        cells = [
            fmt_pi(s6["median_max_pi"]) if s6 else PLACEHOLDER,
            fmt_pi(s14["median_max_pi"]) if s14 else PLACEHOLDER,
            fmt_pct(s6["one_hot_fraction"]) if s6 else PLACEHOLDER,
            fmt_pct(s14["one_hot_fraction"]) if s14 else PLACEHOLDER,
            fmt_pi(s6["median_second_mode"]) if s6 else PLACEHOLDER,
            fmt_pi(s14["median_second_mode"]) if s14 else PLACEHOLDER,
        ]
        body.append(f"{_lead_label(step, soft_rows)} & " + " & ".join(cells) + r" \\")
    tail = [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(head + body + tail) + "\n"


def build_lambda_table(lambda_stars, soft_rows):
    steps = _steps_union(lambda_stars)
    if not steps:
        steps = _steps_union(soft_rows)
    head = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{$\lambda^\star$} & \multicolumn{2}{c}{matched $\widetilde{R}$} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Lead & 6\,ep & conv. & 6\,ep & conv. \\",
        r"\midrule",
    ]
    body = []
    for step in steps:
        l6, l14 = lambda_stars.get(("6ep", step)), lambda_stars.get(("14ep", step))

        def cell_lam(rec):
            if not rec:
                return PLACEHOLDER
            s = fmt_lambda(rec["lambda_star"])
            if s == PLACEHOLDER or rec.get("bracketed", True):
                return s
            return s + r"$^{\dagger}$"

        def cell_rt(rec):
            return fmt_pi(rec["matched_r_tilde"]) if rec else PLACEHOLDER

        cells = [cell_lam(l6), cell_lam(l14), cell_rt(l6), cell_rt(l14)]
        body.append(f"{_lead_label(step, soft_rows)} & " + " & ".join(cells) + r" \\")
    tail = [
        r"\bottomrule",
        r"\end{tabular}",
        r"% $\dagger$: unbracketed (nearest-row $\lambda^\star$, sweep extended).",
        r"\par\smallskip",
        # C1: across-init range footnote, macro-driven (no hand-typed number) and
        # placeholder-safe -- em-dashes here if the init_mean aggregate is absent.
        r"{\footnotesize Converged \mbox{+48\,h}~$\lambda^\star$ spans "
        r"$[\forecastLambdaStarConvergedLo,\forecastLambdaStarConvergedHi]$ "
        r"(mean~\forecastLambdaStarConvergedMean) across the "
        r"\faithfulnessNInits\ autumn-2023 initialisations; the tabulated "
        r"conv.\ value is the canonical case.}",
    ]
    return "\n".join(head + body + tail) + "\n"


def build_faithfulness_table(faith_rows, soft_rows):
    # Single-init rows only (canonical init A); see build_pi_softening_table.
    faith = {(r["run"], int(r["step"])): r for r in faith_rows
             if r.get("kind", "single") == "single"}
    steps = _steps_union(faith_rows) or _steps_union(soft_rows)
    head = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{$\Delta$CRPS (iid)} & \multicolumn{2}{c}{M1 cov.\ 90\%} "
        r"& \multicolumn{2}{c}{M1 PIT KS} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Lead & 6\,ep & conv. & 6\,ep & conv. & 6\,ep & conv. \\",
        r"\midrule",
    ]
    body = []
    for step in steps:
        f6, f14 = faith.get(("6ep", step)), faith.get(("14ep", step))
        cells = [
            fmt_delta(f6["iid_delta_crps"]) if f6 else PLACEHOLDER,
            fmt_delta(f14["iid_delta_crps"]) if f14 else PLACEHOLDER,
            fmt_cov(f6["m1_star_cov90"]) if f6 else PLACEHOLDER,
            fmt_cov(f14["m1_star_cov90"]) if f14 else PLACEHOLDER,
            fmt_cov(f6["m1_star_pit_ks"]) if f6 else PLACEHOLDER,
            fmt_cov(f14["m1_star_pit_ks"]) if f14 else PLACEHOLDER,
        ]
        body.append(f"{_lead_label(step, soft_rows)} & " + " & ".join(cells) + r" \\")
    tail = [
        r"\bottomrule",
        r"\end{tabular}",
        r"% $\Delta$CRPS is the stochastic iid do-no-harm baseline only; M1 PIT/coverage "
        r"are marginal-position diagnostics (report non-claim \#3).",
    ]
    return "\n".join(head + body + tail) + "\n"


# --------------------------------------------------------------- artifact I/O
def _read_csv_numeric(path):
    if not Path(path).exists():
        return []
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            out = dict(r)
            for k, v in r.items():
                if k in ("run", "column", "run_id", "prefix"):
                    continue
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    out[k] = v
            rows.append(out)
    return rows


def _load_lambda_stars(runs_dir):
    out = {}
    for label, prefix in (("6ep", "phase_4_fc48_6ep"), ("14ep", "phase_4_fc48_14ep")):
        for path in Path(runs_dir).glob(f"{prefix}_step*/lambda_star.json"):
            try:
                step = int(path.parent.name.split("_step")[1])
            except (IndexError, ValueError):
                continue
            rec = json.loads(path.read_text())
            out[(label, step)] = {
                "lambda_star": rec.get("lambda_star"),
                "matched_r_tilde": rec.get("matched_r_tilde"),
                "bracketed": rec.get("bracketed", True),
            }
    return out


# ------------------------------------------- across-init M1 vs smoothed-MAP gap
# Headline comparison ("M1 beats smoothed-MAP at matched R-tilde") resampled over
# forecast INITIAL CONDITIONS rather than cells: the resampling unit an examiner
# expects on the method comparison. m1_star (M1 @ lambda*) and smoothed_map_n10
# sit at matched coherence by construction (lambda* is chosen to bracket the
# blur's R-tilde), so both per-lead gaps below are paired at matched R-tilde.
# Negative = M1 wins. Distinct from `_bootstrap_frac_ci`, which resamples CELLS
# within one field (spatial variability, NOT an across-init CI).
COMPARISON_METHOD = "m1_star"
COMPARISON_BLUR = "smoothed_map_n10"
COMPARISON_NLL_COL = "nll_over_n"               # lower is better
COMPARISON_SMEAR_COL = "dnll_frac_gt_0p125"     # frac dNLL/cell > 0.125, lower is better
COMPARISON_GAP_KEYS = ("gap_nll", "gap_smear")


def _scores_row(scores_path, name):
    """The scores.csv row dict named `name`, or None if file/row absent."""
    if not Path(scores_path).exists():
        return None
    with open(scores_path) as fh:
        for row in csv.DictReader(fh):
            if row.get("name") == name:
                return row
    return None


def comparison_gap_rows(runs_dir, prefix, *, method=COMPARISON_METHOD,
                        blur=COMPARISON_BLUR):
    """Per-lead M1-minus-smoothed-MAP paired gaps for one init prefix.

    Reads each `{prefix}_step{k}/scores.csv` and returns one dict per lead with
    `gap_nll` (NLL/N) and `gap_smear` (frac dNLL/cell > 0.125), both
    method-minus-blur (negative = M1 wins) at matched R-tilde. init_datetime is
    read from the run's lambda_star.json regime when present. A step missing
    either row is skipped, so a not-yet-scored init contributes nothing and the
    across-init aggregate still runs (single-init-safe, like the softening path).
    """

    rows = []
    paths = sorted(Path(runs_dir).glob(f"{prefix}_step*/scores.csv"),
                   key=lambda p: int(p.parent.name.split("_step")[1]))
    for path in paths:
        step = int(path.parent.name.split("_step")[1])
        m1 = _scores_row(path, method)
        bl = _scores_row(path, blur)
        if m1 is None or bl is None:
            continue
        init_dt = None
        ls_path = path.parent / "lambda_star.json"
        if ls_path.exists():
            init_dt = json.loads(ls_path.read_text()).get("regime", {}).get("init_datetime")
        rows.append({
            "prefix": prefix, "step": step,
            "lead_hours": float(m1["lead_hours"]),
            "init_datetime": init_dt,
            "gap_nll": float(m1[COMPARISON_NLL_COL]) - float(bl[COMPARISON_NLL_COL]),
            "gap_smear": float(m1[COMPARISON_SMEAR_COL]) - float(bl[COMPARISON_SMEAR_COL]),
            "m1_r_tilde": float(m1["r_tilde"]),
            "blur_r_tilde": float(bl["r_tilde"]),
        })
    return rows


def aggregate_comparison_gaps(per_init, *, label, column):
    """Across-init mean + min-max range of the M1-minus-blur gaps, per lead.

    `per_init`: list of (prefix, rows) from `comparison_gap_rows`. Mirrors
    `run_forecast_faithfulness.aggregate_converged_faith`: one `kind='init_mean'`
    row per lead step holding the across-init mean of each gap plus `{k}_lo` /
    `{k}_hi` min-max. The spread is a RANGE over distinct synoptic cases, NOT a
    sampling CI -- keep that wording until >= 8 inits make a percentile/t interval
    honest. Fewer than two inits with rows -> [].
    """

    present = [(prefix, rows) for prefix, rows in per_init if rows]
    if len(present) < 2:
        return []
    by_step = {}
    for _prefix, rows in present:
        for row in rows:
            by_step.setdefault(int(row["step"]), []).append(row)
    out = []
    for step in sorted(by_step):
        group = by_step[step]
        inits = sorted({str(r.get("init_datetime")) for r in group})
        agg = {"run": label, "column": column, "kind": "init_mean",
               "n_inits": len(group), "init_datetime": ";".join(inits),
               "step": step, "lead_hours": group[0]["lead_hours"]}
        for key in COMPARISON_GAP_KEYS:
            vals = [float(r[key]) for r in group if r.get(key) is not None]
            agg[key] = sum(vals) / len(vals)
            agg[f"{key}_lo"] = min(vals)
            agg[f"{key}_hi"] = max(vals)
        out.append(agg)
    return out


M4_MATCH_REL_TOL = 0.1  # mirrors run_phase4_real.py's R~-bracket monitor


def _m4_operating_beta(run_dir, rel_tol=M4_MATCH_REL_TOL):
    """Beta whose R-tilde matches M1@lambda*, or None if no beta reaches it.

    Returns None (-> em-dash macro) when the Method-4 sweep cannot reach the
    matched-coherence operating point, which is the honest outcome under the
    near-one-hot unary gaps: the sweep is *effectively pinned* (likelihood and
    field barely move with beta) and its closest R-tilde stays well above the
    smoothed-MAP / M1 target. A bare nearest-beta would otherwise report a grid
    endpoint as if it were a matched comparison (log.md 2026-06-11: "the
    matched-R-tilde M4 comparison is unavailable").
    """

    sweep_path = Path(run_dir) / "method4_sweep.npz"
    scores_path = Path(run_dir) / "scores.csv"
    if not sweep_path.exists() or not scores_path.exists():
        return None
    with np.load(sweep_path) as f:
        betas = np.asarray(f["betas"], dtype=float)
        fields = f["fields"]
        r_tilde = np.asarray(f["r_tilde"], dtype=float)
        nll = np.asarray(f["nll_over_n"], dtype=float)
    if len(betas) < 2 or np.ptp(fields, axis=0).max() <= 1e-9 or np.ptp(nll) <= 1e-9:
        return None
    star_rt = None
    with open(scores_path) as fh:
        for r in csv.DictReader(fh):
            if r["name"] == "m1_star":
                star_rt = float(r["r_tilde"])
                break
    if star_rt is None:
        return None
    nearest = int(np.argmin(np.abs(r_tilde - star_rt)))
    # No beta within rel_tol of the M1/target coherence -> no matched operating point.
    if abs(r_tilde[nearest] - star_rt) > rel_tol * abs(star_rt):
        return None
    return float(betas[nearest])


# ----------------------------------------------------- F1 spatial bootstrap CI
# Matched-R~ smear-fraction comparison: Method 1 @ lambda* vs the smoothed-MAP
# blur, both at frac(dNLL/cell > 0.125). The forecast (+48h) comparison is taken
# on the CONVERGED 14-epoch run (the chapter's final deliverable); the 6ep and
# step-0 (AE) runs are context columns only and never feed a forecast headline
# macro. The bootstrap resamples cells (spatial, within one field/init) to
# bracket the single-init point estimate -- it is NOT an across-init CI (that is
# the GPU-gated track-2 / down-scope; see report F1).
BOOT_METHOD = "m1_star"
BOOT_BLUR = "smoothed_map_n10"
BOOT_THRESHOLD = 0.125
BOOT_N_DRAWS = 2000
BOOT_SEED = 0
BOOT_CI = 0.95


def _bootstrap_frac_ci(run_dir, strata, *, threshold=BOOT_THRESHOLD,
                       n_boot=BOOT_N_DRAWS, seed=BOOT_SEED, ci=BOOT_CI,
                       method=BOOT_METHOD, blur=BOOT_BLUR):
    """Spatial-bootstrap CI on frac>threshold for method/blur and their paired
    gap, per stratum, from a run dir's persisted delta_per_cell.npz + masks.npz.

    Returns `{stratum: {"m1", "blur", "gap", "n"}}` (each a
    `bootstrap_cell_statistic` result) or `{}` if the per-cell deltas are
    absent. The three statistics share the bootstrap resamples (same seed), so
    `gap == m1 - blur` holds resample-by-resample.
    """

    run_dir = Path(run_dir)
    delta_path = run_dir / "delta_per_cell.npz"
    if not delta_path.exists():
        return {}
    with np.load(delta_path) as f:
        if method not in f.files or blur not in f.files:
            return {}
        dm = np.asarray(f[method], dtype=float)
        db = np.asarray(f[blur], dtype=float)
    masks = {}
    masks_path = run_dir / "masks.npz"
    if masks_path.exists():
        with np.load(masks_path) as f:
            masks = {k: np.asarray(f[k], dtype=bool) for k in f.files
                     if f[k].shape == dm.shape}

    def _stat(col):  # frac>threshold of one paired column (0=method, 1=blur)
        return lambda v: float(np.mean(v[:, col] > threshold))

    def _gap(v):
        return float(np.mean(v[:, 0] > threshold) - np.mean(v[:, 1] > threshold))

    out = {}
    for stratum in strata:
        if stratum == "global":
            sel = np.ones(dm.shape[0], dtype=bool)
        elif stratum in masks:
            sel = masks[stratum]
        else:
            continue
        if not sel.any():
            continue
        paired = np.column_stack([dm[sel], db[sel]])
        boot = lambda stat: fth.bootstrap_cell_statistic(
            paired, stat, n_boot=n_boot, ci=ci, rng=np.random.default_rng(seed))
        out[stratum] = {"m1": boot(_stat(0)), "blur": boot(_stat(1)),
                        "gap": boot(_gap), "n": int(sel.sum())}
    return out


def _ci_macros(base, ci_rec):
    if not ci_rec:
        return {base: PLACEHOLDER, f"{base}Lo": PLACEHOLDER, f"{base}Hi": PLACEHOLDER}
    return {base: fmt_pi(ci_rec["point"]),
            f"{base}Lo": fmt_pi(ci_rec["lo"]), f"{base}Hi": fmt_pi(ci_rec["hi"])}


def build_bootstrap_macros(recon, fc):
    """Pure: spatial-bootstrap CI records -> frac>0.125 CI macros (F1).

    `recon`/`fc` are `{stratum: {"m1","blur","gap","n"}}` dicts (possibly empty);
    inner values are `bootstrap_cell_statistic` results. Emits point + central-CI
    bounds for the matched-R~ smear fraction: global + bimodal for the AE recon
    regime, and the bimodal forecast (+48h) comparison F4 leads with. Empty
    records -> placeholders, so the document always compiles.
    """

    rg, rb = recon.get("global", {}), recon.get("bimodal", {})
    fb = fc.get("bimodal", {})
    macros = {}
    macros.update(_ci_macros("reconFracMethod", rg.get("m1")))
    macros.update(_ci_macros("reconFracBlur", rg.get("blur")))
    macros.update(_ci_macros("reconFracGap", rg.get("gap")))
    macros.update(_ci_macros("reconBimodalFracGap", rb.get("gap")))
    macros.update(_ci_macros("fcBimodalFracMethod", fb.get("m1")))
    macros.update(_ci_macros("fcBimodalFracBlur", fb.get("blur")))
    macros.update(_ci_macros("fcBimodalFracGap", fb.get("gap")))
    macros["bootstrapNDraws"] = str(BOOT_N_DRAWS)
    macros["bootstrapCIPct"] = format(100.0 * BOOT_CI, ".0f")
    return macros


# ------------------------------------------------ F1 track-2 across-init spread
# The headline point macros (built above) stay pinned to the canonical single
# init A. These builders add the across-init MEAN + min-max RANGE companions from
# the `kind='init_mean'` aggregate rows the runners now emit, so the report can
# state the softening trend's robustness over the three autumn-2023 synoptic
# cases without moving any single-init number. Range over a few cases, not a CI.

def _maybe_num(value):
    return None if value is None or value == "" else value


def _mean_lo_hi(macros, base, row, key, fmt):
    macros[f"{base}Mean"] = fmt(_maybe_num(row[key])) if row else PLACEHOLDER
    macros[f"{base}Lo"] = fmt(_maybe_num(row.get(f"{key}_lo"))) if row else PLACEHOLDER
    macros[f"{base}Hi"] = fmt(_maybe_num(row.get(f"{key}_hi"))) if row else PLACEHOLDER


def build_softening_spread_macros(soft_rows):
    """Pure: `kind='init_mean'` softening rows -> across-init Mean/Lo/Hi macros.

    For the two headline leads (+6h = step 1, +48h = step 8) of the Converged
    column, emits `{maxPi,oneHot,secondModeMass}{Hour}Converged{Mean,Lo,Hi}` plus
    `softeningNInits`. Reads only the aggregate rows; absent -> placeholders.
    """

    means = {int(r["step"]): r for r in soft_rows if r.get("kind") == "init_mean"}
    macros = {}
    n_inits = None
    for hour, step in HOUR_STEP.items():
        row = means.get(step)
        if row is not None and n_inits is None:
            n_inits = row.get("n_inits")
        _mean_lo_hi(macros, f"maxPi{hour}Converged", row, "median_max_pi", fmt_pi)
        _mean_lo_hi(macros, f"oneHot{hour}Converged", row, "one_hot_fraction", fmt_pct)
        _mean_lo_hi(macros, f"secondModeMass{hour}Converged", row, "median_second_mode", fmt_pi)
    macros["softeningNInits"] = str(int(n_inits)) if n_inits else PLACEHOLDER
    return macros


def build_faithfulness_spread_macros(faith_rows):
    """Pure: `kind='init_mean'` faithfulness rows -> across-init Mean/Lo/Hi macros.

    For the +48h headline lead of the Converged column, emits the across-init
    range of lambda*, do-no-harm delta CRPS, M1 central-90% coverage and M1
    marginal-position PIT-KS, plus `faithfulnessNInits`. Aggregate absent ->
    placeholders.
    """

    means = {int(r["step"]): r for r in faith_rows if r.get("kind") == "init_mean"}
    row = means.get(HEADLINE_STEP)
    macros = {}
    _mean_lo_hi(macros, "forecastLambdaStarConverged", row, "lambda_star", fmt_lambda)
    _mean_lo_hi(macros, "deltaCrpsConverged", row, "iid_delta_crps", fmt_delta)
    # Macro bases must be letters-only: LaTeX \newcommand names cannot contain
    # digits, so "m1Cov90"->"mOneCovNinety" and "m1PitKs"->"mOnePitKs".
    _mean_lo_hi(macros, "mOneCovNinetyConverged", row, "m1_star_cov90", fmt_cov)
    _mean_lo_hi(macros, "mOnePitKsConverged", row, "m1_star_pit_ks", fmt_cov)
    macros["faithfulnessNInits"] = (
        str(int(row["n_inits"])) if row and row.get("n_inits") else PLACEHOLDER
    )
    return macros


def build_crps_extremum_macros(faith_rows):
    """Pure: worst-case |iid_delta_crps| over CONVERGED single+init rows (C2).

    Scans every converged per-lead, per-init row (`run` == the Converged label,
    `kind` in {single, init}); the across-init aggregate (`init_mean`) and the
    6-epoch context column are excluded. Emits `\\deltaCrpsMaxAbs` in
    sci-notation so the do-no-harm bound can be stated numerically even though
    the per-lead \\DeltaCRPS\\ rounds to +/-0.000 at three decimals. Empty -> placeholder."""

    converged = COL_TO_LABEL["Converged"]
    vals = []
    for r in faith_rows:
        if r.get("run") != converged or r.get("kind", "single") not in ("single", "init"):
            continue
        v = _maybe_num(r.get("iid_delta_crps"))
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f):
            vals.append(abs(f))
    return {"deltaCrpsMaxAbs": fmt_sci(max(vals)) if vals else PLACEHOLDER}


# ------------------------------------------ across-init M1 vs smoothed-MAP gap
# Probe-#2 error bar: the headline "M1 beats smoothed-MAP at matched R-tilde"
# resampled over forecast INITS (not cells, like the bootstrap above). Init
# prefixes are discovered from the scored run dirs, so adding inits needs no edit
# here -- the canonical init A plus every `phase_4_fc48_v2_init*` replicate.
CONVERGED_HEADLINE_PREFIX = "phase_4_fc48_14ep"       # init A (canonical)
CONVERGED_REPLICATE_GLOB = "phase_4_fc48_v2_init*"    # replicate inits B, C, ...


def _discover_converged_prefixes(runs_dir):
    """Converged-column init prefixes that have scored run dirs, canonical-first."""
    runs_dir = Path(runs_dir)
    prefixes = []
    if any(runs_dir.glob(f"{CONVERGED_HEADLINE_PREFIX}_step*")):
        prefixes.append(CONVERGED_HEADLINE_PREFIX)
    seen = set()
    for path in sorted(runs_dir.glob(f"{CONVERGED_REPLICATE_GLOB}_step*")):
        prefix = path.name.rsplit("_step", 1)[0]
        if prefix not in seen:
            seen.add(prefix)
            prefixes.append(prefix)
    return prefixes


def build_comparison_gap_macros(gap_means):
    """Pure: `kind='init_mean'` gap rows -> across-init Mean/Lo/Hi gap macros.

    For the two headline leads (+6h = step 1, +48h = step 8) of the Converged
    column emits `gap{Nll,Smear}{Hour}Converged{Mean,Lo,Hi}` (signed; negative =
    M1 beats the smoothed-MAP blur at matched R-tilde) plus `comparisonNInits`.
    The spread is a RANGE over distinct synoptic inits, NOT a sampling CI -- keep
    that wording in prose until >= 8 inits. Aggregate absent -> placeholders.
    """

    means = {int(r["step"]): r for r in gap_means if r.get("kind") == "init_mean"}
    macros = {}
    n_inits = None
    for hour, step in HOUR_STEP.items():
        row = means.get(step)
        if row is not None and n_inits is None:
            n_inits = row.get("n_inits")
        _mean_lo_hi(macros, f"gapNll{hour}Converged", row, "gap_nll", fmt_delta)
        _mean_lo_hi(macros, f"gapSmear{hour}Converged", row, "gap_smear", fmt_delta)
    macros["comparisonNInits"] = str(int(n_inits)) if n_inits else PLACEHOLDER
    return macros


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--soft-csv", type=Path, default=SOFT_CSV)
    parser.add_argument("--faith-csv", type=Path, default=FAITH_CSV)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--ae-run-dir", type=Path, default=AE_RUN_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--allow-empty", action="store_true",
                        help="emit even when the forecast artifacts are absent "
                             "(writes em-dash placeholders; off by default so a "
                             "stray run cannot clobber the committed macros)")
    args = parser.parse_args()

    soft_rows = _read_csv_numeric(args.soft_csv)
    faith_rows = _read_csv_numeric(args.faith_csv)
    lambda_stars = _load_lambda_stars(args.runs_dir)

    # Clobber guard: the emitter's inputs (softening/faithfulness CSVs, per-lead
    # lambda_star.json) are git-ignored, so on a fresh clone they are absent. A
    # bare run would otherwise overwrite the committed macros/tables with
    # placeholders. Refuse unless --allow-empty is passed on purpose.
    if not (soft_rows or faith_rows or lambda_stars) and not args.allow_empty:
        raise SystemExit(
            "[emit] forecast artifacts absent (no softening/faithfulness rows, no "
            "per-lead lambda_star.json under --runs-dir); refusing to overwrite the "
            "committed macros/tables. Regenerate the artifacts (run the Phase 4 "
            "forecast pipeline) or pass --allow-empty to write placeholders on purpose."
        )

    ae = {"lambda_star": None, "beta": None}
    ae_star = Path(args.ae_run_dir) / "lambda_star.json"
    if ae_star.exists():
        ae["lambda_star"] = json.loads(ae_star.read_text()).get("lambda_star")
    if ae["lambda_star"] is None:
        ae["lambda_star"] = AE_LAMBDA_STAR_FALLBACK
        print(f"[emit] AE run dir absent; using documented lambda* fallback {AE_LAMBDA_STAR_FALLBACK}")
    ae["beta"] = _m4_operating_beta(args.ae_run_dir)

    m4_ops = {}
    for label, prefix in (("6ep", "phase_4_fc48_6ep"), ("14ep", "phase_4_fc48_14ep")):
        for step in {s for (lbl, s) in lambda_stars if lbl == label} | {HEADLINE_STEP}:
            m4_ops[(label, step)] = _m4_operating_beta(args.runs_dir / f"{prefix}_step{step}")

    macros = build_macros(soft_rows, lambda_stars, faith_rows, m4_ops, ae)
    # F1: spatial-bootstrap CI on the matched-R~ smear-fraction comparison. The
    # forecast (+48h) comparison is taken on the CONVERGED 14-epoch run -- the
    # chapter headline. The 6ep / step-0 (AE) runs are training-progression
    # context only and never feed a forecast headline macro.
    recon_boot = _bootstrap_frac_ci(args.ae_run_dir, ("global", "bimodal"))
    fc_boot = _bootstrap_frac_ci(args.runs_dir / "phase_4_fc48_14ep_step8",
                                 ("global", "bimodal"))
    macros = {**macros, **build_bootstrap_macros(recon_boot, fc_boot)}
    # F1 track-2: across-init mean + min-max range for the Converged column.
    macros = {**macros, **build_softening_spread_macros(soft_rows),
              **build_faithfulness_spread_macros(faith_rows)}
    # C2: worst-case do-no-harm |dCRPS| over converged single+init rows.
    macros = {**macros, **build_crps_extremum_macros(faith_rows)}
    # #4: across-init M1-minus-smoothed-MAP gap at matched R~ -- the error bar on
    # the headline comparison, resampled over forecast inits (not cells).
    gap_per_init = [(p, comparison_gap_rows(args.runs_dir, p))
                    for p in _discover_converged_prefixes(args.runs_dir)]
    gap_means = aggregate_comparison_gaps(
        gap_per_init, label=COL_TO_LABEL["Converged"], column="Converged")
    macros = {**macros, **build_comparison_gap_macros(gap_means)}
    tables = {
        "pi_softening.tex": build_pi_softening_table(soft_rows),
        "lambda_calibration.tex": build_lambda_table(lambda_stars, soft_rows),
        "faithfulness.tex": build_faithfulness_table(faith_rows, soft_rows),
    }

    (args.report_dir / "macros-results.tex").write_text(render_macros(macros))
    n_filled = sum(1 for v in macros.values() if v != PLACEHOLDER)
    print(f"[emit] wrote macros-results.tex ({n_filled}/{len(macros)} macros filled)")
    tables_dir = args.report_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, text in tables.items():
        (tables_dir / name).write_text(text)
        print(f"[emit] wrote tables/{name}")


if __name__ == "__main__":
    main()
