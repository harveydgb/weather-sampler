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
from sampler_research import method4_mrf as rm4
from sampler_research import phase4_eval
from sampler_research.io import load_sampler_arrays

REPO_ROOT = Path(__file__).resolve().parents[1]
SOFT_CSV = REPO_ROOT / "outputs" / "runs" / "phase_4_forecast_softening" / "softening_by_lead.csv"
FAITH_CSV = REPO_ROOT / "outputs" / "runs" / "phase_4_forecast_faithfulness" / "faithfulness_by_lead.csv"
RUNS_DIR = REPO_ROOT / "outputs" / "runs"
AE_RUN_DIR = REPO_ROOT / "outputs" / "runs" / "phase_4_real"
TOY_SCORES_CSV = REPO_ROOT / "outputs" / "runs" / "stage_a_baselines" / "stage_a_scores.csv"
REPORT_DIR = REPO_ROOT / "report" / "construction"

PLACEHOLDER = r"\ResultPlaceholder"
# Single-scalar macros (lambda*, beta, dCRPS) quote the +48h headline lead (step 8).
HEADLINE_STEP = 8
HOUR_STEP = {"SixHour": 1, "FortyEight": 8}
COL_TO_LABEL = {"SixEp": "6ep", "Converged": "14ep"}
# Chapter-4 synthetic-testbed bracket anchors (I2): the Stage-A baseline rows
# (outputs/runs/stage_a_baselines/stage_a_scores.csv) quoted as toy macros + a toy
# table, so Chapter 4 cites its numbers via macros, never hand-typed. Each tuple is
# (csv `baseline` key, macro infix, table display name). LaTeX \newcommand names
# cannot carry digits/underscores, so the infix is letters-only.
TOY_BASELINES = [
    ("iid", "Iid", r"Independent draw (iid)"),
    ("mode_map", "ModeMap", r"Per-cell MAP (mode)"),
    ("mixture_mean", "MixtureMean", r"Mixture mean"),
    ("smoothed_map", "SmoothedMap", r"Smoothed MAP"),
    ("a_star", "AStar", r"Smoothest faithful ($a^\star$)"),
]
# Documented, ultra-review-verified AE lambda* fallback (log.md 2026-06-11) when
# the local AE run dir is absent; physical value, not a guess.
AE_LAMBDA_STAR_FALLBACK = 93.74

# Chapter-4 Section 4.2 operating-point off-mode smear macros (B5, 30 Jun). The
# off-mode fraction (cells with dNLL/cell > 0.125 nats, i.e. drifted > 0.5 sigma
# off the nearest emitted mode) at each method's figure operating point, recomputed
# from the committed toy fields the SAME way make_toy_figures.py does (against the
# shared Stage-C extracted modes), so the quoted number and the non-smearing figure
# can never diverge. The headline ~14% Joint MAP point is NOT in any CSV (the figure
# computes it inline), which is why it is recomputed here rather than read. Operating
# points are the ones the figures/notes quote:
#   Joint MAP  lambda=0.2  -> R~ ~ 0.25 (the quadratic's R~~0.25 point: the smallest
#              lambda reaching that roughness, and the lowest-smear of the cluster)
#   exact TV   lambda=0.1  -> R~ ~ 0.58 non-smearing knee (before plateau-collapse)
#   Mode-MRF   beta=0.075  -> R~ ~ 0.60 non-smearing knee
TOY_DATA_DIR = REPO_ROOT / "outputs" / "data"
TOY_QUAD_SMEAR_LAMBDA = 0.2
TOY_TV_KNEE_LAMBDA = 0.1
TOY_MRF_KNEE_BETA = 0.075


# --------------------------------------------------------------- formatters
def _fmt(value, spec, transform=lambda v: v):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return PLACEHOLDER
    return format(transform(value), spec)


def fmt_pi(v):
    return _fmt(v, ".3f")


def fmt_pct(v):
    return PLACEHOLDER if v is None else format(100.0 * v, ".1f") + r"\%"


def fmt_pp(v):
    """Signed percentage-point formatter for DIFFERENCES between two cell shares
    (e.g. JointMAP's off-mode fraction minus the blur's). A share of cells is a
    fmt_pct level; the gap between two such shares is a percentage-point
    difference, not a percentage of anything, so it gets its own unit (harvey,
    4 Jul: propagate the pp/% distinction across the reconFrac/fcBimodalFrac
    macro family)."""
    return PLACEHOLDER if v is None else format(100.0 * v, "+.1f") + r"\,pp"


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
    # Trimmed to the two cited headline leads (+6h, +48h); the figure carries the
    # full lead curve, and the prose quotes only these endpoints via macros.
    steps = [s for s in _steps_union(soft_rows) if s in (HOUR_STEP["SixHour"], HOUR_STEP["FortyEight"])]
    head = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{median max-$\pi$} & \multicolumn{2}{c}{one-hot frac.} "
        r"& \multicolumn{2}{c}{2nd-mode mass} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Hours ahead & 6\,ep & 14\,ep & 6\,ep & 14\,ep & 6\,ep & 14\,ep \\",
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
    # matched R-tilde was a constant column (held fixed by construction). Dropped:
    # the roughness-matching target, the bracketed-flag audit and the across-init
    # range now live in the table CAPTION (thesis.tex), not in a separate note block
    # under the table -- a table carries a caption only, no third text section.
    head = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{$\lambda^\star$} \\",
        r"\cmidrule(lr){2-3}",
        r"Hours ahead & 6\,ep & 14\,ep \\",
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

        cells = [cell_lam(l6), cell_lam(l14)]
        body.append(f"{_lead_label(step, soft_rows)} & " + " & ".join(cells) + r" \\")
    tail = [
        r"\bottomrule",
        r"\end{tabular}",
        r"% $\dagger$: unbracketed (nearest-row $\lambda^\star$, sweep extended);"
        r" the matching target, bracketed-flag audit and across-init range live in"
        r" the table caption (thesis.tex), not in a note under the table.",
    ]
    return "\n".join(head + body + tail) + "\n"


def build_faithfulness_table(faith_rows, soft_rows):
    # Single-init rows only (canonical init A); see build_pi_softening_table.
    faith = {(r["run"], int(r["step"])): r for r in faith_rows
             if r.get("kind", "single") == "single"}
    steps = _steps_union(faith_rows) or _steps_union(soft_rows)
    # DEC-R28 (3 Jul): CRPS and the PIT-KS uniformity statistic are out of the main
    # report; only the central marginal-position coverage remains (the Delta-CRPS
    # column had already been dropped, 29 Jun table review). Headers keep the
    # "marg.-pos." qualifier so the table cannot be misread as ensemble
    # calibration (non-claim #3). PIT-KS stays in the CSV/macros, just untabulated.
    head = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{marg.-pos.\ 90\%} \\",
        r"\cmidrule(lr){2-3}",
        r"Hours ahead & 6\,ep & 14\,ep \\",
        r"\midrule",
    ]
    body = []
    for step in steps:
        f6, f14 = faith.get(("6ep", step)), faith.get(("14ep", step))
        cells = [
            fmt_cov(f6["m1_star_cov90"]) if f6 else PLACEHOLDER,
            fmt_cov(f14["m1_star_cov90"]) if f14 else PLACEHOLDER,
        ]
        body.append(f"{_lead_label(step, soft_rows)} & " + " & ".join(cells) + r" \\")
    tail = [
        r"\bottomrule",
        r"\end{tabular}",
        r"% Joint MAP central marginal-position coverage only (report non-claim \#3); "
        r"CRPS and PIT-KS were removed from the report (DEC-R28).",
    ]
    return "\n".join(head + body + tail) + "\n"


# ------------------------------------------- Ch 4 synthetic-testbed brackets (I2)
# Toy bracket macros + table from the Stage-A baseline scores, so Chapter 4 quotes
# its numbers via macros (never hand-typed) exactly like the forecast chapter. Pure
# over the CSV rows; a baseline absent from the rows -> em-dash placeholder.
def build_toy_baseline_macros(toy_rows):
    """Pure: Stage-A baseline rows -> toy NLL/N + R-tilde macros.

    For each anchor in TOY_BASELINES emits `toy{Infix}Nll` and `toy{Infix}Rtilde`
    (3 d.p.). Reads the `baseline`, `nll_over_n` and `r_tilde` columns only; a
    baseline missing from `toy_rows` (or a missing value) -> placeholder.
    """

    by_name = {r.get("baseline"): r for r in toy_rows}
    macros = {}
    for key, infix, _label in TOY_BASELINES:
        row = by_name.get(key)
        macros[f"toy{infix}Nll"] = fmt_pi(row["nll_over_n"]) if row else PLACEHOLDER
        macros[f"toy{infix}Rtilde"] = fmt_pi(row["r_tilde"]) if row else PLACEHOLDER
    return macros


# Section 4.2 operating-point smear macros (B5). The off-mode fraction the figure
# computes inline is not persisted in a CSV, so it is recomputed here against the
# shared Stage-C modes -- identical to make_toy_figures.smear() -- guaranteeing the
# quoted number matches the non-smearing figure.
TOY_SMEAR_MACROS = (
    "toyQuadSmearFrac", "toyQuadSmearRtilde",
    "toyTvExactSmearFrac", "toyTvExactSmearRtilde",
    "toyMrfSmearFrac", "toyMrfSmearRtilde",
)


def _toy_frac_over(field, gmm, modes, valid_mask):
    """frac of cells drifted > 0.5 sigma (dNLL/cell > 0.125 nats) off the nearest
    emitted mode: the figure's `smear(...)["frac_over"][0]`, recomputed verbatim."""
    res = rm4.delta_nll_to_best_mode(
        field, gmm["pi"], gmm["mu"], gmm["sigma"], modes, valid_mask)
    return float(np.ravel(res["frac_over"])[0])


def _pick_sweep_index(values, target, atol=1e-6):
    """Index of the single sweep entry equal to `target` (operating-point select);
    None if absent, so a renamed/rescanned sweep degrades to a placeholder."""
    idx = np.flatnonzero(np.isclose(np.asarray(values, dtype=float), target, atol=atol))
    return int(idx[0]) if idx.size else None


def build_toy_smear_macros(runs_dir, data_dir, dataset="phase_1_homoscedastic"):
    """Recompute the Section 4.2 off-mode smear fraction (and matched R-tilde) at
    each method's figure operating point from the committed toy fields. Any absent
    artifact or missing operating point -> em-dash placeholders (I2)."""

    blank = {k: PLACEHOLDER for k in TOY_SMEAR_MACROS}
    runs_dir, data_dir = Path(runs_dir), Path(data_dir)
    try:
        d = load_sampler_arrays(data_dir / f"{dataset}.npz")
        B = dict(np.load(runs_dir / "stage_b_regularised_map" / f"{dataset}_regularised_map.npz"))
        C = dict(np.load(runs_dir / "stage_c_method4_mrf" / f"{dataset}_method4_mrf.npz"))
        T = dict(np.load(runs_dir / "stage_b_tv_ablation" / f"{dataset}_tv_ablation.npz"))
    except (FileNotFoundError, OSError):
        return blank

    gmm = {"pi": d["pi"], "mu": d["mu"], "sigma": d["sigma"]}
    modes, valid = C["mode_values"], C["valid_mask"]
    iq = _pick_sweep_index(B["lambdas"], TOY_QUAD_SMEAR_LAMBDA)
    it = _pick_sweep_index(T["cut_tv_lambdas"], TOY_TV_KNEE_LAMBDA)
    im = _pick_sweep_index(C["betas"], TOY_MRF_KNEE_BETA)
    if None in (iq, it, im):
        return blank
    return {
        "toyQuadSmearFrac": fmt_pct(_toy_frac_over(B["fields"][iq], gmm, modes, valid)),
        "toyQuadSmearRtilde": fmt_pi(float(B["r_tilde"][iq])),
        "toyTvExactSmearFrac": fmt_pct(_toy_frac_over(T["cut_tv_fields"][it], gmm, modes, valid)),
        "toyTvExactSmearRtilde": fmt_pi(float(T["cut_tv_r_tilde"][it])),
        "toyMrfSmearFrac": fmt_pct(_toy_frac_over(C["fields"][im], gmm, modes, valid)),
        "toyMrfSmearRtilde": fmt_pi(float(C["r_tilde"][im])),
    }


def build_toy_baseline_table(toy_rows):
    """Pure: Stage-A baseline rows -> the toy bracket table fragment.

    One row per TOY_BASELINES anchor (display name, NLL/N, R-tilde); a missing
    anchor renders as em-dashes so the table always compiles.
    """

    by_name = {r.get("baseline"): r for r in toy_rows}
    head = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Reference field & NLL/$N$ & $\widetilde{R}$ \\",
        r"\midrule",
    ]
    body = []
    for key, _infix, label in TOY_BASELINES:
        row = by_name.get(key)
        nll = fmt_pi(row["nll_over_n"]) if row else PLACEHOLDER
        rt = fmt_pi(row["r_tilde"]) if row else PLACEHOLDER
        body.append(f"{label} & {nll} & {rt} " + r"\\")
    tail = [r"\bottomrule", r"\end{tabular}"]
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


def _ci_macros(base, ci_rec, fmt=fmt_pi):
    if not ci_rec:
        return {base: PLACEHOLDER, f"{base}Lo": PLACEHOLDER, f"{base}Hi": PLACEHOLDER}
    return {base: fmt(ci_rec["point"]),
            f"{base}Lo": fmt(ci_rec["lo"]), f"{base}Hi": fmt(ci_rec["hi"])}


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
    # Levels (shares of cells) render as %; gaps (differences between two shares)
    # render as pp -- harvey, 4 Jul: propagate the %/pp distinction report-wide.
    macros.update(_ci_macros("reconFracMethod", rg.get("m1"), fmt=fmt_pct))
    macros.update(_ci_macros("reconFracBlur", rg.get("blur"), fmt=fmt_pct))
    macros.update(_ci_macros("reconFracGap", rg.get("gap"), fmt=fmt_pp))
    macros.update(_ci_macros("reconBimodalFracGap", rb.get("gap"), fmt=fmt_pp))
    macros.update(_ci_macros("fcBimodalFracMethod", fb.get("m1"), fmt=fmt_pct))
    macros.update(_ci_macros("fcBimodalFracBlur", fb.get("blur"), fmt=fmt_pct))
    macros.update(_ci_macros("fcBimodalFracGap", fb.get("gap"), fmt=fmt_pp))
    macros["bootstrapNDraws"] = str(BOOT_N_DRAWS)
    macros["bootstrapCIPct"] = format(100.0 * BOOT_CI, ".0f")
    return macros


# ------------------------------------------------ F1 track-2 across-init spread
# The headline point macros (built above) stay pinned to the canonical single
# init A. These builders add the across-init MEAN + min-max RANGE companions from
# the `kind='init_mean'` aggregate rows the runners now emit, so the report can
# state the softening trend's robustness over the twelve first-of-month 2023 synoptic
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
    """Pure: worst-case |iid_delta_crps| over CONVERGED single+init rows (C2),
    plus the smallest mean per-cell analytic CRPS over the SAME rows (R-12).

    Scans every converged per-lead, per-init row (`run` == the Converged label,
    `kind` in {single, init}); the across-init aggregate (`init_mean`) and the
    6-epoch context column are excluded. Emits `\\deltaCrpsMaxAbs` in
    sci-notation so the do-no-harm bound can be stated numerically even though
    the per-lead \\DeltaCRPS\\ rounds to +/-0.000 at three decimals, and
    `\\crpsAnalyticMinConverged`, the anchor the null is read against: pairing
    the max |delta| with the min analytic CRPS makes the null-vs-scale bound
    hold at every lead and initialisation. Empty -> placeholder."""

    converged = COL_TO_LABEL["Converged"]
    deltas, anchors = [], []

    def _finite(raw):
        v = _maybe_num(raw)
        if v is None:
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if np.isfinite(f) else None

    for r in faith_rows:
        if r.get("run") != converged or r.get("kind", "single") not in ("single", "init"):
            continue
        f = _finite(r.get("iid_delta_crps"))
        if f is not None:
            deltas.append(abs(f))
        a = _finite(r.get("iid_analytic_crps"))
        if a is not None:
            anchors.append(a)
    return {
        "deltaCrpsMaxAbs": fmt_sci(max(deltas)) if deltas else PLACEHOLDER,
        "crpsAnalyticMinConverged": fmt_pi(min(anchors)) if anchors else PLACEHOLDER,
    }


# ----------------------------------------- ERA5 spectrum bracket (S5.5) scalars
# The two numbers the Section 5.5 spectrum prose quotes are read from the native
# spherical C_ell spectra persisted for the canonical +48h step-8 run, never read
# off the figure (D-steer, 30 Jun: a real number the report is stronger for is
# emitted as a tested macro, not hand-typed). Band-power-normalised C_ell per
# field + the ERA5 reference live in spectra.npz; both scalars reproduce the
# 29-Jun log values exactly (0.054 decades; ~199x).
SPECTRUM_LARGE_ELL = 10   # "large scales": low-ell band the C1 cut-check averages over
SPECTRUM_FINE_ELL = 96    # "fine scales": high-ell band (>= O96 ring-Nyquist) for the ratio
SPECTRUM_FIELDS = ("iid_seed0", "mode_map", "mixture_mean",
                   "smoothed_map_n10", "m1_star", "m4_beta1")


def build_spectrum_macros(runs_dir):
    """Native spherical C_ell spectra (canonical +48h step-8) -> two ERA5-bracket
    scalars the Section 5.5 prose quotes via macros:

      \\spectrumDecadeAgreementMax : worst-case large-scale agreement with ERA5 --
        max over fields of the median |log10 C_ell - log10 C_ell^ERA5| over
        ell <= SPECTRUM_LARGE_ELL (~0.054 decades; every field is at or below it,
        so the report can say "every field agrees to within X of a decade").
      \\spectrumEraRatioMethod : the factor by which the imposed coherence prior
        sits below ERA5 at fine scales -- ratio of the ERA5 fine-scale
        (ell >= SPECTRUM_FINE_ELL) band-power median to JointMAP's (~199x).

    Reads only the persisted artifact; absent -> placeholders so the document
    still compiles on a fresh clone (the committed macros carry the real values).
    """

    path = Path(runs_dir) / "phase_4_fc48_14ep_step8" / "spectra.npz"
    if not path.exists():
        return {"spectrumDecadeAgreementMax": PLACEHOLDER,
                "spectrumEraRatioMethod": PLACEHOLDER}
    d = np.load(path, allow_pickle=True)
    ell, era5 = d["ell"], d["era5"]
    large, fine = ell <= SPECTRUM_LARGE_ELL, ell >= SPECTRUM_FINE_ELL
    decade = max(
        float(np.median(np.abs(np.log10(d[f][large]) - np.log10(era5[large]))))
        for f in SPECTRUM_FIELDS if f in d
    )
    ratio = float(np.median(era5[fine]) / np.median(d["m1_star"][fine]))
    return {"spectrumDecadeAgreementMax": format(decade, ".3f"),
            "spectrumEraRatioMethod": str(int(round(ratio)))}


# ------------------------------------------ across-init M1 vs smoothed-MAP gap
# Probe-#2 error bar: the headline "M1 beats smoothed-MAP at matched R-tilde"
# resampled over forecast INITS (not cells, like the bootstrap above). Init
# prefixes are discovered from the scored run dirs, so adding inits needs no edit
# here -- the canonical init A plus every `phase_4_fc48_v2_init*` replicate.
CONVERGED_HEADLINE_PREFIX = "phase_4_fc48_14ep"       # init A = 2023-11-01 (canonical)
# First-of-month 2023 replicates only (`init2023MM01`): this deliberately excludes
# the deprecated pilot inits B (..._init20231010T12) and C (..._init20231215),
# whose months are already covered by the Oct-01/Dec-01 first-of-month cases, so
# the across-init set is exactly the 12 first-of-month synoptic cases.
CONVERGED_REPLICATE_GLOB = "phase_4_fc48_v2_init2023??01"


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


# ------------------------------------------------ where the multimodal cells live
# I5 (audit): the geography of the well-separated (>2 sigma) bimodal cells, read
# STRICTLY from the FORECAST-regime +48h step-8 masks.npz (NOT the reconstruction
# audit, a different regime). bimodal = the >2 sigma well-separated subset; the
# lat_* masks are the disjoint polar/mid/tropics bands. Emits the tropical share
# and tropics-vs-global enrichment so any geography stated in prose is macro-driven.
def build_bimodal_geography_macros(run_dir):
    """Pure-ish: forecast +48h masks.npz -> tropical share + enrichment of the
    >2 sigma bimodal cells. Missing file/keys -> placeholders (never raises)."""

    macros = {
        "bimodalTropicalFracFortyEight": PLACEHOLDER,
        "bimodalTropicalEnrichFortyEight": PLACEHOLDER,
    }
    masks_path = Path(run_dir) / "masks.npz"
    if not masks_path.exists():
        return macros
    m = np.load(masks_path)
    if not {"bimodal", "lat_tropics"} <= set(m.files):
        return macros
    bimodal = m["bimodal"].astype(bool)
    tropics = m["lat_tropics"].astype(bool)
    n_bi = int(bimodal.sum())
    if n_bi == 0:
        return macros
    trop_share = float((bimodal & tropics).sum()) / n_bi          # fraction in tropics
    global_rate = bimodal.mean()                                  # global bimodal rate
    trop_rate = float((bimodal & tropics).sum()) / int(tropics.sum())
    macros["bimodalTropicalFracFortyEight"] = fmt_pct(trop_share)
    macros["bimodalTropicalEnrichFortyEight"] = (
        fmt_lambda(trop_rate / global_rate) if global_rate > 0 else PLACEHOLDER)
    return macros


# ---------------------------------------- DEC-1(c) stratum-definition constants
# The bimodal-stratum selection rule's two fixed analysis constants (the pi >= 0.1
# weight gate and the 2-sigma separation multiplier), read from the pipeline rule
# itself (`phase4_eval.practically_bimodal_mask` signature defaults) rather than
# re-typed here, so the S5.2 rendered definition can never drift from the mask
# every stratified number is computed on.
def build_stratum_constant_macros():
    """Pure: the bimodal-rule constants -> `\\stratumPiGate` / `\\stratumSepMultiplier`."""

    import inspect

    params = inspect.signature(phase4_eval.practically_bimodal_mask).parameters
    return {
        "stratumPiGate": format(params["pi_min"].default, "g"),
        "stratumSepMultiplier": format(params["min_separation"].default, "g"),
    }


# ------------------------------------------ PR-1 enrichment ratios (S1-1 rewrite)
# The S5.5 enrichment-rewrite ratios (PR-1 condition 1: every rendered ratio
# macro-emitted, never hand-typed): frac(dNLL > cut | stratum) / frac(dNLL > cut),
# computed from the persisted per-cell deltas + masks EXACTLY as the enrichment
# figure's bars are (make_phase4_figures.fig_bimodal_enrichment), for Joint MAP
# and the blur, on the 2-sigma and 1-sigma strata, at the 0.125-nat headline cut
# and the 0.5-nat deep cut (the DEC-4 deep tail, phase4_eval.DNLL_THRESHOLDS[1]).
ENRICH_CUTS = ((BOOT_THRESHOLD, "Headline"), (phase4_eval.DNLL_THRESHOLDS[1], "Deep"))
ENRICH_STRATA = (("bimodal", "TwoSigma"), ("bimodal_1sigma", "OneSigma"))
ENRICH_METHODS = ((BOOT_METHOD, "Method"), (BOOT_BLUR, "Blur"))
ENRICH_MACROS = tuple(
    f"enrich{mfix}{sfix}{cfix}"
    for _, mfix in ENRICH_METHODS for _, sfix in ENRICH_STRATA for _, cfix in ENRICH_CUTS
)


def build_enrichment_ratio_macros(run_dir):
    """Figure-faithful enrichment ratios from delta_per_cell.npz + masks.npz.

    Emits the eight `enrich{Method,Blur}{TwoSigma,OneSigma}{Headline,Deep}` ratios
    (2 d.p., matching the figure's bar heights at the headline cut). Any absent
    artifact/key -> em-dash placeholders, so a fresh clone still compiles."""

    blank = {name: PLACEHOLDER for name in ENRICH_MACROS}
    run_dir = Path(run_dir)
    delta_path, masks_path = run_dir / "delta_per_cell.npz", run_dir / "masks.npz"
    if not delta_path.exists() or not masks_path.exists():
        return blank
    with np.load(delta_path) as f:
        if not all(key in f.files for key, _ in ENRICH_METHODS):
            return blank
        deltas = {key: np.asarray(f[key], dtype=float) for key, _ in ENRICH_METHODS}
    with np.load(masks_path) as f:
        if not all(key in f.files for key, _ in ENRICH_STRATA):
            return blank
        masks = {key: np.asarray(f[key], dtype=bool) for key, _ in ENRICH_STRATA}
    macros = {}
    for mkey, mfix in ENRICH_METHODS:
        d = deltas[mkey]
        for skey, sfix in ENRICH_STRATA:
            sel = masks[skey]
            for cut, cfix in ENRICH_CUTS:
                global_frac = float(np.mean(d > cut))
                mask_frac = float(np.mean(d[sel] > cut))
                macros[f"enrich{mfix}{sfix}{cfix}"] = _fmt(
                    mask_frac / global_frac if global_frac > 0 else None, ".2f")
    return macros


# --------------------------------------- DEC-4 threshold-sensitivity macros (S5.5)
# The sensitivity sentence's numbers: the stratified (2-sigma bimodal) gap at the
# 0.25-nat (~0.7-sigma-equivalent) cut where it still favours the sampler and at
# the 0.5-nat deep cut where it reverses, plus the deep-tail stratum fraction
# (Joint MAP's frac > 0.5 nat within the stratum, the ~6% of DEC-4's
# decomposition). Thin wrapper on `_bootstrap_frac_ci` at varied threshold -- the
# exact reproduction path of the 2 Jul log entry; no new runs, and the headline
# 0.125-nat macros above are untouched.
SENSITIVITY_QUARTER_CUT = 0.25
SENSITIVITY_DEEP_CUT = phase4_eval.DNLL_THRESHOLDS[1]  # 0.5 nat


def _ci_delta_macros(base, ci_rec, fmt=fmt_delta):
    """Signed (+/-) point + CI-bound macros, for gap quantities whose sign is
    the story (the DEC-4 ruling quotes them signed: -0.034 ... +0.017). These
    are differences between two cell-shares, so `fmt_pp` is passed at the call
    sites (harvey, 4 Jul); `fmt_delta` stays the default for any future signed
    quantity that is not itself a share difference."""

    if not ci_rec:
        return {base: PLACEHOLDER, f"{base}Lo": PLACEHOLDER, f"{base}Hi": PLACEHOLDER}
    return {base: fmt(ci_rec["point"]),
            f"{base}Lo": fmt(ci_rec["lo"]), f"{base}Hi": fmt(ci_rec["hi"])}


def build_threshold_sensitivity_macros(run_dir):
    """DEC-4: `fcBimodalFracGap{QuarterNat,HalfNat}{,Lo,Hi}` + `\\fcDeepTailStratumFrac`
    from the canonical +48h run's persisted per-cell deltas. Absent -> placeholders."""

    quarter = _bootstrap_frac_ci(run_dir, ("bimodal",),
                                 threshold=SENSITIVITY_QUARTER_CUT).get("bimodal", {})
    deep = _bootstrap_frac_ci(run_dir, ("bimodal",),
                              threshold=SENSITIVITY_DEEP_CUT).get("bimodal", {})
    macros = {}
    macros.update(_ci_delta_macros("fcBimodalFracGapQuarterNat", quarter.get("gap"), fmt=fmt_pp))
    macros.update(_ci_delta_macros("fcBimodalFracGapHalfNat", deep.get("gap"), fmt=fmt_pp))
    deep_m1 = deep.get("m1")
    macros["fcDeepTailStratumFrac"] = (
        fmt_pct(deep_m1["point"]) if deep_m1 else PLACEHOLDER)
    return macros


# ------------------------------------------- PR-4 lambda* restart stability (S5.4)
# Evidence behind "stable to the optimiser's random restarts": the persisted
# seed-stability probe (robustness_probes.json `lambda_star_seed_stability`,
# run_phase4_probes.py) re-solves the resolved sweep rows with the restart RNG
# reseeded (+1000/+2000 offsets, same 4-restart optimiser) and re-runs the
# lambda* selection. The macro is the worst-case |lambda*(reseeded) - lambda*(base)|
# over the offsets -- across-RESTART variation, deliberately distinct from the
# across-INIT range macros (forecastLambdaStarConverged{Lo,Hi}), which measure
# variation across initial conditions and must never be quoted as restart evidence.
def build_restart_stability_macro(run_dir):
    """`\\lambdaStarRestartMaxShift` from the canonical run's persisted probe.
    Missing file/probe/offsets -> em-dash placeholder."""

    path = Path(run_dir) / "robustness_probes.json"
    if not path.exists():
        return {"lambdaStarRestartMaxShift": PLACEHOLDER}
    rec = json.loads(path.read_text()).get("lambda_star_seed_stability", {})
    base = rec.get("base_lambda_star")
    offsets = rec.get("offsets", {})
    shifts = [abs(float(o["lambda_star"]) - float(base))
              for o in offsets.values() if o.get("lambda_star") is not None
              ] if base is not None else []
    return {"lambdaStarRestartMaxShift": fmt_sci(max(shifts)) if shifts else PLACEHOLDER}


# ----------------------------------------------- DEC-9 resolved-band ceiling l_res
def build_spectrum_resolution_macro(runs_dir):
    """`\\spectrumEllRes`: the spectrum's resolved-band ceiling l_res, read from the
    canonical +48h step-8 spectra.npz `lmax_resolved` (the empirical Parseval
    ceiling the transform reports; 16 Jun log). Absent -> placeholder."""

    path = Path(runs_dir) / "phase_4_fc48_14ep_step8" / "spectra.npz"
    if not path.exists():
        return {"spectrumEllRes": PLACEHOLDER}
    with np.load(path) as d:
        return {"spectrumEllRes": str(int(d["lmax_resolved"]))}


# ------------------------------------------- App D optimality-certificate macros
# (decision 5, 2 Jul PM): the three toy optimality certificates quoted in the
# numerical-implementation appendix. All five numbers are DERIVED from the
# persisted certificate CSVs rather than pinned, so a re-run of the ablations
# re-derives them; the structural assertions fail loudly if the certificate
# story itself changes (in which case the appendix prose must be revisited,
# not silently re-numbered).
def build_certificate_macros(tv_rows, m5_rows):
    """Certificate macros from stage_b_tv_scores.csv + stage_d_scores.csv rows.

    * certQuadLambdaGlobalMin -- smallest lambda from which the exact min-cut
      solution of the discretised quadratic objective matches Adam's energy to
      within the discretisation's own quantisation bound (and at every larger
      lambda: the certified set must be a suffix of the grid).
    * certQuadFloorLambdaLo/Hi -- the contiguous lambda range just below that
      threshold where the cut solution is better but only within the Stage B
      restart-to-restart floor (`beats_stage_b` False).
    * certTvFailLambdaMax -- largest lambda at which Adam-on-Huber fails its
      min-cut certificate (`adam_matches_cut` False); failures must form a
      prefix of the lambda grid.
    * certLangevinDeltaJ / certLangevinChainSpread -- the single flagged
      annealed-Langevin improvement (`material_improvement` True) and that
      run's own chain-to-chain energy spread, which must contain it.

    Both inputs absent -> all placeholders (fresh-clone behaviour).
    """

    out = {name: PLACEHOLDER for name in (
        "certQuadLambdaGlobalMin", "certQuadFloorLambdaLo",
        "certQuadFloorLambdaHi", "certTvFailLambdaMax",
        "certLangevinDeltaJ", "certLangevinChainSpread")}

    quad = sorted((r for r in tv_rows if r["arm"] == "cut-quad"),
                  key=lambda r: float(r["lambda"]))
    if quad:
        certified = [r for r in quad
                     if abs(float(r["delta_j_vs_stage_b"]))
                     <= float(r["quantisation_bound"])]
        if not certified or certified != quad[len(quad) - len(certified):]:
            raise ValueError("cut-quad certified set is not a lambda-grid suffix")
        floor = [r for r in quad if r not in certified
                 and r["beats_stage_b"] == "False"]
        below = [r for r in quad if r not in certified and r not in floor]
        if not floor or [float(r["lambda"]) for r in below + floor + certified] \
                != [float(r["lambda"]) for r in quad]:
            raise ValueError("cut-quad within-floor range is not contiguous "
                             "below the certified threshold")
        out["certQuadLambdaGlobalMin"] = format(float(certified[0]["lambda"]), "g")
        out["certQuadFloorLambdaLo"] = format(float(floor[0]["lambda"]), "g")
        out["certQuadFloorLambdaHi"] = format(float(floor[-1]["lambda"]), "g")

    cut_tv = sorted((r for r in tv_rows if r["arm"] == "cut-tv"),
                    key=lambda r: float(r["lambda"]))
    if cut_tv:
        fails = [r for r in cut_tv if r["adam_matches_cut"] == "False"]
        if not fails or fails != cut_tv[:len(fails)]:
            raise ValueError("adam-tv certificate failures are not a "
                             "lambda-grid prefix")
        out["certTvFailLambdaMax"] = format(float(fails[-1]["lambda"]), "g")

    flagged = [r for r in m5_rows if r["material_improvement"] == "True"]
    if flagged:
        if len(flagged) != 1:
            raise ValueError("expected exactly one flagged Langevin improvement")
        delta_j = float(flagged[0]["delta_j_vs_method1"])
        spread = float(flagged[0]["chain_energy_spread"])
        if not abs(delta_j) < spread:
            raise ValueError("flagged Langevin delta-J is not inside its own "
                             "chain-energy spread; appendix claim invalid")
        out["certLangevinDeltaJ"] = format(delta_j, ".4f")
        out["certLangevinChainSpread"] = format(spread, ".3f")

    return out


# --------------------------------------------------- App E compute-cost macros
# (decision 6, 2 Jul PM): training wall-times from the once-persisted sacct
# record (outputs/runs/compute_record.json -- scheduler accounting ages out,
# which is why the record is a committed artifact) and the sampler audit cost
# from the pipeline's own timings.json.
def _elapsed_to_hours(text):
    h, m, s = (int(part) for part in str(text).split(":"))
    return h + m / 60.0 + s / 3600.0


def build_compute_macros(record, timings):
    """`\\compute*` macros from compute_record.json + phase_4_real/timings.json.

    `record`/`timings`: parsed JSON dicts (or None when the file is absent ->
    placeholders). The audit total sums the timed stages, excluding the
    `icm_s_per_sweep` entry, which is a per-sweep rate rather than a stage.
    """

    out = {name: PLACEHOLDER for name in (
        "computeAeWallHours", "computeForecastWallHours",
        "computeExtractionWallMinutes", "computeAuditWallSeconds")}
    if record:
        jobs = record["jobs"]
        out["computeAeWallHours"] = format(
            _elapsed_to_hours(jobs["ae_training"]["elapsed"]), ".1f")
        out["computeForecastWallHours"] = format(
            _elapsed_to_hours(jobs["forecast_training_v2"]["elapsed"]), ".1f")
        out["computeExtractionWallMinutes"] = format(
            _elapsed_to_hours(jobs["extraction"]["elapsed"]) * 60.0, ".1f")
    if timings:
        total = sum(v for k, v in timings.items() if k != "icm_s_per_sweep")
        out["computeAuditWallSeconds"] = format(total, ".0f")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--soft-csv", type=Path, default=SOFT_CSV)
    parser.add_argument("--faith-csv", type=Path, default=FAITH_CSV)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--ae-run-dir", type=Path, default=AE_RUN_DIR)
    parser.add_argument("--toy-csv", type=Path, default=TOY_SCORES_CSV)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--allow-empty", action="store_true",
                        help="emit even when the forecast artifacts are absent "
                             "(writes em-dash placeholders; off by default so a "
                             "stray run cannot clobber the committed macros)")
    args = parser.parse_args()

    soft_rows = _read_csv_numeric(args.soft_csv)
    faith_rows = _read_csv_numeric(args.faith_csv)
    lambda_stars = _load_lambda_stars(args.runs_dir)
    # Toy (Ch 4) artifacts are independent of the forecast pipeline and never gate
    # the clobber guard below: a toy-only run must not license overwriting the
    # forecast macros. Absent -> placeholders (em-dash).
    toy_rows = _read_csv_numeric(args.toy_csv)

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
    # I5: geography of the >2 sigma bimodal cells, from the same converged +48h
    # step-8 masks the bootstrap reads (forecast regime, not the recon audit).
    macros = {**macros, **build_bimodal_geography_macros(
        args.runs_dir / "phase_4_fc48_14ep_step8")}
    # S5.5: ERA5 spectrum bracket scalars (large-scale decade agreement +
    # fine-scale headroom ratio), read from the canonical +48h step-8 spectra.npz.
    macros = {**macros, **build_spectrum_macros(args.runs_dir)}
    # I2: Chapter-4 synthetic-testbed bracket macros (Stage-A baselines).
    macros = {**macros, **build_toy_baseline_macros(toy_rows)}
    # B5: Chapter-4 Section 4.2 operating-point off-mode smear fractions, recomputed
    # from the committed toy fields (figure-faithful) so the quoted number cannot
    # drift from its non-smearing figure.
    macros = {**macros, **build_toy_smear_macros(args.runs_dir, TOY_DATA_DIR)}
    # Round-1 review emissions (2 Jul worklist), appended after the verified set
    # so the committed file grows strictly additively:
    # DEC-1(c): the bimodal-stratum rule's fixed analysis constants.
    macros = {**macros, **build_stratum_constant_macros()}
    # PR-1 condition 1: the S5.5 enrichment-rewrite ratios (figure-faithful).
    macros = {**macros, **build_enrichment_ratio_macros(
        args.runs_dir / "phase_4_fc48_14ep_step8")}
    # DEC-4: threshold-sensitivity gap CIs + deep-tail stratum fraction.
    macros = {**macros, **build_threshold_sensitivity_macros(
        args.runs_dir / "phase_4_fc48_14ep_step8")}
    # PR-4: lambda* stability across the optimiser's random restarts.
    macros = {**macros, **build_restart_stability_macro(
        args.runs_dir / "phase_4_fc48_14ep_step8")}
    # DEC-9: the spectrum's resolved-band ceiling l_res.
    macros = {**macros, **build_spectrum_resolution_macro(args.runs_dir)}
    # App D (decision 5, 2 Jul PM): toy optimality-certificate numbers, derived
    # from the persisted Stage B-TV / Stage D certificate CSVs.
    macros = {**macros, **build_certificate_macros(
        _read_csv_numeric(args.runs_dir / "stage_b_tv_ablation" / "stage_b_tv_scores.csv"),
        _read_csv_numeric(args.runs_dir / "stage_d_method5_langevin" / "stage_d_scores.csv"))}
    # App E (decision 6, 2 Jul PM): compute costs from the once-persisted sacct
    # record + the sampler pipeline's timings.json.
    compute_record_path = args.runs_dir / "compute_record.json"
    timings_path = args.ae_run_dir / "timings.json"
    macros = {**macros, **build_compute_macros(
        json.loads(compute_record_path.read_text()) if compute_record_path.exists() else None,
        json.loads(timings_path.read_text()) if timings_path.exists() else None)}
    tables = {
        "pi_softening.tex": build_pi_softening_table(soft_rows),
        "lambda_calibration.tex": build_lambda_table(lambda_stars, soft_rows),
        "faithfulness.tex": build_faithfulness_table(faith_rows, soft_rows),
        "toy_baselines.tex": build_toy_baseline_table(toy_rows),
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
