"""Run the +48 h Joint MAP experiment with variance normalisation in the objective.

The production Method 1 objective uses raw edge roughness during optimisation and
only reports the variance-normalised roughness afterwards.  This separate
experiment instead minimises

    J_lambda(x) = NLL(x) / N + lambda * R_tilde(x)
    R_tilde(x) = [(x.T @ L @ x) / |E|] / Var(x).

It reuses the canonical +48 h graph, anchors and extracted GMM modes, but writes
all new artifacts to a separate run directory.  The resulting regional figure
has the same six panel roles as thesis Figure 5.2 / ``region_maps.png``; its two
Joint MAP panels (matched roughness and the 95% faithfulness-budget point) both
come from the variance-normalised objective.

Run from the repository root after the canonical +48 h data and Phase 4 run
artifacts have been restored/generated::

    .venv/bin/python scripts/run_variance_normalised_map.py

Default outputs::

    outputs/runs/phase_4_fc48_14ep_step8_variance_normalised/
    outputs/figures/forecast_14ep_step8/region_maps_variance_normalised.png

The canonical run directory and ``region_maps.png`` are read-only inputs and are
never overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sampler_research.baselines import gmm_nll_over_n, mode_field
from sampler_research.graph import field_variance, scale_free_roughness, sparse_laplacian
from sampler_research.io import load_real_marginal
from sampler_research.method4_mrf import delta_nll_to_best_mode
from sampler_research.phase4_eval import select_lambda_star
from sampler_research.regularised_map import nll_gradient

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_NPZ = REPO_ROOT / "outputs" / "data" / "phase_4_fc48_14ep_step8_2t.npz"
BASE_RUN_DIR = REPO_ROOT / "outputs" / "runs" / "phase_4_fc48_14ep_step8"
OUT_DIR = REPO_ROOT / "outputs" / "runs" / (
    "phase_4_fc48_14ep_step8_variance_normalised"
)
FIGURE_PATH = REPO_ROOT / "outputs" / "figures" / "forecast_14ep_step8" / (
    "region_maps_variance_normalised.png"
)

LAMBDA_GRID = (0.0, 0.5, 2.0, 8.0, 30.0, 100.0, 300.0, 1000.0)
N_RESTARTS = 4
N_STEPS = 400
LR = 0.05
VAR_FLOOR = 1e-12
TARGET_ANCHOR = "smoothed_map_n10"

KEEP_FRAC = 0.95
DNLL_THRESHOLD = 0.125
BUDGET_TOL = 0.005


@dataclass
class VarianceNormalisedResult:
    """Result of one multi-restart solve at a fixed lambda."""

    field: np.ndarray
    energy: float
    nll_over_n: float
    r_tilde: float
    variance: float
    variance_collapsed: bool
    restart_energies: np.ndarray
    restart_spread: float
    restart_field_spread: float
    lr_used: float | None = None


def _validate_objective_inputs(field, laplacian, n_edges, var_floor):
    field = np.asarray(field, dtype=float)
    if field.size < 2:
        raise ValueError("variance-normalised roughness needs at least two cells")
    if laplacian.shape != (field.size, field.size):
        raise ValueError("laplacian shape must match the flattened field size")
    if n_edges <= 0:
        raise ValueError("n_edges must be positive")
    if var_floor <= 0.0:
        raise ValueError("var_floor must be positive")
    return field


def variance_normalised_roughness(
    field,
    laplacian,
    n_edges,
    *,
    var_floor=VAR_FLOOR,
):
    """Return ``S_edge(x) / Var(x)`` using the graph's optimisation weights.

    Unlike the reporting helper, this function raises on variance collapse.  A
    denominator floor would make an exactly constant field score as zero and
    quietly restore the collapse loophole that the normalisation is intended to
    remove.
    """

    field = _validate_objective_inputs(field, laplacian, n_edges, var_floor)
    flat = field.reshape(-1)
    variance = field_variance(flat)
    if variance <= var_floor:
        raise FloatingPointError(
            f"field variance {variance:.3g} is at/below floor {var_floor:.3g}"
        )
    smooth = float(flat @ (laplacian @ flat)) / float(n_edges)
    return smooth / variance


def variance_normalised_objective(
    field,
    pi,
    mu,
    sigma,
    lam,
    laplacian,
    n_edges,
    *,
    var_floor=VAR_FLOOR,
):
    """``NLL/N + lambda * S_edge/Var`` for the experimental optimiser."""

    if lam < 0.0:
        raise ValueError("lambda must be non-negative")
    nll = gmm_nll_over_n(field, pi, mu, sigma)
    if lam == 0.0:
        return nll
    r_tilde = variance_normalised_roughness(
        field, laplacian, n_edges, var_floor=var_floor
    )
    return nll + float(lam) * r_tilde


def variance_normalised_gradient(
    field,
    pi,
    mu,
    sigma,
    lam,
    laplacian,
    n_edges,
    *,
    var_floor=VAR_FLOOR,
):
    """Closed-form gradient of :func:`variance_normalised_objective`.

    For ``S = x.T L x / |E|`` and ``V = mean((x - mean(x))**2)``,

    ``grad(S/V) = grad(S)/V - S*grad(V)/V**2``.
    """

    if lam < 0.0:
        raise ValueError("lambda must be non-negative")
    field = _validate_objective_inputs(field, laplacian, n_edges, var_floor)
    grad_nll = nll_gradient(field, pi, mu, sigma) / field.size
    if lam == 0.0:
        return grad_nll

    flat = field.reshape(-1)
    centred = flat - flat.mean()
    variance = float(np.mean(centred**2))
    if variance <= var_floor:
        raise FloatingPointError(
            f"field variance {variance:.3g} is at/below floor {var_floor:.3g}"
        )

    lap_x = np.asarray(laplacian @ flat, dtype=float)
    smooth = float(flat @ lap_x) / float(n_edges)
    grad_smooth = 2.0 * lap_x / float(n_edges)
    grad_variance = 2.0 * centred / float(flat.size)
    grad_ratio = grad_smooth / variance - smooth * grad_variance / variance**2
    return grad_nll + float(lam) * grad_ratio.reshape(field.shape)


def _adam_descent(x0, grad_fn, n_steps, lr, betas=(0.9, 0.999), eps=1e-8):
    x = np.asarray(x0, dtype=float).copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2 = betas
    for step in range(1, n_steps + 1):
        grad = np.asarray(grad_fn(x), dtype=float)
        if not np.all(np.isfinite(grad)):
            raise FloatingPointError(f"non-finite gradient at Adam step {step}")
        m = b1 * m + (1.0 - b1) * grad
        v = b2 * v + (1.0 - b2) * grad * grad
        m_hat = m / (1.0 - b1**step)
        v_hat = v / (1.0 - b2**step)
        x -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return x


def minimise_variance_normalised(
    pi,
    mu,
    sigma,
    lam,
    *,
    edges,
    laplacian,
    n_restarts=N_RESTARTS,
    n_steps=N_STEPS,
    lr=LR,
    restart_scale=0.15,
    rng=None,
    var_floor=VAR_FLOOR,
):
    """Minimise the variance-normalised objective from the Method 1 starts."""

    pi = np.asarray(pi, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    edges = np.asarray(edges, dtype=np.int64)
    if pi.ndim != 2 or pi.shape != mu.shape or pi.shape != sigma.shape:
        raise ValueError("pi, mu and sigma must share an [N, K] shape")
    if laplacian.shape != (pi.shape[0], pi.shape[0]):
        raise ValueError("laplacian shape must match the number of cells")
    if edges.ndim != 2 or edges.shape[1] != 2 or len(edges) == 0:
        raise ValueError("edges must be a non-empty [E, 2] array")
    if n_restarts < 1 or n_steps < 1 or lr <= 0.0 or restart_scale < 0.0:
        raise ValueError("invalid optimiser settings")

    rng = np.random.default_rng() if rng is None else rng
    warm, _ = mode_field(pi, mu, sigma)
    n_edges = len(edges)

    def grad_fn(x):
        return variance_normalised_gradient(
            x,
            pi,
            mu,
            sigma,
            lam,
            laplacian,
            n_edges,
            var_floor=var_floor,
        )

    fields = []
    energies = []
    for restart in range(n_restarts):
        x0 = (
            warm.copy()
            if restart == 0
            else warm + rng.normal(scale=restart_scale, size=warm.shape)
        )
        x_opt = _adam_descent(x0, grad_fn, n_steps=n_steps, lr=lr)
        energy = variance_normalised_objective(
            x_opt,
            pi,
            mu,
            sigma,
            lam,
            laplacian,
            n_edges,
            var_floor=var_floor,
        )
        fields.append(x_opt)
        energies.append(energy)

    fields = np.stack(fields)
    energies = np.asarray(energies, dtype=float)
    if not np.any(np.isfinite(energies)):
        raise FloatingPointError("every restart ended with a non-finite objective")
    best = int(np.nanargmin(energies))
    best_field = fields[best]
    r_tilde, collapsed = scale_free_roughness(best_field, edges)
    field_spread = float(np.max(fields.max(axis=0) - fields.min(axis=0)))
    return VarianceNormalisedResult(
        field=best_field,
        energy=float(energies[best]),
        nll_over_n=gmm_nll_over_n(best_field, pi, mu, sigma),
        r_tilde=float(r_tilde),
        variance=field_variance(best_field),
        variance_collapsed=bool(collapsed),
        restart_energies=energies,
        restart_spread=float(np.nanmax(energies) - np.nanmin(energies)),
        restart_field_spread=field_spread,
    )


def _solve_guarded(
    pi,
    mu,
    sigma,
    lam,
    edges,
    laplacian,
    *,
    seed,
    n_restarts,
    n_steps,
    lr,
    restart_scale,
):
    """Mirror the production warm-start guard, with one learning-rate retry."""

    warm, _ = mode_field(pi, mu, sigma)
    warm_energy = variance_normalised_objective(
        warm, pi, mu, sigma, lam, laplacian, len(edges)
    )
    lr_used = float(lr)
    last = None
    for attempt in range(2):
        last = minimise_variance_normalised(
            pi,
            mu,
            sigma,
            lam,
            edges=edges,
            laplacian=laplacian,
            n_restarts=n_restarts,
            n_steps=n_steps,
            lr=lr_used,
            restart_scale=restart_scale,
            rng=np.random.default_rng(seed),
        )
        if (
            np.isfinite(last.energy)
            and last.energy <= warm_energy + 1e-9
            and not last.variance_collapsed
        ):
            last.lr_used = lr_used
            return last
        if attempt == 0:
            print(
                f"[variance-normalised] lambda={lam:g}: failed warm-start guard; "
                "halving lr once"
            )
            lr_used *= 0.5
    raise RuntimeError(
        f"lambda={lam:g} failed the warm-start/variance guard at lr={lr_used:g}; "
        f"last energy={last.energy if last is not None else None}"
    )


def parse_lambdas(value):
    """Parse a comma-separated, non-negative lambda grid."""

    try:
        lambdas = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("lambdas must be comma-separated numbers") from exc
    if not lambdas or any(not np.isfinite(x) or x < 0.0 for x in lambdas):
        raise argparse.ArgumentTypeError("lambdas must be finite and non-negative")
    if not any(x > 0.0 for x in lambdas):
        raise argparse.ArgumentTypeError("at least one positive lambda is required")
    return tuple(sorted(set(lambdas)))


def _budget_bracket(lambdas, fractions, target):
    """Return positive-lambda rows immediately below/above a budget crossing."""

    rows = sorted(
        (float(lam), float(frac))
        for lam, frac in zip(lambdas, fractions)
        if lam > 0.0 and np.isfinite(frac)
    )
    below = [row for row in rows if row[1] <= target]
    if not below:
        raise ValueError("no positive-lambda sweep row satisfies the faithfulness budget")
    lo = max(below, key=lambda row: row[0])
    above = [row for row in rows if row[0] > lo[0] and row[1] > target]
    if not above:
        raise ValueError("sweep does not reach above the faithfulness-budget crossing")
    hi = min(above, key=lambda row: row[0])
    return lo, hi


def interpolate_budget_lambda(lo, hi, target=1.0 - KEEP_FRAC):
    """Interpolate a faithfulness-budget crossing linearly in log(lambda)."""

    (lam_lo, frac_lo), (lam_hi, frac_hi) = lo, hi
    if not (0.0 < lam_lo < lam_hi):
        raise ValueError("budget interpolation requires 0 < lambda_lo < lambda_hi")
    if not (frac_lo <= target <= frac_hi):
        raise ValueError("budget target is not bracketed by the supplied rows")
    if np.isclose(frac_lo, frac_hi):
        return float(np.sqrt(lam_lo * lam_hi))
    weight = (target - frac_lo) / (frac_hi - frac_lo)
    return float(np.exp(np.log(lam_lo) + weight * (np.log(lam_hi) - np.log(lam_lo))))


def _mode_fraction(field, pi, mu, sigma, mode_values, valid_mask):
    delta = delta_nll_to_best_mode(field, pi, mu, sigma, mode_values, valid_mask)
    return float(np.mean(delta["per_cell"] > DNLL_THRESHOLD))


def _extend_lambdas(existing, direction):
    positive = np.asarray([x for x in existing if x > 0.0], dtype=float)
    if direction == "up":
        base = float(positive.max())
        return (base * np.sqrt(10.0), base * 10.0)
    if direction == "down":
        base = float(positive.min())
        return (base / 10.0, base / np.sqrt(10.0))
    raise ValueError(f"unknown extension direction {direction!r}")


def _load_required_npz(path, keys):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"required artifact is absent: {path}")
    with np.load(path) as data:
        missing = [key for key in keys if key not in data.files]
        if missing:
            raise KeyError(f"{path} is missing arrays: {missing}")
        return {key: np.asarray(data[key]) for key in keys}


def _figure_fields(base_run_dir, star_field, star_lambda, budget_field, budget_lambda):
    """Build Figure-5.2 fields in the canonical row-wise order."""

    anchors = _load_required_npz(
        Path(base_run_dir) / "anchors.npz",
        ("mode_map", "iid_seed0", TARGET_ANCHOR),
    )
    fields = {
        "Per-cell MAP": anchors["mode_map"],
        "Independent draw": anchors["iid_seed0"],
        (
            "Joint MAP "
            f"($\\widetilde{{R}}$ penalty, $\\lambda^\\star={star_lambda:.3g}$)"
        ): star_field,
        "Smoothed MAP (n=10)": anchors[TARGET_ANCHOR],
        (
            "Joint MAP "
            f"($\\widetilde{{R}}$ penalty, $\\lambda={budget_lambda:.3g}$ budget)"
        ): budget_field,
    }
    era5_path = Path(base_run_dir) / "era5_reference.npz"
    if era5_path.exists():
        fields["ERA5 reference"] = _load_required_npz(era5_path, ("era5",))["era5"]
    else:
        print(f"[figure] WARNING: {era5_path} absent; rendering without ERA5")
    return fields


def _lead_hours(data_path, base_run_dir):
    meta_path = Path(data_path).with_name(Path(data_path).stem + "_meta.json")
    if meta_path.exists():
        try:
            return float(json.loads(meta_path.read_text())["lead_hours"])
        except (KeyError, TypeError, ValueError, OSError):
            pass
    provenance = Path(base_run_dir) / "provenance.json"
    if provenance.exists():
        try:
            regime = json.loads(provenance.read_text())["regime"]
            return float(regime["lead_hours"])
        except (KeyError, TypeError, ValueError, OSError):
            pass
    return 48.0


def render_region_figure(
    latlons,
    fields,
    figure_path,
    *,
    lead_hours=48.0,
):
    """Render with the same regional plotting helper and layout as Figure 5.2."""

    # Running ``python scripts/...`` puts this directory on sys.path, allowing
    # us to reuse the canonical renderer without changing its global run paths.
    from make_real_figures import _plot_region_fields

    fig, _ = _plot_region_fields(
        latlons,
        fields,
        suptitle=(
            f"Forecast +{lead_hours:g} h: 2-metre temperature, "
            "North Atlantic / Europe"
        ),
        figsize=(10.8, 10.8),
        top=0.91,
        hspace=0.18,
    )
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    print(f"[figure] wrote {figure_path}")


def _save_outputs(out_dir, sweep_rows, star, budget, record):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(sweep_rows, key=lambda item: item[0])
    np.savez(
        out_dir / "variance_normalised_sweep.npz",
        lambdas=np.asarray([item[0] for item in ordered]),
        fields=np.stack([item[1].field for item in ordered]),
        energy=np.asarray([item[1].energy for item in ordered]),
        nll_over_n=np.asarray([item[1].nll_over_n for item in ordered]),
        r_tilde=np.asarray([item[1].r_tilde for item in ordered]),
        variance=np.asarray([item[1].variance for item in ordered]),
        off_mode_fraction=np.asarray([item[2] for item in ordered]),
        restart_spread=np.asarray([item[1].restart_spread for item in ordered]),
        restart_field_spread=np.asarray(
            [item[1].restart_field_spread for item in ordered]
        ),
        lr_used=np.asarray([item[1].lr_used for item in ordered]),
    )
    np.savez(
        out_dir / "variance_normalised_operating_points.npz",
        field_star=star.field,
        field_budget=budget.field,
        lambda_star=np.float64(record["lambda_star"]),
        lambda_budget=np.float64(record["lambda_budget"]),
    )
    (out_dir / "results.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"[output] wrote variance-normalised artifacts under {out_dir}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_NPZ)
    parser.add_argument("--base-run-dir", type=Path, default=BASE_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--figure", type=Path, default=FIGURE_PATH)
    parser.add_argument(
        "--lambdas",
        type=parse_lambdas,
        default=LAMBDA_GRID,
        help="comma-separated initial lambda grid",
    )
    parser.add_argument("--n-restarts", type=int, default=N_RESTARTS)
    parser.add_argument("--n-steps", type=int, default=N_STEPS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-extensions", type=int, default=3)
    parser.add_argument("--skip-figure", action="store_true")
    args = parser.parse_args(argv)

    required = (
        args.data,
        args.base_run_dir / "masks.npz",
        args.base_run_dir / "anchors.npz",
        args.base_run_dir / "modes.npz",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error(
            "the +48 h input/run artifacts must be restored or generated first; missing:\n  "
            + "\n  ".join(missing)
        )
    if args.n_restarts < 1 or args.n_steps < 1 or args.lr <= 0.0:
        parser.error("n-restarts, n-steps and lr must be positive")
    if args.max_extensions < 0:
        parser.error("max-extensions must be non-negative")

    data = load_real_marginal(args.data)
    pi, mu, sigma = data["pi"], data["mu"], data["sigma"]
    masks = _load_required_npz(args.base_run_dir / "masks.npz", ("edges",))
    anchors = _load_required_npz(
        args.base_run_dir / "anchors.npz", (TARGET_ANCHOR,)
    )
    modes = _load_required_npz(
        args.base_run_dir / "modes.npz", ("mode_values", "valid_mask")
    )
    edges = masks["edges"].astype(np.int64, copy=False)
    laplacian = sparse_laplacian(pi.shape[0], edges)
    target_r_tilde, target_collapsed = scale_free_roughness(
        anchors[TARGET_ANCHOR], edges
    )
    if target_collapsed:
        parser.error(f"{TARGET_ANCHOR} is variance-collapsed")
    restart_scale = float(np.median(sigma))
    print(
        f"[setup] loaded N={pi.shape[0]}, K={pi.shape[1]}, |E|={len(edges)}; "
        f"target R~={target_r_tilde:.6g}; restart scale={restart_scale:.4g}"
    )

    sweep_rows = []
    solved_lambdas = set()

    def solve_and_add(lam):
        lam = float(lam)
        if lam in solved_lambdas:
            return
        result = _solve_guarded(
            pi,
            mu,
            sigma,
            lam,
            edges,
            laplacian,
            seed=args.seed + len(sweep_rows),
            n_restarts=args.n_restarts,
            n_steps=args.n_steps,
            lr=args.lr,
            restart_scale=restart_scale,
        )
        frac = _mode_fraction(
            result.field,
            pi,
            mu,
            sigma,
            modes["mode_values"],
            modes["valid_mask"],
        )
        sweep_rows.append((lam, result, frac))
        solved_lambdas.add(lam)
        print(
            f"[sweep] lambda={lam:g}: J={result.energy:.5g} "
            f"NLL/N={result.nll_over_n:.5g} R~={result.r_tilde:.6g} "
            f"Var={result.variance:.5g} off-mode={frac:.2%}"
        )

    for lam in args.lambdas:
        solve_and_add(lam)

    # First ensure that the matched-roughness operating point is bracketed.
    for extension in range(args.max_extensions + 1):
        lambdas = np.asarray([item[0] for item in sweep_rows])
        r_tildes = np.asarray([item[1].r_tilde for item in sweep_rows])
        collapsed = np.asarray([item[1].variance_collapsed for item in sweep_rows])
        selection = select_lambda_star(
            lambdas, r_tildes, collapsed, target_r_tilde
        )
        if selection.bracketed:
            break
        if extension == args.max_extensions:
            raise RuntimeError(
                "could not bracket the matched-roughness point after "
                f"{args.max_extensions} lambda-grid extensions"
            )
        new_lambdas = _extend_lambdas(lambdas, selection.extend_direction)
        print(
            f"[sweep] R~ target unbracketed ({selection.extend_direction}); "
            f"extending with {new_lambdas}"
        )
        for lam in new_lambdas:
            solve_and_add(lam)

    # The budget point should normally lie below lambda*.  Extend if its 5%
    # crossing is not yet in the same sweep.
    budget_target = 1.0 - KEEP_FRAC
    for extension in range(args.max_extensions + 1):
        lambdas = np.asarray([item[0] for item in sweep_rows])
        fractions = np.asarray([item[2] for item in sweep_rows])
        try:
            budget_lo, budget_hi = _budget_bracket(
                lambdas, fractions, budget_target
            )
            break
        except ValueError as exc:
            if extension == args.max_extensions:
                raise RuntimeError(str(exc)) from exc
            positive_frac = fractions[lambdas > 0.0]
            direction = "up" if np.all(positive_frac <= budget_target) else "down"
            new_lambdas = _extend_lambdas(lambdas, direction)
            print(f"[sweep] budget unbracketed ({direction}); extending with {new_lambdas}")
            for lam in new_lambdas:
                solve_and_add(lam)

    # Re-select after any budget-driven extensions, then solve the interpolated
    # operating point rather than substituting a neighbouring sweep field.
    lambdas = np.asarray([item[0] for item in sweep_rows])
    selection = select_lambda_star(
        lambdas,
        np.asarray([item[1].r_tilde for item in sweep_rows]),
        np.asarray([item[1].variance_collapsed for item in sweep_rows]),
        target_r_tilde,
    )
    if not selection.bracketed:
        raise RuntimeError("matched-roughness selection lost its bracket")
    star = _solve_guarded(
        pi,
        mu,
        sigma,
        selection.lambda_star,
        edges,
        laplacian,
        seed=args.seed + 10_000,
        n_restarts=args.n_restarts,
        n_steps=args.n_steps,
        lr=args.lr,
        restart_scale=restart_scale,
    )

    # Match the canonical budget procedure: log-lambda interpolation, followed
    # by at most one corrective interpolation if the achieved fraction misses by
    # more than half a percentage point.
    budget_lambda = interpolate_budget_lambda(
        budget_lo, budget_hi, target=budget_target
    )
    budget = None
    budget_frac = None
    budget_solves = []
    for attempt in range(2):
        budget = _solve_guarded(
            pi,
            mu,
            sigma,
            budget_lambda,
            edges,
            laplacian,
            seed=args.seed + 20_000,
            n_restarts=args.n_restarts,
            n_steps=args.n_steps,
            lr=args.lr,
            restart_scale=restart_scale,
        )
        budget_frac = _mode_fraction(
            budget.field,
            pi,
            mu,
            sigma,
            modes["mode_values"],
            modes["valid_mask"],
        )
        budget_solves.append((float(budget_lambda), float(budget_frac)))
        if abs(budget_frac - budget_target) <= BUDGET_TOL or attempt == 1:
            break
        if budget_frac > budget_target:
            budget_hi = (budget_lambda, budget_frac)
        else:
            budget_lo = (budget_lambda, budget_frac)
        budget_lambda = interpolate_budget_lambda(
            budget_lo, budget_hi, target=budget_target
        )

    record = {
        "objective": "NLL/N + lambda * ((x^T L x / |E|) / Var(x))",
        "data_path": str(args.data.resolve()),
        "base_run_dir": str(args.base_run_dir.resolve()),
        "target_anchor": TARGET_ANCHOR,
        "target_r_tilde": float(target_r_tilde),
        "lambda_star": float(selection.lambda_star),
        "star_r_tilde": float(star.r_tilde),
        "star_nll_over_n": float(star.nll_over_n),
        "star_variance": float(star.variance),
        "lambda_budget": float(budget_lambda),
        "budget_off_mode_fraction": float(budget_frac),
        "budget_r_tilde": float(budget.r_tilde),
        "budget_nll_over_n": float(budget.nll_over_n),
        "budget_variance": float(budget.variance),
        "budget_keep_fraction_target": KEEP_FRAC,
        "dnll_threshold": DNLL_THRESHOLD,
        "budget_solves": budget_solves,
        "n_restarts": args.n_restarts,
        "n_steps": args.n_steps,
        "lr": args.lr,
        "restart_scale": restart_scale,
        "seed": args.seed,
    }
    _save_outputs(args.out_dir, sweep_rows, star, budget, record)

    if not args.skip_figure:
        fields = _figure_fields(
            args.base_run_dir,
            star.field,
            selection.lambda_star,
            budget.field,
            budget_lambda,
        )
        render_region_figure(
            data["latlons"],
            fields,
            args.figure,
            lead_hours=_lead_hours(args.data, args.base_run_dir),
        )

    print(
        f"[done] lambda*={selection.lambda_star:.6g} "
        f"(R~={star.r_tilde:.6g}; target={target_r_tilde:.6g}); "
        f"budget lambda={budget_lambda:.6g} (off-mode={budget_frac:.2%})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
