"""Input/output helpers for sampler research data."""

import json
from pathlib import Path

import numpy as np

SAMPLER_KEYS = ("pi", "mu", "sigma", "coords")
REAL_MARGINAL_KEYS = ("pi", "mu_2t", "sigma_2t", "latlons")
# Per-lead keys in a forecast-format GMM dict (gmm_inference.py --forecast-steps).
FORECAST_STEP_KEYS = ("pi", "mu_channel", "sigma_channel", "latlons")


def load_npz(path):
    """Load all arrays and metadata from a `.npz` file."""

    with np.load(Path(path), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_sampler_arrays(path):
    """Load only sampler-facing arrays from a toy `.npz` file."""

    data = load_npz(path)
    missing = [key for key in SAMPLER_KEYS if key not in data]
    if missing:
        raise KeyError(f"{path} is missing sampler arrays: {missing}")
    return {key: np.asarray(data[key]) for key in SAMPLER_KEYS}


def load_real_marginal(path):
    """Load the 2t marginal of a converted real GMM `.npz` file.

    Returns `{"pi": [N,K], "mu": [N,K], "sigma": [N,K], "latlons": [N,2]}`,
    all float64. Uses the precomputed `mu_2t`/`sigma_2t` aliases (audited
    bitwise-equal to `mu[:, :, 3]`); does NOT load the `[N, K, 72]` tensors.
    Validates: all finite; `pi >= 0` and rows sum to 1 (atol 1e-5); `sigma > 0`.
    """

    with np.load(Path(path), allow_pickle=False) as data:
        missing = [key for key in REAL_MARGINAL_KEYS if key not in data.files]
        if missing:
            raise KeyError(f"{path} is missing real-marginal arrays: {missing}")
        pi = np.asarray(data["pi"], dtype=np.float64)
        mu = np.asarray(data["mu_2t"], dtype=np.float64)
        sigma = np.asarray(data["sigma_2t"], dtype=np.float64)
        latlons = np.asarray(data["latlons"], dtype=np.float64)

    if pi.ndim != 2 or pi.shape != mu.shape or pi.shape != sigma.shape:
        raise ValueError("pi, mu_2t, and sigma_2t must share an [N, K] shape")
    if latlons.shape != (pi.shape[0], 2):
        raise ValueError("latlons must have shape [N, 2]")
    for name, arr in (("pi", pi), ("mu_2t", mu), ("sigma_2t", sigma), ("latlons", latlons)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values")
    if np.any(pi < 0.0):
        raise ValueError("pi contains negative weights")
    if not np.allclose(pi.sum(axis=-1), 1.0, atol=1e-5):
        raise ValueError("pi rows must sum to 1 (atol 1e-5)")
    if np.any(sigma <= 0.0):
        raise ValueError("sigma_2t must be strictly positive")

    return {"pi": pi, "mu": mu, "sigma": sigma, "latlons": latlons}


def save_npz(path, arrays):
    """Save arrays/metadata to `.npz`, creating the parent directory."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path


def forecast_step_to_marginal(step):
    """Map one forecast-format step dict to the `load_real_marginal` npz contract.

    A forecast step (from `gmm_inference.py --forecast-steps N`) carries the
    selected-channel marginals as `mu_channel`/`sigma_channel`; this renames them
    to the `mu_2t`/`sigma_2t` aliases the loader expects. Accepts numpy arrays or
    torch tensors (via `np.asarray`); returns plain numpy arrays.
    """

    missing = [k for k in FORECAST_STEP_KEYS if k not in step]
    if missing:
        raise KeyError(f"forecast step missing keys: {missing}")
    return {
        "pi": np.asarray(step["pi"]),
        "mu_2t": np.asarray(step["mu_channel"]),
        "sigma_2t": np.asarray(step["sigma_channel"]),
        "latlons": np.asarray(step["latlons"]),
    }


def write_forecast_marginals(obj, out_dir, prefix=None):
    """Emit one `{prefix}_step{k}_2t.npz` + `_meta.json` per forecast lead step.

    `obj` is a forecast-format GMM dict: a top-level `steps` mapping the absolute
    forecast step `k` to a step dict, plus shared metadata (`target_datetime` =
    init time, `from_run_id`, ...). Each emitted npz matches the AE contract
    `load_real_marginal` requires (`pi`/`mu_2t`/`sigma_2t`/`latlons`); each meta
    JSON carries `lead_hours`, `valid_datetime`, the init time, and `from_run_id`
    so downstream stages can label the regime/lead. When `prefix` is None it is
    derived from the horizon (e.g. `phase_4_fc48` for a 48 h max lead). Returns
    the list of written npz `Path`s.
    """

    if "steps" not in obj:
        raise KeyError("not a forecast-format dict (no top-level 'steps')")
    steps = obj["steps"]
    if not steps:
        raise ValueError("forecast dict has an empty 'steps' mapping")
    leads = [float(s.get("lead_hours", 0.0)) for s in steps.values()]
    if prefix is None:
        prefix = f"phase_4_fc{int(round(max(leads)))}"

    # Per-channel de-standardisation stats (physical = mu * std + mean) for the
    # selected channel, carried into every per-lead meta so downstream figures
    # can report lambda* in physical units (correlation length km; drift in K).
    # The .pt stores the full [n_channels] vectors at top level; we surface only
    # the channel_of_interest scalar (ch_idx) -- a torch-free numeric.
    norm_mean_channel = norm_std_channel = None
    ch_idx = obj.get("ch_idx")
    if obj.get("norm_mean") is not None and obj.get("norm_std") is not None and ch_idx is not None:
        nm = np.asarray(obj["norm_mean"], dtype=float).reshape(-1)
        ns = np.asarray(obj["norm_std"], dtype=float).reshape(-1)
        ci = int(ch_idx)
        if 0 <= ci < nm.size and 0 <= ci < ns.size:
            norm_mean_channel = float(nm[ci])
            norm_std_channel = float(ns[ci])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    incoming_run_id = obj.get("from_run_id")
    incoming_init = obj.get("target_datetime")
    written = []
    for k in sorted(steps, key=lambda kk: int(kk)):
        step = steps[k]
        stem = f"{prefix}_step{int(k)}_2t"

        # Provenance guard: the runner defaults to a fixed prefix
        # (DEFAULT_PREFIX) regardless of --forecast-pt, so converting a second
        # run into the same prefix would silently overwrite the first run's
        # per-lead npz. Key the guard on the (from_run_id, init_datetime) PAIR,
        # not from_run_id alone: the multi-init replicates of one trained model
        # (F1 track-2) share from_run_id and differ only by init_datetime, so a
        # run-id-only guard would not catch an init-vs-init clobber. Refuse to
        # overwrite an existing lead whose meta carries a different pair;
        # re-converting the same run+init is allowed (idempotent overwrite).
        # Use a distinct --prefix per init.
        existing_meta = out_dir / f"{stem}_meta.json"
        if existing_meta.exists():
            try:
                prior = json.loads(existing_meta.read_text())
                prior_run_id = prior.get("from_run_id")
                prior_init = prior.get("init_datetime")
            except (ValueError, OSError):
                prior_run_id = prior_init = None
            if (prior_run_id, prior_init) != (incoming_run_id, incoming_init):
                raise FileExistsError(
                    f"{existing_meta.name} already exists from a different run "
                    f"(from_run_id={prior_run_id!r}, init_datetime={prior_init!r}); "
                    f"refusing to overwrite with from_run_id={incoming_run_id!r}, "
                    f"init_datetime={incoming_init!r}. Pass a distinct --prefix per "
                    f"init (current prefix={prefix!r})."
                )

        arrays = forecast_step_to_marginal(step)
        lead = step.get("lead_hours")
        meta = {
            "regime": "forecast",
            "forecast_step": int(k),
            "lead_hours": float(lead) if lead is not None else None,
            "valid_datetime": step.get("valid_datetime"),
            "init_datetime": obj.get("target_datetime"),
            "from_run_id": obj.get("from_run_id"),
            "mini_epoch": obj.get("mini_epoch"),
            "channel_of_interest": obj.get("channel_of_interest"),
            "ch_idx": obj.get("ch_idx"),
            "forecast_steps": obj.get("forecast_steps"),
            "time_step": obj.get("time_step"),
            "norm_mean_channel": norm_mean_channel,
            "norm_std_channel": norm_std_channel,
            "norm_stats_source": obj.get("norm_stats_source"),
        }
        np.savez_compressed(out_dir / f"{stem}.npz", **arrays)
        (out_dir / f"{stem}_meta.json").write_text(
            json.dumps(meta, indent=2, default=str, sort_keys=True) + "\n"
        )
        written.append(out_dir / f"{stem}.npz")
    return written


def validate_forecast_dict(obj, *, expected_n_steps=None, expected_ch_idx=None):
    """Structural pre-flight check of a forecast-format GMM dict (plan Step 0).

    Torch-free: arrays may be numpy or torch tensors (read via ``np.asarray``),
    so the same checker runs in the research venv on a converted artifact and in
    the WeatherGenerator venv on a freshly-loaded ``.pt``. Returns
    ``(report, problems)`` where ``report`` is a JSON-able summary
    (steps/leads/shapes/init/provenance/norm-stats presence) and ``problems`` is
    a list of human-readable strings; an empty ``problems`` list means the
    artifact matches the forecast contract ``load_real_marginal`` downstream
    requires. Never raises on a malformed artifact -- it reports.

    Checks: top-level ``steps`` present and non-empty; step keys are a
    contiguous ``1..N`` run; every step carries ``pi``/``mu_channel``/
    ``sigma_channel``/``latlons`` with a consistent ``[N, K]`` / ``[N, 2]``
    shape; per-step ``pi`` rows sum to 1, ``sigma_channel > 0``, all finite;
    ``lead_hours`` increases monotonically; ``norm_mean``/``norm_std`` present
    and long enough to index ``ch_idx``. Optional ``expected_n_steps`` /
    ``expected_ch_idx`` add equality assertions.
    """

    problems = []
    report = {}
    if not isinstance(obj, dict):
        return {"top_level_type": type(obj).__name__}, ["not a dict"]
    report["top_level_keys"] = sorted(obj.keys())
    if "steps" not in obj:
        return report, ["not a forecast-format dict (no top-level 'steps')"]
    steps = obj["steps"]
    report["init_datetime"] = obj.get("target_datetime")
    report["from_run_id"] = obj.get("from_run_id")
    report["mini_epoch"] = obj.get("mini_epoch")
    report["channel_of_interest"] = obj.get("channel_of_interest")
    report["ch_idx"] = obj.get("ch_idx")
    report["forecast_steps"] = obj.get("forecast_steps")
    report["time_step"] = obj.get("time_step")
    if not steps:
        return report, ["forecast dict has an empty 'steps' mapping"]

    try:
        step_keys = sorted(steps, key=lambda kk: int(kk))
    except (TypeError, ValueError):
        return report, [f"step keys are not integer-like: {list(steps)!r}"]
    int_keys = [int(k) for k in step_keys]
    report["n_steps"] = len(int_keys)
    report["step_keys"] = int_keys
    if int_keys != list(range(1, len(int_keys) + 1)):
        problems.append(f"step keys are not a contiguous 1..N run: {int_keys}")
    if expected_n_steps is not None and len(int_keys) != expected_n_steps:
        problems.append(f"expected {expected_n_steps} steps, found {len(int_keys)}")

    n0 = k0 = None
    leads = []
    per_step = []
    for key in step_keys:
        step = steps[key]
        missing = [kk for kk in FORECAST_STEP_KEYS if kk not in step]
        if missing:
            problems.append(f"step {key}: missing keys {missing}")
            continue
        pi = np.asarray(step["pi"], dtype=np.float64)
        mu = np.asarray(step["mu_channel"], dtype=np.float64)
        sigma = np.asarray(step["sigma_channel"], dtype=np.float64)
        latlons = np.asarray(step["latlons"], dtype=np.float64)
        lead = step.get("lead_hours")
        leads.append(None if lead is None else float(lead))
        per_step.append(
            {"step": int(key), "lead_hours": leads[-1], "n": int(pi.shape[0]),
             "k": int(pi.shape[-1]) if pi.ndim == 2 else None}
        )
        if pi.ndim != 2 or mu.shape != pi.shape or sigma.shape != pi.shape:
            problems.append(f"step {key}: pi/mu/sigma not matching [N, K] ({pi.shape}/{mu.shape}/{sigma.shape})")
            continue
        if latlons.shape != (pi.shape[0], 2):
            problems.append(f"step {key}: latlons shape {latlons.shape} != [{pi.shape[0]}, 2]")
        if n0 is None:
            n0, k0 = pi.shape
        elif pi.shape != (n0, k0):
            problems.append(f"step {key}: shape {pi.shape} differs from step-1 [{n0}, {k0}] (breaks shared graph)")
        for nm, arr in (("pi", pi), ("mu_channel", mu), ("sigma_channel", sigma), ("latlons", latlons)):
            if not np.all(np.isfinite(arr)):
                problems.append(f"step {key}: {nm} has non-finite entries")
        if not np.allclose(pi.sum(axis=-1), 1.0, atol=1e-5):
            problems.append(f"step {key}: pi rows do not sum to 1 (max dev {np.abs(pi.sum(-1) - 1).max():.2e})")
        if np.any(sigma <= 0.0):
            problems.append(f"step {key}: sigma_channel has non-positive entries")
    report["n_cells"] = n0
    report["n_components"] = k0
    report["lead_hours"] = leads
    report["per_step"] = per_step
    finite_leads = [v for v in leads if v is not None]
    if len(finite_leads) == len(leads) and any(
        b <= a for a, b in zip(finite_leads, finite_leads[1:])
    ):
        problems.append(f"lead_hours not strictly increasing: {leads}")

    nm = obj.get("norm_mean")
    ns = obj.get("norm_std")
    report["has_norm_stats"] = nm is not None and ns is not None
    if not report["has_norm_stats"]:
        problems.append("norm_mean/norm_std missing (physical-units panels unavailable)")
    else:
        nm = np.asarray(nm, dtype=float).reshape(-1)
        ns = np.asarray(ns, dtype=float).reshape(-1)
        ci = obj.get("ch_idx")
        if ci is not None and 0 <= int(ci) < nm.size and 0 <= int(ci) < ns.size:
            report["norm_mean_channel"] = float(nm[int(ci)])
            report["norm_std_channel"] = float(ns[int(ci)])
        else:
            problems.append(f"ch_idx {ci!r} out of range for norm stats (len {nm.size})")
    if expected_ch_idx is not None and obj.get("ch_idx") != expected_ch_idx:
        problems.append(f"expected ch_idx {expected_ch_idx}, found {obj.get('ch_idx')!r}")

    return report, problems


def load_real_checkpoint(path):
    """Load a real GMM checkpoint without importing torch at module import time."""

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Loading `.pt` real-output checkpoints requires an environment with torch installed."
        ) from exc

    try:
        checkpoint = torch.load(str(Path(path)), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(Path(path)), map_location="cpu")
    return dict(checkpoint)


def real_checkpoint_schema(checkpoint):
    """Return a compact `(key, shape)` schema for display/reporting."""

    schema = []
    for key, value in checkpoint.items():
        if hasattr(value, "shape"):
            shape = tuple(value.shape)
        else:
            shape = f"({type(value).__name__})"
        schema.append((key, shape))
    return schema
