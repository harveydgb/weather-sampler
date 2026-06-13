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
    """Load the 2t marginal of a converted real GMM `.npz` (phase_4_data_audit §1).

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

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for k in sorted(steps, key=lambda kk: int(kk)):
        step = steps[k]
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
        }
        stem = f"{prefix}_step{int(k)}_2t"
        np.savez_compressed(out_dir / f"{stem}.npz", **arrays)
        (out_dir / f"{stem}_meta.json").write_text(
            json.dumps(meta, indent=2, default=str, sort_keys=True) + "\n"
        )
        written.append(out_dir / f"{stem}.npz")
    return written


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
