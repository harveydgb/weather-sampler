"""Input/output helpers for sampler research data."""

from pathlib import Path

import numpy as np

SAMPLER_KEYS = ("pi", "mu", "sigma", "coords")
REAL_MARGINAL_KEYS = ("pi", "mu_2t", "sigma_2t", "latlons")


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
