"""Input/output helpers for sampler research data."""

from pathlib import Path

import numpy as np

SAMPLER_KEYS = ("pi", "mu", "sigma", "coords")


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
