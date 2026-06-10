"""Convert torch-saved GMM parameter dicts to .npz + JSON metadata.

The research venv has no torch; run this with the WeatherGenerator venv:

    ~/WeatherGenerator/.venv/bin/python scripts/convert_real_gmm_pt_to_npz.py

All tensors / numeric containers (recursively flattened with "/"-joined key
paths) are written to the .npz; everything else (strings, datetimes, config
dicts) goes to a JSON sidecar so nothing in the .pt is silently dropped.
Downstream audit / sampler code uses only numpy + the research venv.
"""

import json
from pathlib import Path

import numpy as np
import torch

SRC = Path("/users/harvey_bermingham/model_outputs")
DST = Path("/users/harvey_bermingham/weather-sampler-research/outputs/data")

FILES = {
    "gmm_params_gmm_era5_32ep_v3_me31_2t.pt": "phase_4_real_2t",
    "gmm_params_gmm_era5_32ep_v3_me31_2t.iidsource.pt": "phase_4_real_2t_iidsource",
}


def flatten(prefix: str, value, arrays: dict, meta: dict) -> None:
    """Recursively split a value into npz-able arrays vs JSON metadata."""
    if isinstance(value, torch.Tensor):
        arrays[prefix] = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        arrays[prefix] = value
    elif isinstance(value, dict):
        for k, v in value.items():
            flatten(f"{prefix}/{k}" if prefix else str(k), v, arrays, meta)
    elif (
        isinstance(value, (list, tuple))
        and len(value) > 0
        and all(isinstance(x, (int, float, np.integer, np.floating)) for x in value)
    ):
        arrays[prefix] = np.asarray(value)
    else:
        meta[prefix] = value


def convert(src: Path, stem: str) -> None:
    obj = torch.load(src, map_location="cpu", weights_only=False)
    print(f"\n=== {src.name} ===")
    print(f"top-level type: {type(obj).__name__}")
    if not isinstance(obj, dict):
        raise TypeError(f"expected dict, got {type(obj)}")
    print(f"top-level keys ({len(obj)}): {list(obj.keys())}")

    arrays: dict = {}
    meta: dict = {}
    for k, v in obj.items():
        flatten(str(k), v, arrays, meta)

    for k in sorted(arrays):
        a = arrays[k]
        print(f"  [array] {k:28s} shape={tuple(a.shape)!s:18s} dtype={a.dtype}")
    for k in sorted(meta):
        print(f"  [meta ] {k:28s} {type(meta[k]).__name__}: {meta[k]!r}")

    DST.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DST / f"{stem}.npz", **arrays)
    with open(DST / f"{stem}_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str, sort_keys=True)
    print(f"  -> {stem}.npz ({len(arrays)} arrays) + {stem}_meta.json ({len(meta)} meta keys)")


if __name__ == "__main__":
    for name, stem in FILES.items():
        convert(SRC / name, stem)
