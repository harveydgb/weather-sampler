"""Convert torch-saved GMM parameter dicts to .npz + JSON metadata.

The research venv has no torch; run this with the WeatherGenerator venv:

    # legacy masking-AE (step-0) files, unchanged default behaviour:
    ~/WeatherGenerator/.venv/bin/python scripts/convert_real_gmm_pt_to_npz.py

    # forecast-format .pt (gmm_inference.py --forecast-steps N): one npz per lead
    ~/WeatherGenerator/.venv/bin/python scripts/convert_real_gmm_pt_to_npz.py \
        --input ~/model_outputs/gmm_params_..._f8.pt

Two input formats are supported and auto-detected:

* AE / step-0 format (no top-level ``steps``): every tensor / numeric container
  is recursively flattened (``/``-joined key paths) into the .npz; everything
  else (strings, datetimes, config dicts) goes to a JSON sidecar so nothing in
  the .pt is silently dropped.
* Forecast format (top-level ``steps`` dict from
  ``gmm_inference.py --forecast-steps N``): one ``{prefix}_step{k}_2t.npz`` +
  ``_meta.json`` per lead step, matching the AE contract that
  ``io.load_real_marginal`` requires (keys ``pi``/``mu_2t``/``sigma_2t``/
  ``latlons``, with ``mu_2t``<-``mu_channel`` and ``sigma_2t``<-``sigma_channel``).

Downstream audit / sampler code uses only numpy + the research venv.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Import the torch-free forecast writer from the package so it stays unit-testable
# in the research venv (this script's torch dependency does not reach it).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sampler_research" / "src"))
from sampler_research.io import write_forecast_marginals  # noqa: E402

SRC = Path("/users/harvey_bermingham/model_outputs")
DST = Path("/users/harvey_bermingham/weather-sampler-research/outputs/data")

# Legacy masking-AE inputs converted when the script is run with no --input.
LEGACY_AE_FILES = {
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


def _emit_ae(obj: dict, stem: str, out_dir: Path) -> None:
    """Flatten an AE / step-0 dict to {stem}.npz + {stem}_meta.json."""
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

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{stem}.npz", **arrays)
    with open(out_dir / f"{stem}_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str, sort_keys=True)
    print(f"  -> {stem}.npz ({len(arrays)} arrays) + {stem}_meta.json ({len(meta)} meta keys)")


def _emit_forecast(obj: dict, out_dir: Path, prefix) -> None:
    """One per-lead npz + meta from a forecast-format dict (torch-free writer)."""
    # Convert tensors to numpy up front so the package writer stays torch-free.
    for k, step in obj["steps"].items():
        for key, value in list(step.items()):
            if isinstance(value, torch.Tensor):
                step[key] = value.detach().cpu().numpy()
    # Top-level de-standardisation stats too (carried into per-lead meta).
    for key in ("norm_mean", "norm_std"):
        if isinstance(obj.get(key), torch.Tensor):
            obj[key] = obj[key].detach().cpu().numpy()
    written = write_forecast_marginals(obj, out_dir, prefix=prefix)
    print(f"  forecast: {len(written)} lead steps")
    for p in written:
        print(f"  -> {p.name} + {p.stem}_meta.json")


def convert(src: Path, out_dir: Path, fmt: str = "auto", stem=None, prefix=None) -> None:
    obj = torch.load(src, map_location="cpu", weights_only=False)
    print(f"\n=== {src.name} ===")
    if fmt == "auto":
        fmt = "forecast" if isinstance(obj, dict) and "steps" in obj else "ae"
    if fmt == "forecast":
        _emit_forecast(obj, out_dir, prefix)
    else:
        _emit_ae(obj, stem or src.stem, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, nargs="*",
                        help="input .pt path(s); if omitted, convert the two legacy AE files")
    parser.add_argument("--out-dir", type=Path, default=DST)
    parser.add_argument("--format", choices=["auto", "ae", "forecast"], default="auto")
    parser.add_argument("--prefix", default=None,
                        help="forecast npz prefix (default: derived from horizon, e.g. phase_4_fc48)")
    parser.add_argument("--stem", default=None,
                        help="AE output stem override (single-input only)")
    args = parser.parse_args()

    if not args.input:
        for name, stem in LEGACY_AE_FILES.items():
            convert(SRC / name, args.out_dir, fmt="ae", stem=stem)
        return

    if args.stem and len(args.input) > 1:
        parser.error("--stem can only be used with a single --input")
    for src in args.input:
        convert(src, args.out_dir, fmt=args.format, stem=args.stem, prefix=args.prefix)


if __name__ == "__main__":
    main()
