"""ERA5 2t reference on the native O96 grid (spectrum_era5_plan.md T3).

A small, dependency-isolated loader for the ERA5 *direction-of-realism* reference
drawn on the spectrum and the variogram. This is kept as a SCRIPT (not in the
importable package) so the anemoi/zarr dependency never enters
``sampler_research``; the Phase-4 runner calls it through a thin wrapper that
swallows every failure (cut criterion C4 -> spectrum ships sample-only).

RUNG-3, never a target, never validation: ERA5 only orients the *direction* of
Method 1's high-l correction among siblings that share the decoder mean. Low-l
offsets vs ERA5 are common-mode decoder error (debug-scale model + token-aligned
artefact, audit S4). See the plan S0/S2/S8.

The ERA5 stream is the same O96 zarr the decoder trained on
(``aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8.zarr``, ``type: anemoi``,
6h cadence, 1979-2023), so NO regrid is needed: the loader asserts the zarr
latlons match the sample latlons row-for-row.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

#: The era5_1deg stream filename (WeatherGenerator config/streams/era5_1deg/era5.yml).
ERA5_O96_ZARR_NAME = "aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8.zarr"
#: Latlon row-match tolerance (deg). The grids are byte-identical in principle;
#: this only absorbs float32 vs float64 storage jitter.
LATLON_MATCH_TOL_DEG = 1e-3


class Era5ReferenceUnavailable(RuntimeError):
    """Raised when any hard gate fails; the caller skips the ERA5 series."""


def resolve_zarr_path(zarr_path=None):
    """Resolve the ERA5 O96 zarr path.

    Order: explicit ``zarr_path`` -> ``$ERA5_O96_ZARR`` (direct override) ->
    the WeatherGenerator private conf (``$WEATHERGENERATOR_PRIVATE_CONF``), whose
    data root is joined with the known stream filename. Raises
    ``Era5ReferenceUnavailable`` if nothing resolves to an existing path.
    """
    import os

    candidates = []
    if zarr_path is not None:
        candidates.append(Path(zarr_path).expanduser())
    env_direct = os.environ.get("ERA5_O96_ZARR")
    if env_direct:
        candidates.append(Path(env_direct).expanduser())

    conf_path = os.environ.get("WEATHERGENERATOR_PRIVATE_CONF")
    if conf_path:
        conf_file = Path(conf_path).expanduser()
        if not conf_file.exists():
            raise Era5ReferenceUnavailable(
                f"WEATHERGENERATOR_PRIVATE_CONF points at a missing file: {conf_file}"
            )
        root = _data_root_from_private_conf(conf_file)
        if root is not None:
            candidates.append(Path(root).expanduser() / ERA5_O96_ZARR_NAME)

    for cand in candidates:
        if cand.exists():
            return cand
    if not candidates:
        raise Era5ReferenceUnavailable(
            "no ERA5 zarr path given and neither ERA5_O96_ZARR nor "
            "WEATHERGENERATOR_PRIVATE_CONF is set"
        )
    tried = ", ".join(str(c) for c in candidates)
    raise Era5ReferenceUnavailable(f"no existing ERA5 zarr among: {tried}")


def _data_root_from_private_conf(conf_file):
    """Best-effort extraction of a data root from the private conf YAML.

    The conf schema is private and may evolve; we look for common keys
    (``data_root`` / ``dataset_root`` / ``zarr_root`` / ``root``) at the top level
    or under a ``paths``/``data`` block. Returns None if nothing plausible found.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise Era5ReferenceUnavailable("pyyaml not available to read private conf") from exc
    try:
        conf = yaml.safe_load(conf_file.read_text()) or {}
    except yaml.YAMLError as exc:
        raise Era5ReferenceUnavailable(f"could not parse private conf: {exc}") from exc

    keys = ("data_root", "dataset_root", "zarr_root", "datasets_root", "root")
    blocks = [conf]
    for block_key in ("paths", "data", "datasets"):
        if isinstance(conf.get(block_key), dict):
            blocks.append(conf[block_key])
    for block in blocks:
        if not isinstance(block, dict):
            continue
        for k in keys:
            if isinstance(block.get(k), str):
                return block[k]
    return None


def _open_anemoi(zarr_path):
    """Open the anemoi dataset; raise Era5ReferenceUnavailable on import failure."""
    try:
        from anemoi.datasets import open_dataset
    except ImportError as exc:
        raise Era5ReferenceUnavailable(
            "anemoi.datasets not importable in this environment "
            "(ERA5 reference needs the WeatherGenerator stack)"
        ) from exc
    try:
        return open_dataset(str(zarr_path))
    except Exception as exc:  # noqa: BLE001 - any open failure -> skip
        raise Era5ReferenceUnavailable(f"could not open ERA5 zarr {zarr_path}: {exc}") from exc


def _time_index(dataset, valid_datetime):
    """Index of ``valid_datetime`` on the dataset's (6h) time axis, or raise."""
    want = np.datetime64(np.datetime64(valid_datetime), "h")
    dates = np.asarray(dataset.dates).astype("datetime64[h]")
    matches = np.nonzero(dates == want)[0]
    if matches.size == 0:
        raise Era5ReferenceUnavailable(
            f"valid_datetime {valid_datetime} not on the ERA5 6h time axis "
            f"(span {dates[0]}..{dates[-1]})"
        )
    return int(matches[0])


def _channel_index(dataset, channel):
    variables = list(getattr(dataset, "variables", []) or [])
    if channel not in variables:
        raise Era5ReferenceUnavailable(
            f"channel {channel!r} not in ERA5 variables ({variables[:8]}...)"
        )
    return variables.index(channel)


def _assert_latlons_match(dataset, sample_latlons):
    lat = np.asarray(dataset.latitudes, dtype=float).reshape(-1)
    lon = np.mod(np.asarray(dataset.longitudes, dtype=float).reshape(-1), 360.0)
    sample = np.asarray(sample_latlons, dtype=float)
    if lat.shape[0] != sample.shape[0]:
        raise Era5ReferenceUnavailable(
            f"ERA5 grid has {lat.shape[0]} points, samples have {sample.shape[0]}"
        )
    slat = sample[:, 0]
    slon = np.mod(sample[:, 1], 360.0)
    dlat = np.max(np.abs(lat - slat))
    dlon = np.max(np.abs((lon - slon + 180.0) % 360.0 - 180.0))
    if dlat > LATLON_MATCH_TOL_DEG or dlon > LATLON_MATCH_TOL_DEG:
        raise Era5ReferenceUnavailable(
            f"ERA5 latlons do not match sample latlons row-for-row "
            f"(max dlat={dlat:.3g}, dlon={dlon:.3g} deg) -- NO regrid is performed"
        )


def load_era5_2t_on_o96(valid_datetime, sample_latlons, *, channel="2t",
                        standardise="own", zarr_path=None, meta_dir=None):
    """Read ERA5 2t for one valid datetime on the native O96 grid.

    HARD GATE: the zarr must open, ``valid_datetime`` must be present on the 6h
    time axis, and the returned latlons must match ``sample_latlons`` row-for-row
    (same O96 ordering -- NO regrid). Raises ``Era5ReferenceUnavailable`` on any
    failure (caller skips the ERA5 series; cut criterion C4).

    ``standardise='own'`` subtracts the spatial mean and divides by the spatial
    std (shape-only comparison; amplitude is confounded by missing
    de-standardisation, audit S1). The monopole is dropped in the transform, so
    the offset is irrelevant either way. If ``meta_dir`` is given, an
    ``era5_ref_meta.json`` provenance sidecar is written there.
    """
    if standardise not in ("own", "stream"):
        raise ValueError("standardise must be 'own' or 'stream'")

    zarr_path = resolve_zarr_path(zarr_path)
    dataset = _open_anemoi(zarr_path)
    t = _time_index(dataset, valid_datetime)
    v = _channel_index(dataset, channel)
    _assert_latlons_match(dataset, sample_latlons)

    # anemoi datasets index as ds[time] -> [variable, ensemble, cell]; take the
    # first ensemble member. Squeeze defensively for layout variants.
    arr = np.asarray(dataset[t])
    field = np.asarray(arr[v]).reshape(arr.shape[-1] if arr.ndim >= 1 else -1)
    field = np.asarray(field, dtype=float).reshape(-1)
    if field.shape[0] != np.asarray(sample_latlons).shape[0]:
        # Fall back through an explicit [var, ens, cell] view.
        arr3 = arr.reshape(len(dataset.variables), -1, np.asarray(sample_latlons).shape[0])
        field = np.asarray(arr3[v, 0], dtype=float).reshape(-1)

    if standardise == "own":
        mean = float(np.mean(field))
        std = float(np.std(field))
        if std <= 0:
            raise Era5ReferenceUnavailable("ERA5 field has zero spatial variance")
        z = (field - mean) / std
    else:  # 'stream' normalisation -- only if trivially recoverable
        stats = getattr(dataset, "statistics", None)
        if not stats or "mean" not in stats or "stdev" not in stats:
            raise Era5ReferenceUnavailable("stream normalisation stats unavailable")
        z = (field - float(stats["mean"][v])) / float(stats["stdev"][v])

    if meta_dir is not None:
        _write_meta(meta_dir, zarr_path, valid_datetime, channel, standardise,
                    sample_latlons)
    return z


def _write_meta(meta_dir, zarr_path, valid_datetime, channel, standardise, sample_latlons):
    ll = np.ascontiguousarray(np.asarray(sample_latlons, dtype=np.float64))
    latlon_hash = hashlib.sha256(ll.tobytes()).hexdigest()
    meta = {
        "zarr_path": str(zarr_path),
        "valid_datetime": str(valid_datetime),
        "channel": channel,
        "standardise": standardise,
        "n_points": int(ll.shape[0]),
        "latlon_sha256": latlon_hash,
        "note": "rung-3 direction-of-realism reference, NOT a target (plan S0/S8)",
    }
    out = Path(meta_dir) / "era5_ref_meta.json"
    out.write_text(json.dumps(meta, indent=2) + "\n")


def main():  # pragma: no cover - manual smoke entry point
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--valid-datetime", required=True)
    p.add_argument("--latlons-npz", required=True,
                   help="npz carrying a 'latlons' [N,2] array (the sample grid)")
    p.add_argument("--channel", default="2t")
    p.add_argument("--standardise", default="own", choices=("own", "stream"))
    p.add_argument("--zarr-path", default=None)
    p.add_argument("--meta-dir", default=None)
    args = p.parse_args()

    with np.load(args.latlons_npz) as f:
        latlons = np.asarray(f["latlons"], dtype=float)
    try:
        z = load_era5_2t_on_o96(
            args.valid_datetime, latlons, channel=args.channel,
            standardise=args.standardise, zarr_path=args.zarr_path,
            meta_dir=args.meta_dir,
        )
    except Era5ReferenceUnavailable as exc:
        raise SystemExit(f"ERA5 reference unavailable: {exc}")
    print(f"loaded ERA5 {args.channel} @ {args.valid_datetime}: "
          f"N={z.shape[0]}, mean={z.mean():.3g}, std={z.std():.3g}")


if __name__ == "__main__":
    main()
