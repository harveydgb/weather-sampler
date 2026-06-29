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


def _match_permutation(dataset, sample_latlons):
    """Permutation ``perm`` such that ERA5 row ``perm[i]`` is sample row ``i``.

    The ERA5 stream is the SAME O96 grid as the samples but stored in a
    different point order (anemoi ring order vs the decoder/token order of the
    extracted sample npz), so a raw row-for-row comparison sees the full grid
    span. We recover the exact relabelling by sorting both grids ring-by-ring
    (snap each array's latitudes to its own 192 Gaussian-ring values, then sort
    by longitude within the ring) and pairing the sorted sequences.

    This is a pure REORDER of identical points -- NO regrid, NO interpolation.
    The strict same-grid gate is preserved: raises ``Era5ReferenceUnavailable``
    if the point counts differ, if any matched pair is farther apart than
    ``LATLON_MATCH_TOL_DEG`` (=> genuinely different grid, cut criterion C4), or
    if the matching is not a bijection.
    """
    elat = np.asarray(dataset.latitudes, dtype=float).reshape(-1)
    elon = np.mod(np.asarray(dataset.longitudes, dtype=float).reshape(-1), 360.0)
    sample = np.asarray(sample_latlons, dtype=float)
    if elat.shape[0] != sample.shape[0]:
        raise Era5ReferenceUnavailable(
            f"ERA5 grid has {elat.shape[0]} points, samples have {sample.shape[0]}"
        )
    slat = sample[:, 0]
    slon = np.mod(sample[:, 1], 360.0)

    def _key_order(lat, lon):
        # Order by (lat, lon) on coarse keys: rounding to 1e-4 deg absorbs the
        # float32<->float64 storage jitter (grid spacing is ~0.9 deg, so distinct
        # points never share a key), and folding lon mod 360 after rounding keeps
        # a 0 deg point that jittered to ~359.9999 next to 0, not at the ring end.
        klat = np.round(lat, 4)
        klon = np.mod(np.round(np.mod(lon, 360.0), 4), 360.0)
        return np.lexsort((klon, klat))

    order_e = _key_order(elat, elon)
    order_s = _key_order(slat, slon)
    inv_s = np.empty_like(order_s)
    inv_s[order_s] = np.arange(order_s.size)
    perm = order_e[inv_s]

    dlat = np.max(np.abs(elat[perm] - slat))
    dlon = np.max(np.abs((elon[perm] - slon + 180.0) % 360.0 - 180.0))
    if dlat > LATLON_MATCH_TOL_DEG or dlon > LATLON_MATCH_TOL_DEG:
        raise Era5ReferenceUnavailable(
            f"ERA5 grid is not the sample O96 grid after ring alignment "
            f"(max dlat={dlat:.3g}, dlon={dlon:.3g} deg) -- NO regrid is performed"
        )
    if not np.array_equal(np.sort(perm), np.arange(perm.size)):
        raise Era5ReferenceUnavailable(
            "ERA5<->sample point matching is not a bijection (duplicate grid points?)"
        )
    return perm


def load_era5_2t_on_o96(valid_datetime, sample_latlons, *, channel="2t",
                        standardise="own", zarr_path=None, meta_dir=None,
                        norm_mean=None, norm_std=None):
    """Read ERA5 2t for one valid datetime on the native O96 grid.

    HARD GATE: the zarr must open, ``valid_datetime`` must be present on the 6h
    time axis, and the returned latlons must match ``sample_latlons`` row-for-row
    (same O96 ordering -- NO regrid). Raises ``Era5ReferenceUnavailable`` on any
    failure (caller skips the ERA5 series; cut criterion C4).

    ``standardise='own'`` subtracts the spatial mean and divides by the spatial
    std (shape-only comparison). ``standardise='stream'`` uses stream stats from
    the anemoi dataset, if exposed. ``standardise='meta'`` uses the forecast
    extraction metadata normalisation (``norm_mean_channel``/``norm_std_channel``)
    so the ERA5 map shares the same colour scale as decoder-normalised fields.
    If ``meta_dir`` is given, an ``era5_ref_meta.json`` provenance sidecar is
    written there.
    """
    if standardise not in ("own", "stream", "meta"):
        raise ValueError("standardise must be 'own', 'stream' or 'meta'")

    zarr_path = resolve_zarr_path(zarr_path)
    dataset = _open_anemoi(zarr_path)
    t = _time_index(dataset, valid_datetime)
    v = _channel_index(dataset, channel)
    # Same O96 grid, possibly a different point order -> exact reorder (no regrid).
    perm = _match_permutation(dataset, sample_latlons)

    # anemoi datasets index as ds[time] -> [variable, ensemble, cell]; take the
    # first ensemble member. Squeeze defensively for layout variants.
    arr = np.asarray(dataset[t])
    field = np.asarray(arr[v]).reshape(arr.shape[-1] if arr.ndim >= 1 else -1)
    field = np.asarray(field, dtype=float).reshape(-1)
    if field.shape[0] != np.asarray(sample_latlons).shape[0]:
        # Fall back through an explicit [var, ens, cell] view.
        arr3 = arr.reshape(len(dataset.variables), -1, np.asarray(sample_latlons).shape[0])
        field = np.asarray(arr3[v, 0], dtype=float).reshape(-1)
    # Relabel ERA5's native ring order into the sample's row order.
    field = field[perm]

    if standardise == "own":
        mean = float(np.mean(field))
        std = float(np.std(field))
        if std <= 0:
            raise Era5ReferenceUnavailable("ERA5 field has zero spatial variance")
        z = (field - mean) / std
    elif standardise == "stream":  # only if trivially recoverable
        stats = getattr(dataset, "statistics", None)
        if not stats or "mean" not in stats or "stdev" not in stats:
            raise Era5ReferenceUnavailable("stream normalisation stats unavailable")
        z = (field - float(stats["mean"][v])) / float(stats["stdev"][v])
    else:
        if norm_mean is None or norm_std is None:
            raise Era5ReferenceUnavailable(
                "standardise='meta' requires norm_mean and norm_std"
            )
        norm_std = float(norm_std)
        if norm_std <= 0:
            raise Era5ReferenceUnavailable("metadata norm_std must be positive")
        z = (field - float(norm_mean)) / norm_std

    if meta_dir is not None:
        _write_meta(meta_dir, zarr_path, valid_datetime, channel, standardise,
                    sample_latlons, norm_mean=norm_mean, norm_std=norm_std)
    return z


def _write_meta(meta_dir, zarr_path, valid_datetime, channel, standardise, sample_latlons,
                *, norm_mean=None, norm_std=None):
    ll = np.ascontiguousarray(np.asarray(sample_latlons, dtype=np.float64))
    latlon_hash = hashlib.sha256(ll.tobytes()).hexdigest()
    meta = {
        "zarr_path": str(zarr_path),
        "valid_datetime": str(valid_datetime),
        "channel": channel,
        "standardise": standardise,
        "norm_mean": None if norm_mean is None else float(norm_mean),
        "norm_std": None if norm_std is None else float(norm_std),
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
    p.add_argument("--standardise", default="own", choices=("own", "stream", "meta"))
    p.add_argument("--norm-meta-json", default=None,
                   help="metadata JSON carrying norm_mean_channel/norm_std_channel")
    p.add_argument("--out-npz", default=None,
                   help="optional output npz path; writes array as key 'era5'")
    p.add_argument("--zarr-path", default=None)
    p.add_argument("--meta-dir", default=None)
    args = p.parse_args()

    with np.load(args.latlons_npz) as f:
        latlons = np.asarray(f["latlons"], dtype=float)
    norm_mean = norm_std = None
    if args.norm_meta_json is not None:
        meta = json.loads(Path(args.norm_meta_json).read_text())
        norm_mean = meta.get("norm_mean_channel")
        norm_std = meta.get("norm_std_channel")
    try:
        z = load_era5_2t_on_o96(
            args.valid_datetime, latlons, channel=args.channel,
            standardise=args.standardise, zarr_path=args.zarr_path,
            meta_dir=args.meta_dir, norm_mean=norm_mean, norm_std=norm_std,
        )
    except Era5ReferenceUnavailable as exc:
        raise SystemExit(f"ERA5 reference unavailable: {exc}")
    if args.out_npz is not None:
        out = Path(args.out_npz)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, era5=z)
    print(f"loaded ERA5 {args.channel} @ {args.valid_datetime}: "
          f"N={z.shape[0]}, mean={z.mean():.3g}, std={z.std():.3g}")


if __name__ == "__main__":
    main()
