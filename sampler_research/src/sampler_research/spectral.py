"""Power-spectrum diagnostics for produced 2D toy fields.

These are secondary roughness diagnostics, not optimisation objectives. They
check whether a field places too much variance at high spatial frequencies and
whether the radially averaged spectrum has a roughly decreasing shape.
"""

import numpy as np


def _as_2d_field(field):
    arr = np.asarray(field, dtype=float)
    if arr.ndim != 2:
        raise ValueError("spectral diagnostics require a 2D field")
    return arr


def _fft_power_and_radius(field):
    """Return non-shifted FFT power and normalised radial frequency grid."""

    arr = _as_2d_field(field)
    height, width = arr.shape
    demeaned = arr - arr.mean()
    power = np.abs(np.fft.fft2(demeaned)) ** 2 / arr.size

    ky = np.fft.fftfreq(height)
    kx = np.fft.fftfreq(width)
    radius = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    return power, radius


def radial_power_spectrum(field, n_bins=None):
    """Radially average the 2D periodogram of a demeaned field.

    Returns ``(bin_centres, mean_power, counts)``. The zero-frequency/DC bin is
    excluded because the field mean is not a roughness signal. Empty bins have
    ``nan`` mean power and zero count.
    """

    arr = _as_2d_field(field)
    power, radius = _fft_power_and_radius(arr)
    non_dc = radius > 0.0

    if n_bins is None:
        n_bins = max(2, min(arr.shape) // 2)
    if n_bins < 1:
        raise ValueError("n_bins must be positive")

    max_radius = float(radius[non_dc].max())
    edges = np.linspace(0.0, max_radius + np.finfo(float).eps, n_bins + 1)
    bin_ids = np.clip(np.digitize(radius[non_dc], edges) - 1, 0, n_bins - 1)

    counts = np.bincount(bin_ids, minlength=n_bins).astype(int)
    sums = np.bincount(bin_ids, weights=power[non_dc], minlength=n_bins)
    mean_power = np.full(n_bins, np.nan, dtype=float)
    np.divide(sums, counts, out=mean_power, where=counts > 0)

    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, mean_power, counts


def spectral_roughness(
    field,
    *,
    n_bins=None,
    high_freq_radius_fraction=0.5,
    var_floor=1e-12,
):
    """Return secondary power-spectrum roughness diagnostics.

    ``spectral_hf_ratio`` is the fraction of non-DC spectral power at radial
    frequencies above ``high_freq_radius_fraction`` of the maximum available
    frequency. It is amplitude-scale invariant for non-constant fields, so it
    does not reward a sampler merely for shrinking the field toward its mean.

    ``spectral_slope`` is the log-log slope of the radially averaged spectrum;
    negative values indicate decreasing power with frequency. On the 8x8 toy
    this is a coarse shape sanity check, not a fitted physical law.

    ``spectral_monotone_fraction`` is the fraction of adjacent radial bins whose
    mean power does not increase. It is a necessary-condition style check: high
    values are plausible, but not sufficient evidence of spatial coherence.
    """

    if not (0.0 < high_freq_radius_fraction <= 1.0):
        raise ValueError("high_freq_radius_fraction must satisfy 0 < value <= 1")

    arr = _as_2d_field(field)
    centres, mean_power, _ = radial_power_spectrum(arr, n_bins=n_bins)
    power, radius = _fft_power_and_radius(arr)

    non_dc = radius > 0.0
    total_power = float(power[non_dc].sum())
    collapsed = total_power <= var_floor

    if collapsed:
        return {
            "spectral_hf_ratio": 0.0,
            "spectral_slope": np.nan,
            "spectral_monotone_fraction": np.nan,
            "spectral_collapsed": True,
        }

    high_cut = high_freq_radius_fraction * float(radius[non_dc].max())
    high_power = float(power[non_dc & (radius >= high_cut)].sum())
    hf_ratio = high_power / total_power

    valid = np.isfinite(mean_power) & (mean_power > 0.0) & (centres > 0.0)
    if np.count_nonzero(valid) >= 2:
        slope = float(np.polyfit(np.log(centres[valid]), np.log(mean_power[valid]), deg=1)[0])
        valid_power = mean_power[valid]
        monotone_fraction = float(np.mean(np.diff(valid_power) <= 0.0))
    else:
        slope = np.nan
        monotone_fraction = np.nan

    return {
        "spectral_hf_ratio": float(hf_ratio),
        "spectral_slope": slope,
        "spectral_monotone_fraction": monotone_fraction,
        "spectral_collapsed": False,
    }
