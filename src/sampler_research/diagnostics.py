"""Diagnostic calculations for Phase 1 and real-output notebooks."""

import numpy as np


def empirical_variogram(
    coords,
    fields,
    n_bins=12,
):
    """Compute a full-pair Euclidean empirical variogram for small grids."""

    flat_coords = np.asarray(coords, dtype=float).reshape(-1, coords.shape[-1])
    diff = flat_coords[:, None, :] - flat_coords[None, :, :]
    pair_dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
    iu = np.triu_indices(flat_coords.shape[0], k=1)
    pair_dist = pair_dist_matrix[iu]

    bins = np.linspace(0.0, float(pair_dist.max()), n_bins + 1)
    bin_ids = np.clip(np.digitize(pair_dist, bins) - 1, 0, n_bins - 1)
    centres = 0.5 * (bins[:-1] + bins[1:])

    variograms = {}
    for name, field in fields.items():
        flat = np.asarray(field, dtype=float).reshape(-1)
        sq_diff = (flat[:, None] - flat[None, :]) ** 2
        pair_sq = sq_diff[iu]
        variograms[name] = np.array(
            [
                0.5 * pair_sq[bin_ids == b].mean() if np.any(bin_ids == b) else np.nan
                for b in range(n_bins)
            ]
        )
    return centres, variograms


def spherical_pair_dist_deg(
    lat1_deg,
    lon1_deg,
    lat2_deg,
    lon2_deg,
):
    """Great-circle angular separation in degrees."""

    lat1 = np.radians(lat1_deg)
    lon1 = np.radians(lon1_deg)
    lat2 = np.radians(lat2_deg)
    lon2 = np.radians(lon2_deg)
    cos_d = (
        np.sin(lat1) * np.sin(lat2)
        + np.cos(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    )
    dist = np.degrees(np.arccos(np.clip(cos_d, -1.0, 1.0)))
    return np.where(np.isclose(cos_d, 1.0), 0.0, dist)


def sampled_spherical_variogram(
    latlons_deg,
    fields,
    n_pairs=60_000,
    bin_width_deg=2.0,
    seed=42,
):
    """Sample point pairs and compute angular-distance variograms for large grids."""

    latlons = np.asarray(latlons_deg, dtype=float)
    if latlons.ndim != 2 or latlons.shape[1] != 2:
        raise ValueError("latlons_deg must have shape [N, 2]")

    rng = np.random.default_rng(seed)
    n = latlons.shape[0]
    i_p = rng.integers(0, n, size=n_pairs)
    j_p = rng.integers(0, n, size=n_pairs)
    keep = i_p != j_p
    i_p = i_p[keep]
    j_p = j_p[keep]

    dist_deg = spherical_pair_dist_deg(
        latlons[i_p, 0],
        latlons[i_p, 1],
        latlons[j_p, 0],
        latlons[j_p, 1],
    )
    bins = np.arange(0.0, 180.0 + bin_width_deg, bin_width_deg)
    centres = 0.5 * (bins[:-1] + bins[1:])
    bin_ids = np.clip(np.digitize(dist_deg, bins) - 1, 0, len(centres) - 1)

    variograms = {}
    for name, field in fields.items():
        values = np.asarray(field, dtype=float).reshape(-1)
        semivariance = 0.5 * (values[i_p] - values[j_p]) ** 2
        variograms[name] = np.array(
            [
                semivariance[bin_ids == b].mean() if np.any(bin_ids == b) else np.nan
                for b in range(len(centres))
            ]
        )
    return centres, variograms, dist_deg


# --------------------------------------------------------------------------- #
# Angular power spectrum C_l on the O96 reduced-Gaussian sphere.
#
# RUNG-3 BRACKET DIAGNOSTIC ONLY. There is no GMM-derivable *target* spectrum
# (an open problem): the decoder emits independent per-location
# GMMs and discards spatial covariance by Working Decision #4, so the target
# spectrum is exactly the thrown-away information. C_l is read as a one-sided
# bracket (iid white floor above / mixture-mean over-smooth below; samplers
# between, and below iid at high l) -- never an optimisation target, never
# validation.
#
# The ducc0 engine applies the spherical-harmonic analysis on the native O96
# iso-latitude rings. We never regrid to lat-lon because a regrid would low-pass
# the high-l tail, the band of interest. The pure-numpy native engine is kept as
# an explicit cross-check, not as the default report path.
# --------------------------------------------------------------------------- #

#: Nominal angular-degree ceiling for the O96 octahedral grid (linear T_L191
#: truncation; 192 Gaussian latitudes). The *resolved* band reported by the
#: transform is trimmed below this by the Parseval check (expect ~180).
O96_LMAX_NOMINAL = 191


def _o96_ring_layout(latlons_deg):
    """Group O96 points into their 192 Gaussian-latitude rings.

    Returns a dict with, per ring (ordered north->south, i.e. descending lat):
      ``row_index``  list of int arrays indexing ``latlons_deg`` rows, each
                     sorted by ascending longitude;
      ``x``          sin(latitude) = Gauss node for the ring;
      ``w``          Gauss-Legendre weight for the ring (sums to 2 over rings);
      ``nphi``       number of longitudes on the ring.

    Asserts the ring latitudes match ``numpy.polynomial.legendre.leggauss``
    nodes to tolerance -- the exactness precondition. A non-O96 grid fails here
    loudly rather than silently corrupting the high-l tail.
    """

    latlons = np.asarray(latlons_deg, dtype=float)
    if latlons.ndim != 2 or latlons.shape[1] != 2:
        raise ValueError("latlons_deg must have shape [N, 2]")

    lat = latlons[:, 0]
    lon = np.mod(latlons[:, 1], 360.0)
    # Group by latitude ring. Round to absorb float32 jitter (audit S5: rings are
    # exact Gaussian latitudes, distinct to far better than 1e-4 deg).
    lat_key = np.round(lat, 4)
    uniq = np.unique(lat_key)
    n_rings = uniq.size

    # leggauss returns nodes/weights ascending in x; np.unique gives latitudes
    # ascending, and latitude is monotone in x (x = sin lat), so the ascending
    # rings pair one-to-one with the ascending Gauss nodes/weights.
    nodes, weights = np.polynomial.legendre.leggauss(n_rings)
    expected_lat = np.degrees(np.arcsin(nodes))  # ascending, paired to uniq
    max_err = float(np.max(np.abs(uniq - expected_lat)))
    if max_err > 1e-2:
        raise ValueError(
            "latlons do not lie on Gauss-Legendre latitudes for "
            f"{n_rings} rings (max |lat - GL node| = {max_err:.3g} deg); "
            "this transform requires an O96-style reduced-Gaussian grid"
        )

    # Build per-ring (north->south) row indices sorted by longitude, plus the
    # paired x (=sin lat) and Gauss weight.
    ring_lat = uniq[::-1]  # north -> south (descending latitude)
    lat_to_x = dict(zip(np.round(uniq, 4), nodes))
    lat_to_w = dict(zip(np.round(uniq, 4), weights))
    row_index = []
    x = np.empty(n_rings)
    w = np.empty(n_rings)
    nphi = np.empty(n_rings, dtype=int)
    for r, lat_r in enumerate(ring_lat):
        sel = np.nonzero(lat_key == lat_r)[0]
        sel = sel[np.argsort(lon[sel])]
        row_index.append(sel)
        x[r] = lat_to_x[np.round(lat_r, 4)]
        w[r] = lat_to_w[np.round(lat_r, 4)]
        nphi[r] = sel.size
    return {
        "row_index": row_index,
        "x": x,
        "w": w,
        "nphi": nphi,
        "lon": lon,
        "n_rings": n_rings,
    }


def _ring_longitude_fft(values, layout, mmax):
    """Per-ring longitude DFT g_m(theta_j), m = 0..mmax (exact on uniform rings).

    Returns the complex array ``G`` of shape ``[n_rings, mmax+1]`` with
    ``G[j, m] = sum_p f(theta_j, phi_p) exp(-i m phi_p)`` (NOT divided by nphi --
    the 2*pi/nphi longitude-quadrature factor is applied in the projection).
    Wavenumbers above a ring's Nyquist (m > nphi//2) are exact zeros.
    """

    values = np.asarray(values, dtype=float).reshape(-1)
    lon_rad = np.radians(layout["lon"])
    n_rings = layout["n_rings"]
    G = np.zeros((n_rings, mmax + 1), dtype=complex)
    m_axis = np.arange(mmax + 1)
    for j in range(n_rings):
        idx = layout["row_index"][j]
        f_ring = values[idx]
        phi = lon_rad[idx]
        ring_nyq = min(mmax, idx.size // 2)
        # Direct DFT to the common m-grid (idx.size <= 400, mmax <= 191 -> cheap
        # and avoids any phi0 / zero-pad bookkeeping).
        phases = np.exp(-1j * np.outer(m_axis[: ring_nyq + 1], phi))
        G[j, : ring_nyq + 1] = phases @ f_ring
    return G


def _normalised_alf(x, lmax, m):
    """Fully-normalised associated Legendre P-bar_{l,m}(x) for l = m..lmax.

    Normalised so ``\\int_{-1}^{1} P-bar_{l,m}(x)^2 dx = 1`` (no Condon-Shortley
    sign; the sign cancels in power). Stable forward column recurrence in l at
    fixed m (Holmes & Featherstone). Returns ``[lmax-m+1, len(x)]``.
    """

    x = np.asarray(x, dtype=float)
    s = np.sqrt(np.clip(1.0 - x * x, 0.0, None))  # sin(theta)
    out = np.zeros((lmax - m + 1, x.size))

    # Sectoral seed P-bar_{m,m}.
    pmm = np.full(x.size, 1.0 / np.sqrt(2.0))  # P-bar_{0,0}
    for mm in range(1, m + 1):
        pmm = np.sqrt((2.0 * mm + 1.0) / (2.0 * mm)) * s * pmm
    out[0] = pmm
    if lmax == m:
        return out

    # First step up in l: P-bar_{m+1,m}.
    p_prev2 = pmm
    p_prev1 = np.sqrt(2.0 * m + 3.0) * x * pmm
    out[1] = p_prev1
    for ell in range(m + 2, lmax + 1):
        a = np.sqrt(
            (2.0 * ell - 1.0) * (2.0 * ell + 1.0)
            / ((ell - m) * (ell + m))
        )
        b = np.sqrt(
            (2.0 * ell + 1.0) * (ell - m - 1.0) * (ell + m - 1.0)
            / ((2.0 * ell - 3.0) * (ell - m) * (ell + m))
        )
        p_curr = a * x * p_prev1 - b * p_prev2
        out[ell - m] = p_curr
        p_prev2 = p_prev1
        p_prev1 = p_curr
    return out


def _alm_native(G, layout, lmax):
    """Gauss-Legendre latitude projection -> a_{l,m} (m >= 0), native engine.

    ``a_{l,m} = sqrt(2*pi) * sum_j (w_j / nphi_j) * G[j,m] * P-bar_{l,m}(x_j)``
    with the orthonormal spherical-harmonic convention
    ``Y_{l,m} = (1/sqrt(2*pi)) P-bar_{l,m}(cos theta) e^{i m phi}`` (matches the
    ducc0 a_lm convention). Returns ``[lmax+1, lmax+1]`` complex, lower-triangular
    in (l, m).
    """

    x = layout["x"]
    coeff_ring = layout["w"] / layout["nphi"]  # w_j / nphi_j per ring
    alm = np.zeros((lmax + 1, lmax + 1), dtype=complex)
    sqrt_2pi = np.sqrt(2.0 * np.pi)
    for m in range(lmax + 1):
        plm = _normalised_alf(x, lmax, m)  # [lmax-m+1, n_rings]
        weighted = coeff_ring * G[:, m]  # [n_rings] complex
        # a_{l,m} for l=m..lmax
        alm[m:, m] = sqrt_2pi * (plm @ weighted)
    return alm


def _power_from_alm(alm, lmax):
    """Angular power spectrum C_l from the m>=0 a_lm of a real field.

    Real field => a_{l,-m} = (-1)^m conj(a_{l,m}), so
    ``C_l = (|a_{l,0}|^2 + 2 sum_{m=1}^{l} |a_{l,m}|^2) / (2l + 1)``.
    """

    cl = np.zeros(lmax + 1)
    for ell in range(lmax + 1):
        m0 = np.abs(alm[ell, 0]) ** 2
        mpos = np.sum(np.abs(alm[ell, 1: ell + 1]) ** 2)
        cl[ell] = (m0 + 2.0 * mpos) / (2.0 * ell + 1.0)
    return cl


def _alm_ducc0(field, layout, lmax):
    """a_{l,m} (m >= 0) via ducc0's exact general SHT analysis on the O96 rings.

    Uses the orthonormal convention (matches the native engine). The field is
    premultiplied by the per-point quadrature weight w_j * (2*pi / nphi_j) and
    fed to ``adjoint_synthesis`` (= sum over points of value * conj(Y)),
    which equals the analysis integral on this exact-quadrature grid.
    """

    import ducc0  # hard dependency; raises ImportError if the engine is missing

    values = np.asarray(field, dtype=float).reshape(-1)
    n_rings = layout["n_rings"]
    # ducc0 wants rings described by colatitude theta, nphi, phi0, ringstart and
    # a flat map laid out ring-contiguous. Build that layout, applying the
    # longitude quadrature factor 2*pi/nphi and the Gauss weight w_j per point.
    theta = np.arccos(np.clip(layout["x"], -1.0, 1.0))  # colatitude in [0, pi]
    nphi = layout["nphi"].astype(np.uint64)
    flat = np.empty(values.size, dtype=np.float64)
    phi0 = np.empty(n_rings, dtype=np.float64)
    ringstart = np.empty(n_rings, dtype=np.uint64)
    cursor = 0
    lon_rad = np.radians(layout["lon"])
    for j in range(n_rings):
        idx = layout["row_index"][j]
        wq = float(layout["w"][j]) * (2.0 * np.pi / idx.size)
        ringstart[j] = cursor
        phi0[j] = float(lon_rad[idx][0])
        flat[cursor: cursor + idx.size] = values[idx] * wq
        cursor += idx.size

    mmax = lmax
    nalm = (mmax + 1) * (mmax + 2) // 2
    alm_packed = ducc0.sht.adjoint_synthesis(
        map=flat.reshape(1, -1),
        theta=theta.astype(np.float64),
        nphi=nphi,
        phi0=phi0,
        ringstart=ringstart,
        lmax=lmax,
        mmax=mmax,
        spin=0,
        nthreads=1,
    )[0]
    assert alm_packed.size == nalm
    # ducc0 packs a_lm in m-major triangular order: for m=0..mmax, l=m..lmax.
    alm = np.zeros((lmax + 1, lmax + 1), dtype=complex)
    pos = 0
    for m in range(mmax + 1):
        count = lmax - m + 1
        alm[m:, m] = alm_packed[pos: pos + count]
        pos += count
    return alm


def _weighted_variance(field, layout, *, remove_monopole=True):
    """Area-weighted variance Var_w(f) with per-point weight w_j * (2*pi/nphi_j),
    normalised to total weight 4*pi. Used by the Parseval check."""

    values = np.asarray(field, dtype=float).reshape(-1)
    w_point = np.empty(values.size)
    for j in range(layout["n_rings"]):
        idx = layout["row_index"][j]
        w_point[idx] = float(layout["w"][j]) * (2.0 * np.pi / idx.size)
    total = w_point.sum()  # == 4*pi
    mean = (w_point * values).sum() / total if remove_monopole else 0.0
    return float((w_point * (values - mean) ** 2).sum() / total)


def sampled_spherical_power_spectrum(
    latlons_deg,
    fields,
    *,
    lmax=None,
    remove_monopole=True,
    normalise="band_power",
    engine="ducc0",
    lmax_resolved=None,
    parseval_tol=0.01,
):
    """Angular power spectrum C_l on the O96 reduced-Gaussian sphere.

    Uses the open-source ducc0 spherical-harmonic transform on the native O96
    reduced-Gaussian rings by default, with NO lat-lon regrid. RUNG-3 bracket
    diagnostic only -- there is no GMM-derivable target spectrum (open problem).
    Not an optimisation target, not validation; the ERA5 reference
    (added by the caller as an extra series) is a direction-of-realism reference,
    never a target.

    Parameters
    ----------
    latlons_deg : [N, 2]
        The SAME latlons as the samples (O96). Asserted to lie on Gauss-Legendre
        latitudes.
    fields : {name: [N]}
        Standardised z-fields. The monopole (l=0) is removed internally when
        ``remove_monopole`` (mirrors the variogram de-trending: shape-only
        comparison, since de-standardisation stats are unavailable, audit S1).
    lmax : int, optional
        Transform ceiling; default the grid-derived nominal (``O96_LMAX_NOMINAL``).
    remove_monopole : bool
        Drop l=0 (trend) from the reported spectra.
    normalise : {None, "band_power"}
        ``"band_power"`` divides each C_l by its sum over the resolved band
        (unit resolved-band power; shape-only). ``None`` keeps raw C_l.
    engine : {"native", "ducc0", "auto"}
        Transform backend. ``"ducc0"`` (default) is the required open-source C++
        SHT library used for reported spectra; ``"native"`` is the pure-numpy
        cross-check engine; ``"auto"`` uses ducc0 if importable else native. The
        two engines are validated to agree to < 1e-3 over the resolved band
        (test_spherical_spectrum.py), with native round-trip and Parseval checks
        retained as correctness gates.
    lmax_resolved : int, optional
        Force the resolved-band ceiling. Default: empirical -- the largest l for
        which the Parseval identity ``sum_l (2l+1) C_l ~= 4*pi*Var_w(f)`` holds to
        ``parseval_tol`` (relative), trimmed from the nominal ceiling downward and
        taken as the min across fields (expect ~180).
    parseval_tol : float
        Relative Parseval tolerance defining the resolved band.

    Returns
    -------
    (ell, spectra, lmax_resolved)
        ``ell`` is the integer-l axis over the resolved band (starting at 1 when
        ``remove_monopole``); ``spectra[name]`` is C_l for that field over the same
        band; ``lmax_resolved`` is the reported resolved ceiling.
    """

    if normalise not in (None, "band_power"):
        raise ValueError("normalise must be None or 'band_power'")
    layout = _o96_ring_layout(latlons_deg)
    if lmax is None:
        lmax = min(O96_LMAX_NOMINAL, layout["n_rings"] - 1)

    resolved_engine = engine
    if engine == "auto":
        try:
            import ducc0  # noqa: F401
            resolved_engine = "ducc0"
        except ImportError:
            resolved_engine = "native"

    def _cl(field):
        if resolved_engine == "ducc0":
            alm = _alm_ducc0(field, layout, lmax)
        elif resolved_engine == "native":
            mmax = lmax
            G = _ring_longitude_fft(field, layout, mmax)
            alm = _alm_native(G, layout, lmax)
        else:
            raise ValueError(f"unknown engine {engine!r}")
        return _power_from_alm(alm, lmax)

    raw = {name: _cl(field) for name, field in fields.items()}

    # Empirical resolved ceiling: largest l where the cumulative Parseval sum is
    # within tol of the area-weighted field variance, min across fields.
    if lmax_resolved is None:
        ceilings = []
        ell_full = np.arange(lmax + 1)
        for name, field in fields.items():
            var_w = _weighted_variance(field, layout, remove_monopole=True)
            target = 4.0 * np.pi * var_w
            if target <= 0:
                ceilings.append(lmax)
                continue
            cum = np.cumsum((2.0 * ell_full + 1.0) * raw[name])
            # The monopole carries the (removed) mean; start the cumulative sum
            # at l=1 so the comparison matches the de-trended variance.
            cum = cum - (2.0 * 0 + 1.0) * raw[name][0]
            rel = cum / target
            # Largest l with rel within [1-tol, 1+tol]; require we have reached
            # >= 1-tol at all, else fall back to nominal.
            within = np.abs(rel - 1.0) <= parseval_tol
            if np.any(within):
                ceilings.append(int(np.max(np.nonzero(within))))
            else:
                # Never reaches the variance within tol -> take the l closest to 1.
                ceilings.append(int(np.argmin(np.abs(rel - 1.0))))
        lmax_resolved = max(1, min(ceilings))

    ell_lo = 1 if remove_monopole else 0
    ell = np.arange(ell_lo, lmax_resolved + 1)
    spectra = {}
    for name, cl in raw.items():
        band = cl[ell_lo: lmax_resolved + 1].copy()
        if normalise == "band_power":
            denom = band.sum()
            if denom > 0:
                band = band / denom
        spectra[name] = band
    return ell, spectra, int(lmax_resolved)
