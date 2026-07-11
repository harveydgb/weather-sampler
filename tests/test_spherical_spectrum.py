"""O96 angular power spectrum C_l -- transform-correctness gate (S6).

These tests are the correctness oracle for ``sampled_spherical_power_spectrum``
and its helpers. They run on the REAL O96 latlons (the exactness precondition is
the actual Gauss-Legendre geometry, audit S5), so they require the persisted
phase_4 marginal npz; they skip cleanly if it is absent.

Coverage:
  (a) round-trip / point spectrum -- a single Y_lm concentrates power at that l;
  (b) Parseval -- sum_l (2l+1) C_l == 4*pi * Var_w(f) (area-weighted variance);
  (c) bracket ordering -- iid white noise has more high-l power than a smooth
      field (the spectrum reaches the same bracket conclusion as the variogram);
  (d) engine agreement -- ducc0 vs native agree over the resolved band;
  (e) default engine -- public calls without an engine use the ducc0 path.
"""

import numpy as np
import pytest

from conftest import REPO_ROOT, needs_ducc0
from sampler_research.diagnostics import (
    O96_LMAX_NOMINAL,
    _alm_native,
    _normalised_alf,
    _o96_ring_layout,
    _power_from_alm,
    _ring_longitude_fft,
    _weighted_variance,
    sampled_spherical_power_spectrum,
)

DATA_NPZ = REPO_ROOT / "outputs" / "data" / "phase_4_real_2t.npz"
FORECAST_NPZ = REPO_ROOT / "outputs" / "data" / "phase_4_fc48_14ep_step8_2t.npz"


def _load_latlons():
    path = DATA_NPZ if DATA_NPZ.exists() else FORECAST_NPZ
    if not path.exists():
        pytest.skip("no persisted O96 marginal npz to source latlons from")
    with np.load(path) as f:
        return np.asarray(f["latlons"], dtype=float)


def _real_harmonic(latlons, ell, m):
    """A pure degree-l real field Re(Y_lm) on the grid (normalisation-agnostic)."""
    from scipy.special import sph_harm_y  # scipy >= 1.15 name

    colat = np.radians(90.0 - latlons[:, 0])  # polar angle theta in [0, pi]
    lon = np.radians(latlons[:, 1])           # azimuth phi
    return np.real(sph_harm_y(ell, m, colat, lon))


def _native_cl(latlons, field, lmax):
    layout = _o96_ring_layout(latlons)
    G = _ring_longitude_fft(field, layout, lmax)
    alm = _alm_native(G, layout, lmax)
    return _power_from_alm(alm, lmax)


def test_alf_orthonormal_on_gauss_nodes():
    """P-bar_{l,m} are orthonormal under Gauss-Legendre quadrature: this is the
    normalisation the whole transform rests on."""
    lmax = 24
    x, w = np.polynomial.legendre.leggauss(lmax + 8)
    for m in (0, 1, 5):
        plm = _normalised_alf(x, lmax, m)  # [lmax-m+1, nx]
        gram = (plm * w) @ plm.T
        assert np.allclose(gram, np.eye(lmax - m + 1), atol=1e-9)


def test_point_spectrum_concentrates_at_input_degree():
    latlons = _load_latlons()
    lmax = 40
    for ell, m in ((10, 4), (17, 0)):
        field = _real_harmonic(latlons, ell, m)
        cl = _native_cl(latlons, field, lmax)
        frac = cl[ell] / cl.sum()
        assert frac > 0.99, f"l={ell} m={m}: only {frac:.4f} of power at the degree"


def test_parseval_identity_native():
    latlons = _load_latlons()
    lmax = min(O96_LMAX_NOMINAL, _o96_ring_layout(latlons)["n_rings"] - 1)
    rng = np.random.default_rng(0)
    # A smooth, band-limited test field: superposition of a few low harmonics.
    field = sum(_real_harmonic(latlons, ell, m) * rng.normal()
                for ell, m in ((3, 1), (6, 2), (9, 0), (12, 5)))
    layout = _o96_ring_layout(latlons)
    cl = _native_cl(latlons, field, lmax)
    ell_full = np.arange(lmax + 1)
    lhs = float(np.sum((2.0 * ell_full[1:] + 1.0) * cl[1:]))  # drop monopole
    rhs = 4.0 * np.pi * _weighted_variance(field, layout, remove_monopole=True)
    assert abs(lhs / rhs - 1.0) < 0.01, f"Parseval off: lhs/rhs={lhs / rhs:.5f}"


def test_iid_has_more_high_l_power_than_smooth_field():
    latlons = _load_latlons()
    lmax = 60
    rng = np.random.default_rng(1)
    iid = rng.normal(size=latlons.shape[0])
    smooth = _real_harmonic(latlons, 4, 1) + 0.5 * _real_harmonic(latlons, 6, 2)
    cl_iid = _native_cl(latlons, iid, lmax)
    cl_smooth = _native_cl(latlons, smooth, lmax)
    hi = slice(lmax // 2, lmax + 1)
    assert cl_iid[hi].sum() > cl_smooth[hi].sum()
    # The smooth field's power must live almost entirely at low l (bracket sense).
    assert cl_smooth[hi].sum() / cl_smooth[1:].sum() < 0.05


@needs_ducc0
def test_engine_agreement_ducc0_vs_native():
    latlons = _load_latlons()
    rng = np.random.default_rng(2)
    field = rng.normal(size=latlons.shape[0])
    fields = {"f": field}
    ell_n, spec_n, lr_n = sampled_spherical_power_spectrum(
        latlons, fields, engine="native", normalise=None, lmax_resolved=120
    )
    ell_d, spec_d, lr_d = sampled_spherical_power_spectrum(
        latlons, fields, engine="ducc0", normalise=None, lmax_resolved=120
    )
    assert np.array_equal(ell_n, ell_d)
    rel = np.abs(spec_d["f"] - spec_n["f"]) / np.maximum(np.abs(spec_n["f"]), 1e-30)
    assert np.median(rel) < 1e-3, f"median rel disagreement {np.median(rel):.2e}"


@needs_ducc0
def test_public_api_default_engine_is_ducc0():
    latlons = _load_latlons()
    rng = np.random.default_rng(4)
    fields = {"f": rng.normal(size=latlons.shape[0])}
    ell_default, spec_default, lr_default = sampled_spherical_power_spectrum(
        latlons, fields, normalise=None, lmax_resolved=80
    )
    ell_ducc0, spec_ducc0, lr_ducc0 = sampled_spherical_power_spectrum(
        latlons, fields, engine="ducc0", normalise=None, lmax_resolved=80
    )
    assert lr_default == lr_ducc0
    assert np.array_equal(ell_default, ell_ducc0)
    assert np.allclose(spec_default["f"], spec_ducc0["f"], rtol=0.0, atol=0.0)


@needs_ducc0
def test_public_api_shapes_and_band_power_normalisation():
    latlons = _load_latlons()
    rng = np.random.default_rng(3)
    fields = {"a": rng.normal(size=latlons.shape[0]),
              "b": _real_harmonic(latlons, 8, 3)}
    ell, spectra, lmax_resolved = sampled_spherical_power_spectrum(
        latlons, fields, normalise="band_power"
    )
    assert ell[0] == 1  # monopole dropped
    assert ell[-1] == lmax_resolved
    for name, cl in spectra.items():
        assert cl.shape == ell.shape
        assert np.isclose(cl.sum(), 1.0, atol=1e-9)  # unit resolved-band power
