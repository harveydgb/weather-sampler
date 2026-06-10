import numpy as np

from sampler_research.baselines import mode_field, score_field
from sampler_research.spectral import radial_power_spectrum, spectral_roughness
from sampler_research.toy import Phase1ToyConfig, make_phase1_toy


def test_radial_power_spectrum_excludes_dc_bin():
    rng = np.random.default_rng(0)
    field = rng.normal(size=(8, 8))

    centres, mean_power, counts = radial_power_spectrum(field)

    assert centres.shape == mean_power.shape == counts.shape
    assert counts.sum() == field.size - 1
    assert np.all(centres > 0.0)


def test_spectral_roughness_is_shift_and_scale_invariant():
    rng = np.random.default_rng(1)
    field = rng.normal(size=(8, 8))

    base = spectral_roughness(field)
    transformed = spectral_roughness(7.0 * field + 3.0)

    assert np.isclose(base["spectral_hf_ratio"], transformed["spectral_hf_ratio"])
    assert np.isclose(base["spectral_slope"], transformed["spectral_slope"])
    assert np.isclose(
        base["spectral_monotone_fraction"],
        transformed["spectral_monotone_fraction"],
    )


def test_constant_field_is_spectrally_collapsed():
    result = spectral_roughness(np.full((8, 8), 4.0))

    assert result["spectral_collapsed"]
    assert np.isclose(result["spectral_hf_ratio"], 0.0)
    assert np.isnan(result["spectral_slope"])
    assert np.isnan(result["spectral_monotone_fraction"])


def test_checkerboard_has_more_high_frequency_power_than_smooth_wave():
    y, x = np.indices((8, 8))
    checkerboard = ((x + y) % 2).astype(float)
    smooth = np.cos(2.0 * np.pi * x / 8.0)

    rough = spectral_roughness(checkerboard)
    calm = spectral_roughness(smooth)

    assert rough["spectral_hf_ratio"] > 0.9
    assert calm["spectral_hf_ratio"] < 0.2
    assert rough["spectral_hf_ratio"] > calm["spectral_hf_ratio"]


def test_score_field_exposes_secondary_spectral_keys():
    toy = make_phase1_toy(Phase1ToyConfig())
    pi, mu, sigma = toy["pi"], toy["mu"], toy["sigma"]
    field, _ = mode_field(pi, mu, sigma)

    scores = score_field(field, pi, mu, sigma)

    for key in (
        "spectral_hf_ratio",
        "spectral_slope",
        "spectral_monotone_fraction",
        "spectral_collapsed",
    ):
        assert key in scores
    assert np.isfinite(scores["spectral_hf_ratio"])


def test_score_field_spectral_path_accepts_flat_field():
    toy = make_phase1_toy(Phase1ToyConfig())
    pi, mu, sigma = toy["pi"], toy["mu"], toy["sigma"]
    field, _ = mode_field(pi, mu, sigma)

    grid_scores = score_field(field, pi, mu, sigma)
    flat_scores = score_field(field.reshape(-1), pi, mu, sigma)

    for key in ("nll_over_n", "r_tilde", "spectral_hf_ratio", "spectral_slope"):
        assert np.isclose(grid_scores[key], flat_scores[key])
