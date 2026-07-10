import json

import numpy as np

from sampler_research.baselines import mode_field, smoothest_mode_assignment
from sampler_research.graph import grid_edges_8, scale_free_roughness
from sampler_research.io import SAMPLER_KEYS, load_npz, load_sampler_arrays
from sampler_research.toy import Phase1ToyConfig, make_phase1_toy, save_phase1_toy
from scripts.generate_toy_data import configs_for_variant


def test_homoscedastic_shapes_and_invariants() -> None:
    config = Phase1ToyConfig(use_heteroscedastic_sigma=False)
    toy = make_phase1_toy(config)

    assert toy["variant_name"].item() == "phase_1_homoscedastic"
    assert toy["pi"].shape == (8, 8, 4)
    assert toy["mu"].shape == (8, 8, 4)
    assert toy["sigma"].shape == (8, 8, 4)
    assert toy["coords"].shape == (8, 8, 2)
    assert np.allclose(toy["sigma"], 1.0)
    # The per-cell permutation reorders labels but preserves each cell's value set.
    assert np.allclose(
        np.sort(toy["mu"], axis=-1),
        np.sort(toy["component_fields"], axis=-1),
    )


def test_heteroscedastic_sigma_is_component_shared_and_varies() -> None:
    toy = make_phase1_toy(Phase1ToyConfig(use_heteroscedastic_sigma=True))

    assert toy["variant_name"].item() == "phase_1_heteroscedastic"
    assert toy["sigma"].shape == (8, 8, 4)
    assert np.all(toy["sigma"] > 0)
    assert np.allclose(toy["sigma"], toy["sigma"][..., :1])
    assert toy["sigma"].min() < toy["sigma"].max()


def test_peak_height_offsets_are_applied_to_component_fields() -> None:
    config = Phase1ToyConfig(use_heteroscedastic_sigma=False)
    toy = make_phase1_toy(config)

    assert np.allclose(config.peak_height_offsets, [1.5, 0.0, -1.0, 0.5])

    x = np.linspace(config.x_min, config.x_max, config.grid_n)
    y = np.linspace(config.y_min, config.y_max, config.grid_n)
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
    a = config.peak_offsets[:, 0]
    b = config.peak_offsets[:, 1]
    expected = (
        config.peak_scale * ((x_grid[..., None] + a) ** 2 + (y_grid[..., None] + b) ** 2)
        + config.peak_height
        + config.peak_height_offsets
    )
    assert np.allclose(toy["component_fields"], expected)


def test_random_pi_is_floored_non_uniform_and_truth_free() -> None:
    config = Phase1ToyConfig(use_heteroscedastic_sigma=False)
    toy = make_phase1_toy(config)

    assert toy["pi_mode"].item() == "random_soft_dirichlet"
    assert np.allclose(toy["pi"].sum(axis=-1), 1.0)
    assert np.all(toy["pi"] >= config.pi_floor)
    assert not np.allclose(toy["pi"], config.pi_value)

    # The toys are truth-free: the only "truth" is the emitted GMM.
    assert "truth" not in toy
    assert "raw_pi" not in toy

    # Soft, non-uniform local weights: not one-hot, not effectively uniform.
    # Exact calibration belongs to the real checkpoint.
    max_pi = toy["pi"].max(axis=-1)
    assert 0.30 < np.median(max_pi) < 0.45
    assert np.max(max_pi) < 0.75


def test_random_pi_is_reproducible_by_seed_and_seed_sensitive() -> None:
    first = make_phase1_toy(Phase1ToyConfig(seed=7))
    second = make_phase1_toy(Phase1ToyConfig(seed=7))
    different = make_phase1_toy(Phase1ToyConfig(seed=8))

    assert np.array_equal(first["pi"], second["pi"])
    assert np.array_equal(first["mu"], second["mu"])
    assert not np.array_equal(first["pi"], different["pi"])


def test_homoscedastic_and_heteroscedastic_share_pi_and_mu() -> None:
    """Sigma is the only controlled difference between the two toys."""

    homo = make_phase1_toy(Phase1ToyConfig(seed=0, use_heteroscedastic_sigma=False))
    hetero = make_phase1_toy(Phase1ToyConfig(seed=0, use_heteroscedastic_sigma=True))

    assert np.array_equal(homo["pi"], hetero["pi"])
    assert np.array_equal(homo["mu"], hetero["mu"])
    assert not np.allclose(homo["sigma"], hetero["sigma"])


def test_a_star_is_non_degenerate_and_smoother_than_mode_map() -> None:
    """Regression guard for the flat-mean degeneracy fix.

    With slowly-varying means each cell offers a different value set, so the
    smoothest valid mode assignment `a*` cannot collapse to a global constant.
    It must carry real spatial variation and be strictly smoother than the
    per-cell MAP field (a genuine, beatable lower-roughness anchor).
    """

    toy = make_phase1_toy(Phase1ToyConfig(use_heteroscedastic_sigma=False))
    pi, mu, sigma = toy["pi"], toy["mu"], toy["sigma"]
    edges = grid_edges_8(8, 8)

    mode_f, _ = mode_field(pi, mu, sigma)
    astar, _ = smoothest_mode_assignment(pi, mu)

    assert astar.std() > 1e-6
    r_mode, mode_collapsed = scale_free_roughness(mode_f, edges)
    r_astar, astar_collapsed = scale_free_roughness(astar, edges)
    assert not astar_collapsed
    assert r_astar < r_mode - 1e-9


def test_generate_all_yields_exactly_the_two_variants() -> None:
    variants = [config.variant_name for _, config in configs_for_variant("all", seed=0)]
    assert variants == ["phase_1_homoscedastic", "phase_1_heteroscedastic"]


def test_saved_file_has_debug_metadata_but_loader_exposes_only_sampler_arrays(tmp_path) -> None:
    path = save_phase1_toy(tmp_path, Phase1ToyConfig(use_heteroscedastic_sigma=False))

    all_data = load_npz(path)
    assert set(SAMPLER_KEYS).issubset(all_data)
    for key in ("component_fields", "perm", "seed", "variant_name", "constants"):
        assert key in all_data
    assert "truth" not in all_data
    assert "raw_pi" not in all_data

    constants = json.loads(all_data["constants"].item())
    assert constants["grid_n"] == 8
    assert constants["k"] == 4
    assert constants["pi_mode"] == "random_soft_dirichlet"
    assert constants["sigma_mode"] == "fixed"

    sampler_arrays = load_sampler_arrays(path)
    assert tuple(sampler_arrays) == SAMPLER_KEYS
