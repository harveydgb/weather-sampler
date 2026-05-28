import json

import numpy as np

from sampler_research.baselines import mode_field, smoothest_mode_assignment
from sampler_research.graph import grid_edges_8, roughness_sum, scale_free_roughness, graph_laplacian
from sampler_research.io import SAMPLER_KEYS, load_npz, load_sampler_arrays
from sampler_research.toy import (
    Phase1MultimodalConfig,
    Phase1ToyConfig,
    make_phase1_multimodal_toy,
    make_phase1_toy,
    save_phase1_toy,
)


def test_phase1_baseline_shapes_and_invariants() -> None:
    config = Phase1ToyConfig(use_heteroscedastic_sigma=False)
    toy = make_phase1_toy(config)

    assert toy["pi"].shape == (8, 8, 4)
    assert toy["mu"].shape == (8, 8, 4)
    assert toy["sigma"].shape == (8, 8, 4)
    assert toy["coords"].shape == (8, 8, 2)
    assert np.allclose(toy["pi"].sum(axis=-1), 1.0)
    assert np.allclose(toy["sigma"], 1.0)
    assert np.allclose(
        np.sort(toy["mu"], axis=-1),
        np.sort(toy["component_fields"], axis=-1),
    )
    assert any(np.allclose(toy["component_fields"][..., k], toy["truth"]) for k in range(config.k))


def test_phase1_heteroscedastic_sigma_is_component_shared() -> None:
    toy = make_phase1_toy(Phase1ToyConfig(use_heteroscedastic_sigma=True))

    assert toy["sigma"].shape == (8, 8, 4)
    assert np.all(toy["sigma"] > 0)
    assert np.allclose(toy["sigma"], toy["sigma"][..., :1])
    assert toy["sigma"].min() < toy["sigma"].max()


def test_phase1_regime_boundary_pi_is_soft_non_uniform_and_permuted() -> None:
    config = Phase1ToyConfig(use_regime_boundary_pi=True)
    toy = make_phase1_toy(config)

    assert toy["variant_name"].item() == "phase_1_regime_boundary_pi"
    assert toy["pi_mode"].item() == "regime_boundary_soft"
    assert toy["raw_pi"].shape == (8, 8, 4)
    assert np.allclose(toy["pi"].sum(axis=-1), 1.0)
    assert np.all(toy["pi"] > config.pi_floor)
    assert not np.allclose(toy["pi"], config.pi_value)
    assert np.unique(np.argmax(toy["raw_pi"], axis=-1)).size > 1
    assert np.allclose(
        toy["pi"],
        np.take_along_axis(toy["raw_pi"], toy["perm"], axis=-1),
    )


def test_phase1_multimodal_has_well_separated_shared_means_and_jagged_dominance() -> None:
    config = Phase1MultimodalConfig()
    toy = make_phase1_multimodal_toy(config)

    assert toy["variant_name"].item() == "phase_1_multimodal"
    assert toy["pi_mode"].item() == "multimodal_regime_boundary_soft"
    assert toy["pi"].shape == (8, 8, 4)
    assert toy["mu"].shape == (8, 8, 4)
    assert np.allclose(toy["sigma"], config.sigma_value)

    # Every cell shares the same well-separated mean set (only label order
    # differs per cell), so the global value set is exactly the base means and
    # the per-cell gaps are several sigma.
    assert np.allclose(np.unique(toy["mu"]), np.sort(config.base_means))
    assert config.min_mean_separation >= 2.0 * config.sigma_value

    # Spatially-varying pi: non-uniform, floored, label-locality via perm.
    assert np.allclose(toy["pi"].sum(axis=-1), 1.0)
    assert np.all(toy["pi"] >= config.pi_floor)
    assert not np.allclose(toy["pi"], config.pi_value)
    assert np.allclose(
        toy["pi"], np.take_along_axis(toy["raw_pi"], toy["perm"], axis=-1)
    )

    # The dominance field d_i = argmax_k pi_ik must span more than one regime
    # and be jagged: its value field mu_{i,d_i} (== debug `truth`) is rough.
    dominance = np.argmax(toy["raw_pi"], axis=-1)
    assert np.unique(dominance).size > 1
    laplacian = graph_laplacian(8, 8)
    assert roughness_sum(toy["truth"], laplacian) > 0.0


def test_phase1_multimodal_breaks_the_astar_equals_mode_map_collapse() -> None:
    """On this toy `a*` must be *strictly* smoother than the per-cell MAP field.

    This is the whole reason the toy exists (Stage A memo §5): on the
    overlapping-mean regime-boundary toy the smoothest valid mode assignment
    already equals `mode_map`, leaving no roughness to remove. Well-separated
    means make a value-compatible non-dominant mode available, so `a*` is a real
    lower roughness anchor again.
    """

    toy = make_phase1_multimodal_toy(Phase1MultimodalConfig())
    pi, mu, sigma = toy["pi"], toy["mu"], toy["sigma"]
    edges = grid_edges_8(8, 8)

    mode_f, _ = mode_field(pi, mu, sigma)
    astar, _ = smoothest_mode_assignment(pi, mu)

    assert not np.array_equal(astar, mode_f)
    r_mode, _ = scale_free_roughness(mode_f, edges)
    r_astar, _ = scale_free_roughness(astar, edges)
    assert r_astar < r_mode - 1e-9


def test_phase1_multimodal_rejects_overlapping_means() -> None:
    """A config that reintroduces the overlapping-mean collapse must error."""

    try:
        make_phase1_multimodal_toy(
            Phase1MultimodalConfig(base_means=[0.0, 0.5, 1.0, 1.5], sigma_value=1.0)
        )
    except ValueError:
        return
    raise AssertionError("overlapping base_means should have raised")


def test_saved_phase1_file_contains_debug_metadata_but_sampler_loader_exposes_only_sampler_arrays(tmp_path) -> None:
    path = save_phase1_toy(tmp_path, Phase1ToyConfig(use_heteroscedastic_sigma=False))

    all_data = load_npz(path)
    assert set(SAMPLER_KEYS).issubset(all_data)
    for key in (
        "truth",
        "component_fields",
        "raw_pi",
        "perm",
        "seed",
        "variant_name",
        "constants",
    ):
        assert key in all_data

    constants = json.loads(all_data["constants"].item())
    assert constants["grid_n"] == 8
    assert constants["k"] == 4
    assert constants["pi_mode"] == "uniform"
    assert constants["sigma_mode"] == "fixed"

    sampler_arrays = load_sampler_arrays(path)
    assert tuple(sampler_arrays) == SAMPLER_KEYS
