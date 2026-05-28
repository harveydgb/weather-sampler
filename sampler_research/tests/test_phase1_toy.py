import json

import numpy as np

from sampler_research.io import SAMPLER_KEYS, load_npz, load_sampler_arrays
from sampler_research.toy import Phase1ToyConfig, make_phase1_toy, save_phase1_toy


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
