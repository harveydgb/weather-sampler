import numpy as np

from sampler_research.experimental_local_variance import (
    LocalVarianceNeighborhoods,
    local_scale_free_roughness,
)


def test_local_scale_free_roughness_uses_endpoint_local_variance_average():
    field = np.array([0.0, 2.0, 4.0])
    edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
    neighborhoods = LocalVarianceNeighborhoods(
        indices=(
            np.array([0, 1], dtype=np.int64),
            np.array([0, 1, 2], dtype=np.int64),
            np.array([1, 2], dtype=np.int64),
        ),
        radius_grid_points=1,
        radius_km=1.0,
    )

    roughness = local_scale_free_roughness(field, edges, neighborhoods)

    assert not roughness.variance_collapsed
    assert np.isclose(roughness.r_tilde, 24.0 / 11.0)
    assert np.isclose(roughness.median_local_variance, 1.0)


def test_local_scale_free_roughness_flags_constant_local_collapse():
    field = np.ones(3)
    edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
    neighborhoods = LocalVarianceNeighborhoods(
        indices=(
            np.array([0, 1], dtype=np.int64),
            np.array([0, 1, 2], dtype=np.int64),
            np.array([1, 2], dtype=np.int64),
        ),
        radius_grid_points=1,
        radius_km=1.0,
    )

    roughness = local_scale_free_roughness(field, edges, neighborhoods)

    assert roughness.variance_collapsed
    assert roughness.r_tilde == 0.0
    assert roughness.collapsed_cell_fraction == 1.0
