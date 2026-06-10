"""Phase 1 toy GMM construction.

Two decoder-faithful toys are produced, differing only in the noise scale:

- ``phase_1_homoscedastic``   — fixed ``sigma = 1``.
- ``phase_1_heteroscedastic`` — per-location, component-shared ``sigma``.

Both share the same construction: K slowly-varying quadratic component-mean
surfaces (``component_fields``), individuated by a small per-component height
offset and reordered per cell by a seeded permutation, with independently drawn
floored Dirichlet mixture weights at each grid cell. The component means overlap
on the scale of ``sigma`` on purpose — real decoder GMMs are not cleanly
separated — but the per-cell value sets vary in space, so the smoothest valid
mode-assignment anchor (``a*``) is non-degenerate.

There is no privileged ``truth`` field: the only "truth" is the emitted GMM
(``pi``, ``mu``, ``sigma``). The slowly-varying mean surfaces are kept as
debug-only ``component_fields`` and are never scored. The sampler loader exposes
only ``pi``, ``mu``, ``sigma``, and ``coords``.
"""

import json
from pathlib import Path

import numpy as np

from sampler_research.io import save_npz


class Phase1ToyConfig:
    def __init__(
        self,
        seed=0,
        grid_n=8,
        x_min=-2.0,
        x_max=2.0,
        y_min=-2.0,
        y_max=2.0,
        k=4,
        peak_offsets=None,
        peak_scale=-0.3,
        peak_height=10.0,
        peak_height_offsets=None,
        use_heteroscedastic_sigma=False,
        sigma_value=1.0,
        sigma_min=0.5,
        sigma_slope=0.35,
        dirichlet_alpha=4.0,
        pi_floor=0.05,
    ):
        self.seed = seed
        self.grid_n = grid_n
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.k = k
        # Spatial peak locations of the K quadratic surfaces; peak of component
        # k sits at (-a_k, -b_k).
        self.peak_offsets = (
            np.array(peak_offsets, dtype=float)
            if peak_offsets is not None
            else np.array(
                [
                    [0.0, 0.0],
                    [1.0, 1.0],
                    [1.0, -1.0],
                    [-1.0, -1.0],
                ],
                dtype=float,
            )
        )
        self.peak_scale = peak_scale
        self.peak_height = peak_height
        # Per-component additive height offset. Individuates the K otherwise
        # near-symmetric sheets while keeping them overlapping on the sigma scale.
        self.peak_height_offsets = (
            np.array(peak_height_offsets, dtype=float)
            if peak_height_offsets is not None
            else np.array([1.5, 0.0, -1.0, 0.5], dtype=float)
        )
        self.use_heteroscedastic_sigma = use_heteroscedastic_sigma
        self.sigma_value = sigma_value
        self.sigma_min = sigma_min
        self.sigma_slope = sigma_slope
        self.dirichlet_alpha = dirichlet_alpha
        self.pi_floor = pi_floor

    @property
    def pi_value(self):
        return 1.0 / self.k

    @property
    def pi_mode(self):
        return "random_soft_dirichlet"

    @property
    def sigma_mode(self):
        return "heteroscedastic_spatial" if self.use_heteroscedastic_sigma else "fixed"

    @property
    def variant_name(self):
        return (
            "phase_1_heteroscedastic"
            if self.use_heteroscedastic_sigma
            else "phase_1_homoscedastic"
        )

    @property
    def output_filename(self):
        return f"{self.variant_name}.npz"

    def constants(self):
        return {
            "seed": self.seed,
            "grid_n": self.grid_n,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "k": self.k,
            "peak_offsets": self.peak_offsets.tolist(),
            "peak_scale": self.peak_scale,
            "peak_height": self.peak_height,
            "peak_height_offsets": self.peak_height_offsets.tolist(),
            "sigma_mode": self.sigma_mode,
            "sigma_value": self.sigma_value,
            "sigma_min": self.sigma_min,
            "sigma_slope": self.sigma_slope,
            "pi_mode": self.pi_mode,
            "pi_value": self.pi_value,
            "dirichlet_alpha": self.dirichlet_alpha,
            "pi_floor": self.pi_floor,
        }


def make_random_soft_dirichlet_pi(rng, grid_n, k, alpha, pi_floor):
    """Draw independent floored Dirichlet mixture weights for each grid cell."""

    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if not (0.0 <= pi_floor < 1.0 / k):
        raise ValueError("pi_floor must satisfy 0 <= pi_floor < 1 / k")
    q = rng.dirichlet(np.full(k, alpha, dtype=float), size=(grid_n, grid_n))
    return pi_floor + (1.0 - k * pi_floor) * q


def make_phase1_toy(config=None):
    """Build the Phase 1 toy arrays and debug metadata."""

    config = config or Phase1ToyConfig()
    if config.peak_offsets.shape != (config.k, 2):
        raise ValueError("peak_offsets must have shape [k, 2]")
    if config.peak_height_offsets.shape != (config.k,):
        raise ValueError("peak_height_offsets must have shape [k]")
    if not (0.0 <= config.pi_floor < 1.0 / config.k):
        raise ValueError("pi_floor must satisfy 0 <= pi_floor < 1 / k")
    if config.dirichlet_alpha <= 0.0:
        raise ValueError("dirichlet_alpha must be positive")

    x_1d = np.linspace(config.x_min, config.x_max, config.grid_n)
    y_1d = np.linspace(config.y_min, config.y_max, config.grid_n)
    x_grid, y_grid = np.meshgrid(x_1d, y_1d, indexing="ij")
    coords = np.stack([x_grid, y_grid], axis=-1)

    # The seeded RNG draws the per-cell permutations first, then the Dirichlet
    # weights. Sigma is closed-form (no RNG draws), so the homoscedastic and
    # heteroscedastic toys share identical `pi` and `mu` for a given seed and
    # sigma is the only controlled difference between them.
    rng = np.random.default_rng(config.seed)
    perm = np.empty((config.grid_n, config.grid_n, config.k), dtype=np.int64)
    for i in range(config.grid_n):
        for j in range(config.grid_n):
            perm[i, j] = rng.permutation(config.k)

    offsets = np.asarray(config.peak_offsets, dtype=float)
    a = offsets[:, 0]
    b = offsets[:, 1]
    component_fields = (
        config.peak_scale * ((x_grid[..., None] + a) ** 2 + (y_grid[..., None] + b) ** 2)
        + config.peak_height
        + config.peak_height_offsets
    )
    mu = np.take_along_axis(component_fields, perm, axis=-1)

    if config.use_heteroscedastic_sigma:
        r = np.sqrt(x_grid**2 + y_grid**2)
        sigma = np.broadcast_to(
            (config.sigma_min + config.sigma_slope * r)[..., None],
            (config.grid_n, config.grid_n, config.k),
        ).copy()
    else:
        sigma = np.full((config.grid_n, config.grid_n, config.k), config.sigma_value)

    pi = make_random_soft_dirichlet_pi(
        rng, config.grid_n, config.k, config.dirichlet_alpha, config.pi_floor
    )

    validate_phase1_toy(
        pi=pi,
        mu=mu,
        sigma=sigma,
        coords=coords,
        component_fields=component_fields,
        perm=perm,
        config=config,
    )

    return {
        "pi": pi,
        "mu": mu,
        "sigma": sigma,
        "coords": coords,
        "component_fields": component_fields,
        "perm": perm,
        "seed": np.asarray(config.seed),
        "variant_name": np.asarray(config.variant_name),
        "pi_mode": np.asarray(config.pi_mode),
        "sigma_mode": np.asarray(config.sigma_mode),
        "use_heteroscedastic_sigma": np.asarray(config.use_heteroscedastic_sigma),
        "dirichlet_alpha": np.asarray(config.dirichlet_alpha),
        "pi_floor": np.asarray(config.pi_floor),
        "constants": np.asarray(json.dumps(config.constants(), sort_keys=True)),
    }


def validate_phase1_toy(
    *,
    pi,
    mu,
    sigma,
    coords,
    component_fields,
    perm,
    config,
):
    """Validate the toy invariants from the Phase 1 note."""

    expected_field_shape = (config.grid_n, config.grid_n, config.k)
    if pi.shape != expected_field_shape:
        raise ValueError(f"pi has shape {pi.shape}, expected {expected_field_shape}")
    if mu.shape != expected_field_shape:
        raise ValueError(f"mu has shape {mu.shape}, expected {expected_field_shape}")
    if sigma.shape != expected_field_shape:
        raise ValueError(f"sigma has shape {sigma.shape}, expected {expected_field_shape}")
    if coords.shape != (config.grid_n, config.grid_n, 2):
        raise ValueError(
            f"coords has shape {coords.shape}, expected {(config.grid_n, config.grid_n, 2)}"
        )
    if component_fields.shape != expected_field_shape:
        raise ValueError("component_fields has the wrong shape")
    if perm.shape != expected_field_shape:
        raise ValueError("perm has the wrong shape")

    if not np.allclose(pi.sum(axis=-1), 1.0):
        raise ValueError("pi rows must sum to 1")
    if not (np.all(np.isfinite(pi)) and np.all(pi >= config.pi_floor)):
        raise ValueError("random pi must be finite and stay at or above pi_floor")
    if np.allclose(pi, config.pi_value):
        raise ValueError("random pi must be spatially non-uniform")

    if not (np.all(np.isfinite(sigma)) and np.all(sigma > 0)):
        raise ValueError("sigma must be finite and positive")
    if config.use_heteroscedastic_sigma:
        if not np.allclose(sigma, sigma[..., :1]):
            raise ValueError("heteroscedastic sigma must be component-shared within each cell")
        if not (sigma.min() < sigma.max()):
            raise ValueError("heteroscedastic sigma must vary spatially")
    elif not np.allclose(sigma, config.sigma_value):
        raise ValueError("homoscedastic toy must use sigma_value everywhere")

    # The per-cell permutation reorders labels but must not change the set of
    # component values at any cell.
    if not np.allclose(np.sort(mu, axis=-1), np.sort(component_fields, axis=-1)):
        raise ValueError("per-cell permutation changed the component value set")


def save_phase1_toy(output_dir, config=None):
    """Build and save a Phase 1 toy `.npz` file."""

    config = config or Phase1ToyConfig()
    data = make_phase1_toy(config)
    return save_npz(Path(output_dir) / config.output_filename, data)
