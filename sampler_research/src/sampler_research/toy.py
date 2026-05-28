"""Phase 1 and Phase 1.5 toy GMM construction."""

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
        use_heteroscedastic_sigma=False,
        sigma_value=1.0,
        sigma_min=0.5,
        sigma_slope=0.35,
        use_regime_boundary_pi=False,
        tau_pi=2.0,
        pi_floor=0.05,
    ):
        self.seed = seed
        self.grid_n = grid_n
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.k = k
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
        self.use_heteroscedastic_sigma = use_heteroscedastic_sigma
        self.sigma_value = sigma_value
        self.sigma_min = sigma_min
        self.sigma_slope = sigma_slope
        self.use_regime_boundary_pi = use_regime_boundary_pi
        self.tau_pi = tau_pi
        self.pi_floor = pi_floor

    @property
    def pi_value(self):
        return 1.0 / self.k

    @property
    def pi_mode(self):
        return "regime_boundary_soft" if self.use_regime_boundary_pi else "uniform"

    @property
    def sigma_mode(self):
        return "heteroscedastic_spatial" if self.use_heteroscedastic_sigma else "fixed"

    @property
    def variant_name(self):
        if self.use_regime_boundary_pi and self.use_heteroscedastic_sigma:
            return "phase_1_regime_boundary_pi_heteroscedastic_sigma"
        if self.use_regime_boundary_pi:
            return "phase_1_regime_boundary_pi"
        return (
            "phase_1_field_heteroscedastic_sigma"
            if self.use_heteroscedastic_sigma
            else "phase_1_field"
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
            "sigma_mode": self.sigma_mode,
            "sigma_value": self.sigma_value,
            "sigma_min": self.sigma_min,
            "sigma_slope": self.sigma_slope,
            "pi_mode": self.pi_mode,
            "pi_value": self.pi_value,
            "tau_pi": self.tau_pi,
            "pi_floor": self.pi_floor,
        }


def make_phase1_toy(config=None):
    """Build the Phase 1 toy arrays and debug metadata."""

    config = config or Phase1ToyConfig()
    if config.peak_offsets.shape != (config.k, 2):
        raise ValueError("peak_offsets must have shape [k, 2]")
    if config.use_regime_boundary_pi:
        if not (0.0 <= config.pi_floor < 1.0 / config.k):
            raise ValueError("pi_floor must satisfy 0 <= pi_floor < 1 / k")
        if config.tau_pi <= 0.0:
            raise ValueError("tau_pi must be positive")

    x_1d = np.linspace(config.x_min, config.x_max, config.grid_n)
    y_1d = np.linspace(config.y_min, config.y_max, config.grid_n)
    x_grid, y_grid = np.meshgrid(x_1d, y_1d, indexing="ij")

    coords = np.stack([x_grid, y_grid], axis=-1)
    truth = config.peak_scale * (x_grid**2 + y_grid**2) + config.peak_height

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

    raw_pi = make_raw_regime_boundary_pi(x_grid, y_grid, offsets, config)
    if config.use_regime_boundary_pi:
        pi = np.take_along_axis(raw_pi, perm, axis=-1)
    else:
        pi = np.full((config.grid_n, config.grid_n, config.k), config.pi_value)

    validate_phase1_toy(
        pi=pi,
        mu=mu,
        sigma=sigma,
        coords=coords,
        truth=truth,
        component_fields=component_fields,
        perm=perm,
        config=config,
    )

    return {
        "pi": pi,
        "mu": mu,
        "sigma": sigma,
        "coords": coords,
        "truth": truth,
        "component_fields": component_fields,
        "raw_pi": raw_pi,
        "perm": perm,
        "seed": np.asarray(config.seed),
        "variant_name": np.asarray(config.variant_name),
        "pi_mode": np.asarray(config.pi_mode),
        "sigma_mode": np.asarray(config.sigma_mode),
        "use_regime_boundary_pi": np.asarray(config.use_regime_boundary_pi),
        "use_heteroscedastic_sigma": np.asarray(config.use_heteroscedastic_sigma),
        "constants": np.asarray(json.dumps(config.constants(), sort_keys=True)),
    }


def make_raw_regime_boundary_pi(x_grid, y_grid, offsets, config):
    """Build unpermuted soft regime-boundary mixture weights."""

    if not config.use_regime_boundary_pi:
        return np.full((config.grid_n, config.grid_n, config.k), config.pi_value)

    centres = -np.asarray(offsets, dtype=float)
    dx = x_grid[..., None] - centres[:, 0]
    dy = y_grid[..., None] - centres[:, 1]
    logits = -(dx**2 + dy**2) / (2.0 * config.tau_pi**2)
    logits = logits - logits.max(axis=-1, keepdims=True)
    weights = np.exp(logits)
    softmax = weights / weights.sum(axis=-1, keepdims=True)
    return config.pi_floor + (1.0 - config.k * config.pi_floor) * softmax


def validate_phase1_toy(
    *,
    pi,
    mu,
    sigma,
    coords,
    truth,
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
        raise ValueError(f"coords has shape {coords.shape}, expected {(config.grid_n, config.grid_n, 2)}")
    if truth.shape != (config.grid_n, config.grid_n):
        raise ValueError(f"truth has shape {truth.shape}, expected {(config.grid_n, config.grid_n)}")
    if component_fields.shape != expected_field_shape:
        raise ValueError("component_fields has the wrong shape")
    if perm.shape != expected_field_shape:
        raise ValueError("perm has the wrong shape")

    if not np.allclose(pi.sum(axis=-1), 1.0):
        raise ValueError("pi rows must sum to 1")
    if not (np.all(np.isfinite(pi)) and np.all(pi >= 0.0)):
        raise ValueError("pi must be finite and non-negative")
    if config.use_regime_boundary_pi:
        if not np.all(pi >= config.pi_floor):
            raise ValueError("regime-boundary pi must stay at or above pi_floor")
        if np.allclose(pi, config.pi_value):
            raise ValueError("regime-boundary pi must be spatially non-uniform")
    elif not np.allclose(pi, config.pi_value):
        raise ValueError("uniform-pi toy must use pi_value everywhere")
    if not (np.all(np.isfinite(sigma)) and np.all(sigma > 0)):
        raise ValueError("sigma must be finite and positive")
    if config.use_heteroscedastic_sigma:
        if not np.allclose(sigma, sigma[..., :1]):
            raise ValueError("heteroscedastic sigma must be component-shared within each cell")
    elif not np.allclose(sigma, config.sigma_value):
        raise ValueError("fixed-sigma toy must use sigma_value everywhere")

    if not np.allclose(np.sort(mu, axis=-1), np.sort(component_fields, axis=-1)):
        raise ValueError("per-cell permutation changed the component value set")
    if not any(np.allclose(component_fields[..., k], truth) for k in range(config.k)):
        raise ValueError("truth must coincide with one component field at every cell")


def save_phase1_toy(
    output_dir,
    config=None,
):
    """Build and save a Phase 1 toy `.npz` file."""

    config = config or Phase1ToyConfig()
    data = make_phase1_toy(config)
    return save_npz(Path(output_dir) / config.output_filename, data)
