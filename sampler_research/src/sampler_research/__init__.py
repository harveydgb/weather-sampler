"""Utilities for the sampler research notebooks.

Phase 1/data-diagnostic helpers plus the Phase 2 Stage A baselines (graph,
roughness metrics, and yardstick fields).
"""

from sampler_research.io import SAMPLER_KEYS, load_sampler_arrays
from sampler_research.toy import (
    Phase1ToyConfig,
    make_phase1_toy,
    make_random_soft_dirichlet_pi,
    save_phase1_toy,
)
from sampler_research.graph import (
    graph_laplacian,
    grid_edges_8,
    roughness_edge_mean,
    roughness_sum,
    scale_free_roughness,
)
from sampler_research.spectral import radial_power_spectrum, spectral_roughness
from sampler_research.baselines import (
    gmm_nll_over_n,
    iid_baseline,
    mixture_mean_field,
    mode_field,
    score_field,
    smoothed_map_baseline,
    smoothest_mode_assignment,
    variance_scaled_baseline,
)
from sampler_research.regularised_map import (
    LambdaSweepPoint,
    OptimResult,
    lambda_sweep,
    minimise_at_lambda,
    nll_gradient,
    objective,
    objective_gradient,
)
from sampler_research.method4_mrf import (
    BetaSweepPoint,
    ModeExtraction,
    ValueMRFResult,
    beta_sweep,
    delta_nll_to_best_mode,
    extract_gmm_modes,
    solve_value_mrf,
    value_mrf_energy,
)
from sampler_research.forecast_diag import softening_metrics

__all__ = [
    "Phase1ToyConfig",
    "SAMPLER_KEYS",
    "load_sampler_arrays",
    "make_phase1_toy",
    "make_random_soft_dirichlet_pi",
    "save_phase1_toy",
    "graph_laplacian",
    "grid_edges_8",
    "roughness_edge_mean",
    "roughness_sum",
    "scale_free_roughness",
    "radial_power_spectrum",
    "spectral_roughness",
    "gmm_nll_over_n",
    "iid_baseline",
    "mixture_mean_field",
    "mode_field",
    "score_field",
    "smoothed_map_baseline",
    "smoothest_mode_assignment",
    "variance_scaled_baseline",
    "LambdaSweepPoint",
    "OptimResult",
    "lambda_sweep",
    "minimise_at_lambda",
    "nll_gradient",
    "objective",
    "objective_gradient",
    "BetaSweepPoint",
    "ModeExtraction",
    "ValueMRFResult",
    "beta_sweep",
    "delta_nll_to_best_mode",
    "extract_gmm_modes",
    "solve_value_mrf",
    "value_mrf_energy",
    "softening_metrics",
]
