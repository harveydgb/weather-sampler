"""Public API for the sampler research library.

Re-exports the toy-model construction and array I/O, the graph/Laplacian and
scale-free roughness metrics, the power-spectrum diagnostics, the Stage A
baseline fields, the two samplers (regularised MAP / Method 1 and the
mode-selection value-space MRF / Method 4), and the forecast-regime softening
metric. See each submodule for the details.
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
