"""Forecast-regime softening diagnostics on the emitted per-location GMMs.

The headline result: the head's
mixture weights soften monotonically with forecast lead time. This module is the
single tested home for the *pinned* softening metrics, so the report, the
notebook, and the test suite all quote one implementation.

Definitions are pinned verbatim (quote these, not paraphrases),
on the 2t marginal, per lead:
  * median max-pi      -- median over cells of the largest mixture weight;
  * one-hot fraction   -- fraction of cells with max-pi > 0.9;
  * 2nd-mode mass      -- median over cells of the second-largest mixture weight;
  * practical bimodality -- audit S4 rule (`phase4_eval.practically_bimodal_mask`):
    a component pair with both pi >= 0.1 and |d mu| / max(sigma) > tau, tau in {1, 2};
  * K_eff              -- exp of the weight entropy (`phase4_eval.effective_k`);
    fraction > 1.5 reported.

`softening_metrics` returns exactly the canonical columns, so a per-lead CSV
built from it can be checked cell-for-cell against the pinned values (the
verification gate this module itself provides).
"""

import numpy as np

from sampler_research.phase4_eval import effective_k, practically_bimodal_mask

ONE_HOT_PI = 0.9
KEFF_THRESHOLD = 1.5

# Order of the per-lead metric columns (CSV / table contract).
SOFTENING_METRIC_KEYS = (
    "median_max_pi",
    "one_hot_fraction",
    "median_second_mode",
    "bimodal_frac_1sigma",
    "bimodal_frac_2sigma",
    "keff_frac_gt_1p5",
)


def softening_metrics(pi, mu, sigma):
    """The six pinned softening metrics for one lead's `[N, K]` 2t GMM.

    Returns a plain dict keyed by `SOFTENING_METRIC_KEYS`. Fractions are in
    `[0, 1]` (the log tables print them as percentages); `median_max_pi` and
    `median_second_mode` are weights in `[0, 1]`.
    """

    pi = np.asarray(pi, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if pi.ndim != 2:
        raise ValueError("pi must be [N, K] for a single lead")

    sorted_pi = np.sort(pi, axis=-1)
    max_pi = sorted_pi[:, -1]
    second_pi = sorted_pi[:, -2] if pi.shape[-1] >= 2 else np.zeros(pi.shape[0])
    keff = effective_k(pi)
    return {
        "median_max_pi": float(np.median(max_pi)),
        "one_hot_fraction": float(np.mean(max_pi > ONE_HOT_PI)),
        "median_second_mode": float(np.median(second_pi)),
        "bimodal_frac_1sigma": float(
            np.mean(practically_bimodal_mask(pi, mu, sigma, min_separation=1.0))
        ),
        "bimodal_frac_2sigma": float(
            np.mean(practically_bimodal_mask(pi, mu, sigma, min_separation=2.0))
        ),
        "keff_frac_gt_1p5": float(np.mean(keff > KEFF_THRESHOLD)),
    }
