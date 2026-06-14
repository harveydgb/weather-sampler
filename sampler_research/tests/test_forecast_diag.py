"""Forecast-regime softening metrics: synthetic checks + the log oracle gate.

The oracle tests reproduce the canonical per-lead tables in research_notes/log.md
(2026-06-12 v1 me5; 2026-06-13 v2 me7) to displayed precision -- the same
verification gate the log entries used. They skip when the converted per-lead
npz are absent (local, gitignored artifacts; regenerate via
scripts/convert_real_gmm_pt_to_npz.py --format forecast).
"""

from pathlib import Path

import numpy as np
import pytest

from sampler_research.forecast_diag import SOFTENING_METRIC_KEYS, softening_metrics
from sampler_research.io import load_real_marginal

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "outputs" / "data"

# (prefix, step) -> (median_max_pi, one_hot_%, median_2nd_mode, bimod1_%, bimod2_%, keff_%)
# verbatim from research_notes/log.md.
ORACLE = {
    ("phase_4_fc48_6ep", 1): (0.953, 71.3, 0.042, 15.7, 2.8, 22.5),
    ("phase_4_fc48_6ep", 8): (0.803, 11.2, 0.151, 27.6, 2.9, 87.3),
    ("phase_4_fc48_14ep", 1): (0.944, 67.9, 0.044, 14.9, 2.5, 27.4),
    ("phase_4_fc48_14ep", 8): (0.766, 9.3, 0.178, 34.7, 6.3, 90.0),
}


def test_softening_metrics_keys_and_ranges():
    rng = np.random.default_rng(0)
    pi = rng.dirichlet(np.ones(4), size=500)
    mu = rng.normal(0, 1, size=(500, 4))
    sigma = rng.uniform(0.2, 1.0, size=(500, 4))
    m = softening_metrics(pi, mu, sigma)
    assert set(m) == set(SOFTENING_METRIC_KEYS)
    for v in m.values():
        assert 0.0 <= v <= 1.0


def test_softening_metrics_hand_values():
    # Two cells, K=3. Cell 0 near one-hot; cell 1 evenly split + separated means.
    pi = np.array([[0.95, 0.04, 0.01], [0.5, 0.4, 0.1]])
    mu = np.array([[0.0, 5.0, -5.0], [0.0, 10.0, 0.0]])
    sigma = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    m = softening_metrics(pi, mu, sigma)
    assert m["median_max_pi"] == pytest.approx(np.median([0.95, 0.5]))
    assert m["median_second_mode"] == pytest.approx(np.median([0.04, 0.4]))
    # cell 0 max-pi 0.95 > 0.9 (one-hot); cell 1 0.5 not -> 0.5 fraction
    assert m["one_hot_fraction"] == pytest.approx(0.5)
    # cell 1: components 0 and 1 both pi>=0.1, |0-10|/1 = 10 > 2sigma -> bimodal both thresholds
    assert m["bimodal_frac_2sigma"] == pytest.approx(0.5)
    assert m["bimodal_frac_1sigma"] == pytest.approx(0.5)


@pytest.mark.parametrize("key", list(ORACLE))
def test_softening_matches_log_oracle(key):
    prefix, step = key
    npz = DATA_DIR / f"{prefix}_step{step}_2t.npz"
    if not npz.exists():
        pytest.skip(f"{npz} absent (convert the forecast .pt first)")
    data = load_real_marginal(npz)
    m = softening_metrics(data["pi"], data["mu"], data["sigma"])
    max_pi, one_hot, second, b1, b2, keff = ORACLE[key]
    assert round(m["median_max_pi"], 3) == max_pi
    assert round(m["median_second_mode"], 3) == second
    assert round(100 * m["one_hot_fraction"], 1) == one_hot
    assert round(100 * m["bimodal_frac_1sigma"], 1) == b1
    assert round(100 * m["bimodal_frac_2sigma"], 1) == b2
    assert round(100 * m["keff_frac_gt_1p5"], 1) == keff


def test_softening_monotone_in_lead_when_all_leads_present():
    # RQ1: median max-pi decreases monotonically with lead (both runs), if present.
    for prefix in ("phase_4_fc48_6ep", "phase_4_fc48_14ep"):
        files = sorted(DATA_DIR.glob(f"{prefix}_step*_2t.npz"),
                       key=lambda p: int(p.stem.split("_step")[1].split("_2t")[0]))
        if len(files) < 8:
            pytest.skip(f"{prefix}: full lead set absent")
        max_pi = [softening_metrics(**{k: load_real_marginal(p)[k]
                                       for k in ("pi", "mu", "sigma")})["median_max_pi"]
                  for p in files]
        assert all(b <= a + 1e-9 for a, b in zip(max_pi, max_pi[1:])), max_pi
