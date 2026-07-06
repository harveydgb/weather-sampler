"""Experimental Phase 4 local-variance test sampler diagnostic.

TEST CODE ONLY: this script replays the Method 1 lambda sweep with an
experimental local-variance roughness denominator. It writes sidecar artifacts
only and must be reviewed before any submission, merge, or production sampler
use.

It intentionally does not modify ``method1_sweep.npz``, ``lambda_star.json``, or
``scores.csv``. The fixed-lambda fields are the existing Joint MAP fields; this
script only changes the roughness yardstick used to choose a test lambda.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sampler_research.experimental_local_variance import (
    TEST_ONLY_NOTICE,
    build_local_variance_neighborhoods,
    local_scale_free_roughness,
)
from sampler_research.io import load_real_marginal
from sampler_research.phase4_eval import select_lambda_star


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "outputs" / "runs" / "phase_4_real"
DATA_NPZ = REPO_ROOT / "outputs" / "data" / "phase_4_real_2t.npz"
TARGET_ANCHOR = "smoothed_map_n10"
DEFAULT_RADIUS_GRID_POINTS = 10


def _load_graph(run_dir):
    with np.load(run_dir / "masks.npz") as f:
        edges = np.asarray(f["edges"], dtype=np.int64)
        edge_lengths = (
            np.asarray(f["edge_arc_km"], dtype=float) if "edge_arc_km" in f.files else None
        )
    return edges, edge_lengths


def _fallback_record(lambdas, local_r, valid, target):
    positive = (lambdas > 0.0) & valid & np.isfinite(local_r)
    if not np.any(positive):
        raise ValueError("no positive lambda rows are valid under the local-variance metric")
    idx = int(np.argmin(np.where(positive, lambdas, np.inf)))
    return {
        "lambda_star_local_variance": float(lambdas[idx]),
        "target_local_r_tilde": float(target),
        "matched_local_r_tilde": float(local_r[idx]),
        "bracketed": False,
        "extend_direction": None,
        "clauses": ["no_valid_rows_smallest_lambda_fallback"],
    }


def build_record(data, run_dir, radius_grid_points):
    edges, edge_lengths = _load_graph(run_dir)
    neighborhoods = build_local_variance_neighborhoods(
        data["latlons"],
        edges,
        radius_grid_points=radius_grid_points,
        edge_lengths_km=edge_lengths,
    )

    with np.load(run_dir / "anchors.npz") as f:
        target_field = np.asarray(f[TARGET_ANCHOR], dtype=float)
    target = local_scale_free_roughness(target_field, edges, neighborhoods)
    if target.variance_collapsed:
        raise ValueError(f"{TARGET_ANCHOR} is locally variance-collapsed")

    with np.load(run_dir / "method1_sweep.npz") as f:
        sweep = {key: f[key] for key in f.files}
    lambdas = np.asarray(sweep["lambdas"], dtype=float)
    fields = np.asarray(sweep["fields"], dtype=float)
    valid = np.asarray(sweep.get("warm_start_sane", np.ones(len(lambdas), dtype=bool)), dtype=bool)

    results = [local_scale_free_roughness(field, edges, neighborhoods) for field in fields]
    local_r = np.asarray([res.r_tilde for res in results], dtype=float)
    collapsed = np.asarray([res.variance_collapsed for res in results], dtype=bool)

    try:
        sel = select_lambda_star(lambdas, local_r, collapsed, target.r_tilde, valid=valid)
        record = {
            "lambda_star_local_variance": sel.lambda_star,
            "target_local_r_tilde": sel.target_r_tilde,
            "matched_local_r_tilde": sel.matched_r_tilde,
            "bracketed": sel.bracketed,
            "extend_direction": sel.extend_direction,
            "clauses": sel.clauses,
        }
    except ValueError:
        record = _fallback_record(lambdas, local_r, valid, target.r_tilde)

    record.update(
        {
            "TEST_ONLY_REVIEW_REQUIRED": True,
            "notice": TEST_ONLY_NOTICE,
            "target_anchor": TARGET_ANCHOR,
            "radius_grid_points": int(radius_grid_points),
            "radius_km": target.radius_km,
            "metric": (
                "mean over edges of squared endpoint difference divided by the "
                "average endpoint local variance"
            ),
            "production_artifacts_modified": False,
            "local_r_tilde_by_lambda": [
                {"lambda": float(lam), "local_r_tilde": float(rt), "valid": bool(v)}
                for lam, rt, v in zip(lambdas, local_r, valid)
            ],
        }
    )
    arrays = {
        "lambdas": lambdas,
        "local_r_tilde": local_r,
        "local_variance_collapsed": collapsed,
        "warm_start_sane": valid,
        "radius_grid_points": np.asarray(radius_grid_points, dtype=np.int64),
        "radius_km": np.asarray(target.radius_km, dtype=float),
    }
    return record, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_NPZ)
    parser.add_argument("--out-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--radius-grid-points", type=int, default=DEFAULT_RADIUS_GRID_POINTS)
    args = parser.parse_args()

    data = load_real_marginal(args.data)
    record, arrays = build_record(data, args.out_dir, args.radius_grid_points)

    json_path = args.out_dir / "local_variance_test_sampler.json"
    npz_path = args.out_dir / "local_variance_test_sampler.npz"
    json_path.write_text(json.dumps(record, indent=2) + "\n")
    np.savez(npz_path, **arrays)
    print(f"[local-var-test] wrote {json_path}")
    print(f"[local-var-test] wrote {npz_path}")
    print(f"[local-var-test] {TEST_ONLY_NOTICE}")


if __name__ == "__main__":
    main()
