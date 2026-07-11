"""Pre-flight validation of a forecast-format GMM .pt (plan Step 0).

Loads each .pt with torch (run with the WeatherGenerator venv) and runs the
torch-free `sampler_research.io.validate_forecast_dict` contract check, printing
a structured report and exiting non-zero if any artifact fails. This is the one
reproducible artifact-provenance command (replaces ad-hoc torch snippets).

    ~/WeatherGenerator/.venv/bin/python scripts/validate_forecast_pt.py \
        ~/model_outputs/gmm_params_gmm_fc48_v1_me5_2t_f8.pt \
        ~/model_outputs/gmm_params_gmm_fc48_v2_me7_2t_f8.pt

Checks per file: top-level `steps` 1..N contiguous; per-step pi/mu_channel/
sigma_channel/latlons shapes + finiteness + pi-sums-to-1 + sigma>0; lead_hours
strictly increasing; norm_mean/norm_std present and indexable by ch_idx. With
`--expected-steps`/`--expected-ch-idx` it also asserts the 8-lead / 2t (ch 3)
expectation for these specific runs.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sampler_research.io import validate_forecast_dict  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", type=Path, nargs="+", help="forecast .pt path(s)")
    parser.add_argument("--expected-steps", type=int, default=8)
    parser.add_argument("--expected-ch-idx", type=int, default=3)
    args = parser.parse_args()

    import torch  # imported here so the module stays importable without torch

    any_problems = False
    for path in args.paths:
        print(f"\n=== {path} ===")
        if not Path(path).expanduser().exists():
            print("  MISSING FILE")
            any_problems = True
            continue
        obj = torch.load(str(Path(path).expanduser()), map_location="cpu", weights_only=False)
        report, problems = validate_forecast_dict(
            obj, expected_n_steps=args.expected_steps, expected_ch_idx=args.expected_ch_idx
        )
        print(json.dumps(report, indent=2, default=str, sort_keys=True))
        if problems:
            any_problems = True
            print(f"  PROBLEMS ({len(problems)}):")
            for p in problems:
                print(f"    - {p}")
        else:
            print("  OK: matches the forecast contract (no problems)")

    if any_problems:
        sys.exit("validation FAILED: see PROBLEMS above")
    print("\nAll forecast artifacts validated OK.")


if __name__ == "__main__":
    main()
