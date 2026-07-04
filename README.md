# Weather Sampler Research

Research code, notes, and diagnostics for sampler experiments around the WeatherGenerator
project.

The core problem is post-hoc sampling from decoder outputs that are independent
per-location GMMs. The sampler should produce a single spatially coherent,
high-likelihood field while staying faithful to the local GMMs, without retraining the
decoder.

## Research Method

This repository is developed in small, staged steps:

- Start with the simplest controlled problem that exposes the sampling issue.
- Keep each phase concise, with one canonical note as the source of truth.
- Promote settled decisions once; link to detail instead of duplicating it.
- Keep reusable logic in small Python modules, with notebooks used mainly for reports and
  diagnostics.
- Use synthetic truth only as a construction/debugging scaffold, not as the normal sampler
  efficacy target.
- Compare against simple baselines before adding sampler complexity.

The main notes-system rule is: one fact has one home. The project roadmap lives in
[`research_notes/large_notes.md`](research_notes/large_notes.md); phase-specific details live
in the relevant phase notes.

## Project Phases

- **Pre-phase:** check the real decoder schema and fix the minimal constraints that the toy
  and sampler must respect.
- **Phase 1:** define and generate the controlled 8x8 single-channel synthetic GMM toys:
  `phase_1_homoscedastic` and `phase_1_heteroscedastic`. Canonical spec:
  [`research_notes/phase_1.md`](research_notes/phase_1.md).
- **Phase 2:** choose and implement the core spatial sampling methodology. Canonical spec:
  [`research_notes/phase_2.md`](research_notes/phase_2.md). Detailed method investigation
  (closed; kept as the evidence layer):
  [`research_notes/phase_2_research_plan.md`](research_notes/phase_2_research_plan.md).
- **Phase 3:** evaluate sampler outputs with GMM NLL, declared-prior coherence diagnostics,
  and baselines. The evaluation plan currently lives in
  [`research_notes/large_notes.md`](research_notes/large_notes.md).
- **Phase 4:** run the chosen methods on real WeatherGenerator decoder GMMs on the native O96
  grid. Plan and scope verdicts: [`research_notes/phase_4_plan.md`](research_notes/phase_4_plan.md);
  data audit: [`research_notes/phase_4_data_audit.md`](research_notes/phase_4_data_audit.md).

## Current Status

- Phases 1–3 are complete: toys generated, Stage A–D sampler comparison executed on
  `phase_1_homoscedastic` (carry both Method 1 and Method 4), evaluation machinery built.
  The Stage B-TV / Method 5 toy ablations are closed and their code is archived in place
  (banners at the top of the files; verdicts promoted into the notes).
- **Phase 4 executed (11 June 2026)** on the real `gmm_era5_32ep_v3` extraction (O96 grid,
  N = 40,320): λ\* = 93.7 selected with no fallback, Method 1 beats smoothed-MAP at matched
  `R̃`, Method 4's β sweep pinned by the near-one-hot unary gap. Plan:
  [`research_notes/phase_4_plan.md`](research_notes/phase_4_plan.md); run record and headline
  numbers: [`research_notes/log.md`](research_notes/log.md) 2026-06-11 entries; report
  notebook: `notebooks/04_phase4_real_data.ipynb`.
- That artifact is a masked-autoencoder **step-0 reconstruction** (near-one-hot mixture
  weights — the expected low-uncertainty signature, not a bug); it is kept as the
  reconstruction-regime baseline (the left-hand anchor of the two-regime curve).
- **Forecast regime landed (13 June 2026):** two +6h..+48h checkpoints extracted —
  `gmm_fc48_v1` me5 (6-epoch preview) and `gmm_fc48_v2` me7 (converged, 14-epoch). The
  per-lead Phase 4 audit runs for both via
  [`scripts/run_phase4_forecast_leads.py`](scripts/run_phase4_forecast_leads.py); mixture
  weights **soften monotonically with lead** (median max-π 0.95→0.77 at +48h for the
  converged run) — reproduced cell-for-cell against
  [`research_notes/log.md`](research_notes/log.md). The forecast-specific reporting layer is
  built: softening metrics ([`scripts/run_forecast_softening.py`](scripts/run_forecast_softening.py)),
  marginal-faithfulness / do-no-harm ΔCRPS ([`scripts/run_forecast_faithfulness.py`](scripts/run_forecast_faithfulness.py)),
  and macro/table emission ([`scripts/emit_report_results.py`](scripts/emit_report_results.py));
  report notebook `notebooks/05_phase4_forecast_regime.ipynb`. Single init time (2023-11-01)
  and debug-scale model remain caveats; the 6-epoch column is a lower bound on softening.

## Layout

- `sampler_research/src/sampler_research/`: reusable Python helpers for toy GMMs, sampling, diagnostics, and I/O.
- `sampler_research/tests/`: focused tests for the research helpers.
- `scripts/`: runnable experiment and diagnostic scripts.
- `notebooks/`: exploratory notebooks.
- `research_notes/`: phase notes, meeting notes, and research planning.
- `outputs/`: generated data, figures, and run artifacts. Large/generated files are ignored by git.

## Generated Outputs

Current sampler-facing synthetic data files:

- `outputs/data/phase_1_homoscedastic.npz`
- `outputs/data/phase_1_heteroscedastic.npz`

Both share the same quadratic slowly-varying means and random floored Dirichlet mixture weights;
they differ only in the noise scale (fixed σ vs per-location component-shared σ). Sampler code
should consume only `pi`, `mu`, `sigma`, and `coords` from these files. Extra arrays such as
`component_fields`, `perm`, and variant metadata are for debugging and reproducibility; the toys
are truth-free (the only "truth" is the emitted GMM).

Real-data files (converted from the WeatherGenerator `.pt` artifact by
`scripts/convert_real_gmm_pt_to_npz.py`; facts about their contents live in
[`research_notes/phase_4_data_audit.md`](research_notes/phase_4_data_audit.md)):

- `outputs/data/phase_4_real_2t.npz` (+ `_meta.json`) — the canonical `gmm_era5_32ep_v3` 2t marginal
- `outputs/data/phase_4_real_2t_iidsource.npz` (+ `_meta.json`) — the iid-source companion

The upstream WeatherGenerator training and extraction steps are external to this repository;
the reproduction manual is [`docs/reproducing_the_model.md`](docs/reproducing_the_model.md).

Forecast-regime per-lead files (converted with
`scripts/convert_real_gmm_pt_to_npz.py --format forecast`, one `.npz` + `_meta.json`
per +6h..+48h lead step):

- `outputs/data/phase_4_fc48_6ep_step{1..8}_2t.npz` — 6-epoch preview (`gmm_fc48_v1` me5)
- `outputs/data/phase_4_fc48_14ep_step{1..8}_2t.npz` — converged (`gmm_fc48_v2` me7), canonical init A
- `outputs/data/phase_4_fc48_v2_init*_step{1..8}_2t.npz` — the converged replicate inits (B/C)

Run artifacts (fields, scores, probes) persist under `outputs/runs/` and figures under
`outputs/figures/`; both are regenerated by the `scripts/run_stage_*.py`, `scripts/run_phase4_*.py`,
and `scripts/make_phase4_figures.py` scripts. The Chapter-5 reporting chain (softening →
faithfulness → figures → emitted macros/tables) is wrapped as `make ch5`; see the `Makefile`.
The emitted report numbers are produced only by `scripts/emit_report_results.py`
(`report/construction/macros-results.tex` + `tables/*.tex`) — never hand-edited.

## Repo Alignment Prompt

Use this prompt after adding or reorganising code, notebooks, generated data, or notes:

```text
Review the weather-sampler-research repo for internal alignment after the latest changes.
Check README.md, pyproject.toml, scripts/, notebooks/, research_notes/, sampler_research/src/,
sampler_research/tests/, and outputs/ docs. Look for stale paths, renamed files, missing
optional dependencies, notebook output drift, generated artifact names, phase/status claims,
CLI defaults, imports, and notes-system ownership. Keep one fact in one canonical home,
link instead of duplicating it, do not edit meeting_notes.md except to append raw notes, and
ensure sampler-facing code only consumes pi, mu, sigma, and coords from generated .npz files.
Return findings first, then make low-risk documentation/path fixes.
```

## Local Setup

Use a lightweight local virtual environment for toy-data generation, notebooks, scripts, and
tests. Pick an interpreter that satisfies `requires-python >=3.10`; on this machine,
`python3.11` is suitable while bare `python3` may be older.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,notebooks]"
python -m ipykernel install --user --name weather-sampler-research --display-name "weather-sampler-research (.venv)"
python -m pytest
```

In VS Code/Jupyter, select the `weather-sampler-research (.venv)` kernel for synthetic toy
notebooks and baseline reports.

The repository and report were assessed against the pinned environment in
[`requirements.txt`](requirements.txt) (numpy 2.4.6 / scipy 1.17.1 / matplotlib 3.10.9 /
pytest 9.0.3). For an exact clean-clone reproduction, prefer
`pip install -r requirements.txt` over the floor-only `pyproject.toml` ranges; seeded
results then reproduce bitwise.

For real WeatherGenerator checkpoints, use the existing WeatherGenerator environment so
`torch` and GPU-related dependencies stay owned by that project. To keep that environment
stable, either install this package without extras:

```bash
source /users/harvey_bermingham/WeatherGenerator/.venv/bin/activate
python -m pip install -e .
```

If the WeatherGenerator environment is not already listed as a notebook kernel and it already
has `ipykernel`, register it with:

```bash
python -m ipykernel install --user --name weathergen --display-name "WeatherGenerator"
```

or avoid installing into it and add the source path only for the command you need:

```bash
PYTHONPATH=/users/harvey_bermingham/weather-sampler-research/sampler_research/src \
python scripts/convert_real_gmm_pt_to_npz.py
```

Only the `.pt` → `.npz` conversion step requires `torch` (which is intentionally not a
dependency of this small research package); everything downstream — including the data audit
`scripts/audit_real_gmm.py` and the Phase 4 runner — reads the converted `.npz` with the local
research venv. (`scripts/real_output_diagnostics.py` is archived: it targeted the wiped
`e3fz467m` checkpoint and is superseded by `scripts/audit_real_gmm.py`.)

Note: the upstream extraction that *creates* those `.pt` files (run inside the
WeatherGenerator repo, not this one — `gmm_inference.py`) needs the
`WEATHERGEN_PRIVATE_CONF` environment variable set to the private config path; see
[`research_notes/engineering_log.md`](research_notes/engineering_log.md). This repo's
conversion reads an already-extracted `.pt` and does **not** need that variable.
