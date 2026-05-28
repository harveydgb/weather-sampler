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
- **Phase 1:** define and generate the controlled 8x8 single-channel synthetic GMM toy.
  Canonical spec: [`research_notes/phase_1.md`](research_notes/phase_1.md).
- **Phase 1.5:** generate harder toy variants, including heteroscedastic sigma and
  non-uniform regime-boundary mixture weights.
- **Phase 2:** choose and implement the core spatial sampling methodology. Canonical spec:
  [`research_notes/phase_2.md`](research_notes/phase_2.md). Detailed method investigation:
  [`research_notes/phase_2_research_plan.md`](research_notes/phase_2_research_plan.md).
- **Phase 3:** evaluate sampler outputs with GMM NLL, declared-prior coherence diagnostics,
  and baselines. The evaluation plan currently lives in
  [`research_notes/large_notes.md`](research_notes/large_notes.md).
- **Phase 4:** scale the chosen method toward real WeatherGenerator outputs and native grid
  geometry.

## Current Status

- Phase 1 toy generation and diagnostics are implemented.
- Generated Phase 1 / Phase 1.5 `.npz` files live under `outputs/data/`.
- Phase 2 methodology is specified and being narrowed to primary sampler candidates and
  critical baselines.

## Layout

- `sampler_research/src/sampler_research/`: reusable Python helpers for toy GMMs, sampling, diagnostics, and I/O.
- `sampler_research/tests/`: focused tests for the research helpers.
- `scripts/`: runnable experiment and diagnostic scripts.
- `notebooks/`: exploratory notebooks.
- `research_notes/`: phase notes, meeting notes, and research planning.
- `outputs/`: generated data, figures, and run artifacts. Large/generated files are ignored by git.

## Generated Outputs

Current sampler-facing synthetic data files:

- `outputs/data/phase_1_field.npz`
- `outputs/data/phase_1_field_heteroscedastic_sigma.npz`
- `outputs/data/phase_1_regime_boundary_pi.npz`

Sampler code should consume only `pi`, `mu`, `sigma`, and `coords` from these files. Extra
arrays such as `truth`, `component_fields`, `perm`, and variant metadata are for debugging and
reproducibility.

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
python scripts/real_output_diagnostics.py
```

The real-checkpoint diagnostics require `torch`, which is intentionally not a dependency of
this small research package.
