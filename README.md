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

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
```

The real-checkpoint diagnostics require an environment with `torch` installed, but `torch` is not installed by default for this small research package.
