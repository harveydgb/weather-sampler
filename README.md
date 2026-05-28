# Weather Sampler Research

Research code, notes, and diagnostics for sampler experiments around the WeatherGenerator project.

## Layout

- `sampler_research/src/sampler_research/`: reusable Python helpers for toy GMMs, sampling, diagnostics, and I/O.
- `sampler_research/tests/`: focused tests for the research helpers.
- `scripts/`: runnable experiment and diagnostic scripts.
- `notebooks/`: exploratory notebooks.
- `research_notes/`: phase notes, meeting notes, and research planning.
- `outputs/`: generated data, figures, and run artifacts. Large/generated files are ignored by git.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
```

The real-checkpoint diagnostics require an environment with `torch` installed, but `torch` is not installed by default for this small research package.
