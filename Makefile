# Submission Makefile. The full pipeline (report build, forecast leads, figure
# regeneration) needs the large generated outputs and the LaTeX sources, which
# are not part of this repository — see README.md and docs/reproducing_the_model.md.
PY := python

.PHONY: test toy-data
test:
	$(PY) -m pytest -q
toy-data:
	$(PY) scripts/generate_toy_data.py
