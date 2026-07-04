# Reproduce the Chapter-5 forecast-regime results, in dependency order.
# Each step reads the previous step's persisted artifacts under outputs/.
# Run on a machine that has the converted per-lead .npz present (or run
# `make forecast-leads` first, which needs the WeatherGenerator venv + the .pt).
#
#   make ch5        # softening -> faithfulness -> per-lead figures -> emit macros
#   make test       # the regression suite (203 passed, 1 skipped)
#   make report        # build report/thesis.pdf
#   make summary       # build report/summary.pdf from executive-summary/
#   make report-watch  # keep report/thesis.pdf updated while editing thesis.tex
#
# The emitter refuses to run when the forecast artifacts are absent (so a stray
# run cannot clobber the committed macros); regenerate the artifacts first.

PY := .venv/bin/python

.PHONY: ch5 forecast-leads softening faithfulness figures toy-figures emit test report report-watch summary summary-watch summary-clean summary-texcount

## Full per-lead Phase 4 audit for both checkpoints + the converged replicate
## inits. Needs ~/model_outputs/*.pt and the WeatherGenerator venv (torch).
forecast-leads:
	$(PY) scripts/run_phase4_forecast_leads.py --prefix phase_4_fc48_6ep  --figures
	$(PY) scripts/run_phase4_forecast_leads.py --prefix phase_4_fc48_14ep --figures \
		--forecast-pt ~/model_outputs/gmm_params_gmm_fc48_v2_me7_2t_f8.pt

softening:
	$(PY) scripts/run_forecast_softening.py

faithfulness:
	$(PY) scripts/run_forecast_faithfulness.py

## The two Phase 1/2 toy report figures, from the persisted stage_* artifacts
## (no sweep re-run; replaces the old notebook 02/03 exports).
toy-figures:
	$(PY) scripts/make_toy_figures.py

emit:
	$(PY) scripts/emit_report_results.py

## The Chapter-5 reporting chain (assumes the per-lead runs already landed).
ch5: softening faithfulness emit
	@echo "[ch5] softening + faithfulness CSVs and report macros/tables regenerated"

test:
	$(PY) -m pytest -q

report:
	$(MAKE) -C report/construction all

report-watch:
	$(MAKE) -C report/construction watch

summary:
	$(MAKE) -C executive-summary all

summary-watch:
	$(MAKE) -C executive-summary watch

summary-clean:
	$(MAKE) -C executive-summary clean

summary-texcount:
	$(MAKE) -C executive-summary texcount
