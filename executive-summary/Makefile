SHELL := /bin/sh

SUMMARY := summary
MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
SUMMARY_ROOT := $(abspath $(MAKEFILE_DIR))
REPO_ROOT := $(abspath $(SUMMARY_ROOT)/..)
BUILD_DIR := $(SUMMARY_ROOT)/build
OUTPUT_PDF := $(REPO_ROOT)/report/summary.pdf
LATEXMK := latexmk
TEXCOUNT := texcount

LATEXMK_FLAGS := -pdf -interaction=nonstopmode -halt-on-error -file-line-error -f -outdir=$(BUILD_DIR)

.PHONY: all rebuild watch check-assets prepare-build clean texcount

all: check-assets prepare-build
	cd $(SUMMARY_ROOT) && $(LATEXMK) $(LATEXMK_FLAGS) $(SUMMARY).tex
	cp $(BUILD_DIR)/$(SUMMARY).pdf $(OUTPUT_PDF)

rebuild: check-assets prepare-build
	cd $(SUMMARY_ROOT) && $(LATEXMK) $(LATEXMK_FLAGS) -g $(SUMMARY).tex
	cp $(BUILD_DIR)/$(SUMMARY).pdf $(OUTPUT_PDF)

watch: check-assets prepare-build
	cd $(SUMMARY_ROOT) && $(LATEXMK) $(LATEXMK_FLAGS) -pvc $(SUMMARY).tex

check-assets:
	@test -f "$(REPO_ROOT)/report/construction/macros.tex" || { \
		printf '%s\n' "Missing report/construction/macros.tex."; \
		exit 2; \
	}
	@test -f "$(REPO_ROOT)/report/construction/macros-results.tex" || { \
		printf '%s\n' "Missing report/construction/macros-results.tex."; \
		exit 2; \
	}
	@command -v "$(LATEXMK)" >/dev/null 2>&1 || { \
		printf '%s\n' "Missing latexmk."; \
		printf '%s\n' "Install a LaTeX distribution with latexmk, or compile on Overleaf."; \
		exit 2; \
	}

prepare-build:
	mkdir -p "$(BUILD_DIR)"
	mkdir -p "$(REPO_ROOT)/report"

clean:
	cd $(SUMMARY_ROOT) && $(LATEXMK) -C -outdir=$(BUILD_DIR) $(SUMMARY).tex || true
	rm -f "$(OUTPUT_PDF)"
	rm -rf "$(BUILD_DIR)"
	mkdir -p "$(BUILD_DIR)"

texcount:
	@command -v "$(TEXCOUNT)" >/dev/null 2>&1 || { \
		printf '%s\n' "Missing texcount."; \
		printf '%s\n' "Install texcount, or add it to PATH, before recording the final executive-summary word count."; \
		exit 2; \
	}
	cd $(SUMMARY_ROOT) && $(TEXCOUNT) -inc -total $(SUMMARY).tex
