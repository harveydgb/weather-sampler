#!/usr/bin/env bash
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$SRC/../weather_sampler}"
mkdir -p "$DEST"

COMMON_EXCL=(--exclude='__pycache__/' --exclude='*.pyc' --exclude='*.egg-info/'
  --exclude='.pytest_cache/' --exclude='.ruff_cache/' --exclude='.mypy_cache/'
  --exclude='.ipynb_checkpoints/')

# library (src + tests)
rsync -a --delete "${COMMON_EXCL[@]}" "$SRC/src/" "$DEST/src/"
rsync -a --delete "${COMMON_EXCL[@]}" "$SRC/tests/" "$DEST/tests/"
# scripts, minus the dev throwaways / not-for-submission diagnostics.
# --delete-excluded so an excluded file is actively removed from DEST (a plain
# --exclude would protect an already-exported copy from --delete), keeping the
# scripts dir an exact function of the allowlist on every re-run.
rsync -a --delete --delete-excluded "${COMMON_EXCL[@]}" \
  --exclude='check_threshold_sensitivity.py' \
  --exclude='check_deep_tail_decomp.py' \
  --exclude='real_output_diagnostics.py' \
  --exclude='run_phase4_local_variance_test_sampler.py' \
  --exclude='export_submission.sh' \
  "$SRC/scripts/" "$DEST/scripts/"
rsync -a --delete "${COMMON_EXCL[@]}" "$SRC/notebooks/" "$DEST/notebooks/"
rsync -a --delete "$SRC/docs/" "$DEST/docs/"

# top-level single files
cp "$SRC/README.md" "$SRC/pyproject.toml" "$SRC/requirements.txt" "$SRC/.gitignore" "$DEST/"
cp "$SRC/Makefile.submission" "$DEST/Makefile"
[ -f "$SRC/LICENSE" ] && cp "$SRC/LICENSE" "$DEST/"
if [ -f "$SRC/.github/workflows/ci.yml" ]; then
  mkdir -p "$DEST/.github/workflows"; cp "$SRC/.github/workflows/ci.yml" "$DEST/.github/workflows/"
fi

# report: PDFs only
mkdir -p "$DEST/report"
cp "$SRC/report/thesis.pdf" "$SRC/report/summary.pdf" "$DEST/report/"

# outputs: data README + coastlines + curated figures
mkdir -p "$DEST/outputs/data" "$DEST/outputs/figures"
[ -f "$SRC/outputs/data/README.md" ] && cp "$SRC/outputs/data/README.md" "$DEST/outputs/data/"
rsync -a "$SRC/outputs/data/natural_earth/" "$DEST/outputs/data/natural_earth/"
while IFS= read -r f; do
  [ -z "$f" ] && continue
  case "$f" in \#*) continue;; esac
  mkdir -p "$DEST/outputs/figures/$(dirname "$f")"
  cp "$SRC/outputs/figures/$f" "$DEST/outputs/figures/$f"
done < "$SRC/submission_figures.txt"

echo "Export complete -> $DEST"
echo "This script ran NO git. Hand off to Harvey for git init/commit/push."
