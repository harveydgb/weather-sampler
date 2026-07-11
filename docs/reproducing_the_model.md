# Reproducing the WeatherGenerator GMM model and its emitted mixtures

This manual documents the parts of the pipeline that live **outside this repository** — the
WeatherGenerator model, the ERA5 data, the training runs, and the extraction step that
produces the per-location Gaussian-mixture files this repo consumes. It is written for
someone trying to reproduce the results in the report from scratch.

The boundary is simple:

- **This repository (`weather-sampler-research`)** contains the *sampler* — the methods that
  assemble the emitted per-location mixtures into a single coherent field, plus all
  evaluation and figures. It takes the mixture `.pt`/`.npz` files as fixed input.
- **The WeatherGenerator model** produces those mixtures. It is a *separate* codebase on a
  CSCS HPC system, is not redistributed here, and is what this manual covers.

Everything below is procedure.

---

## 0. What you need before you start (out-of-repo parts and how to access them)

Reproducing the model requires four things that are **not** in this repository:

| Part | What it is | How to access |
| --- | --- | --- |
| **WeatherGenerator code, GMM branch** | The foundation weather model (ECMWF), on the branch that adds a Gaussian-mixture prediction head. | Fork `https://github.com/ecmwf/WeatherGenerator` (project site `https://weathergenerator.eu`) and check out the GMM branch `sophiex/dev/mae-with-gmms`. This is the branch all runs below were trained on. |
| **WeatherGenerator-private** | HPC platform config: data paths, uenv image, and object-store credentials. Kept private because it holds secrets. | Obtain from the WeatherGenerator maintainers. The only file this manual touches is `hpc/alps-clariden/config/paths.yml`. |
| **ERA5 store** | The training/validation data (see §2). | Hosted on CSCS `capstor` under the `a122` allocation: `/capstor/store/cscs/swissai/a122/anemoi/aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8.zarr`. Requires a CSCS account with access to that allocation. |
| **Compute** | A GPU node on CSCS Alps **Clariden**. | CSCS account, SLURM account `a122`. Training used one node (`gpu:4`); extraction uses a single GPU. |

The **output** of this manual — the emitted mixture files `gmm_params_*.pt` — is the handoff
point back into this repository, where `scripts/convert_real_gmm_pt_to_npz.py` converts them
for the sampler.

---

## 1. Fork and branch

1. Fork the WeatherGenerator repository: `https://github.com/ecmwf/WeatherGenerator`
   (project site `https://weathergenerator.eu`).
2. Check out the GMM branch: `sophiex/dev/mae-with-gmms`. This branch supplies the
   Gaussian-mixture prediction head (`num_components: 4`) and the `train` console entry point
   used below. All configuration paths in this manual are relative to the WeatherGenerator
   repo root.
3. Build the environment with the GPU extra (the `train`/inference entry points need it):

   ```bash
   uv sync --extra gpu        # produces .venv/bin/train
   ```

---

## 2. The data: ERA5

- **Dataset:** ERA5 reanalysis in `anemoi` zarr form,
  `aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8.zarr`.
- **Grid:** native **O96 reduced Gaussian grid** — 40,320 locations over 192 Gaussian rings.
  The model emits, and this study evaluates, on exactly this grid; **no regridding** is done
  anywhere.
- **Cadence and span:** 6-hourly, 1979-01-01 → 2023-12-31 (0 missing dates).
- **Variable:** the single surface channel `2t` (2-metre temperature), standardised per
  channel to zero mean and unit variance. De-standardisation constants (`norm_mean`,
  `norm_std`) are carried in the emitted files.
- **Stream config:** `config/streams/era5_1deg/era5.yml` — `type: anemoi`, `token_size: 8`,
  with the GMM head (`num_components: 4`) attached to the ERA5 stream.

**Splits** (identical in both training configs):

- **Training window:** `1979-01-01T00:00` → `2022-12-31T00:00`.
- **Validation window:** `2023-10-01T00:00` → `2023-12-31T00:00`.

Because training ends at the close of 2022, **all of 2023 is leakage-clean**; the Oct–Dec
validation window is a configuration convention, not a data limit. This matters for §5:
forecast initialisations anywhere in 2023 can be scored without leakage (see the
`GMM_VAL_START_DATE` override in §5).

---

## 3. Required changes to the fork (the manual part)

These are the changes needed to get the GMM head to train and emit on Clariden. They are
listed as reproduction steps, each with the symptom and root cause it fixes. None of them
changes the science — they are platform and numerical-plumbing fixes.

### 3.1 Numerical stability in the GMM loss

Two changes in the loss/model path, both required for training to converge rather than diverge
to `NaN`:

- **Loss-counter guard.** In `src/weathergen/train/loss_modules/loss_module_physical.py`, the
  averaging guards were written for a non-negative MSE loss (`if loss > 0.0`). `gmm_nll` is a
  log-density and goes **negative** on success; the guard must be `!= 0.0`, and the final
  division guarded as `loss / max(ctr_streams, 1)`. Without this, the first sub-zero batch
  produces `negative / 0 = -INF → NaN` weights.
- **Component spread floor.** In `src/weathergen/model/model.py`, set `GMM_MIN_SCALE = 1e-2`
  (~0.2 K for `2t`), replacing the `1e-6` literals in `GaussianMixtureDiag(...)` and
  `params_from_raw(...)`. This stops a component collapsing toward a delta spike, where the
  NLL gradient (`~1/σ³`) explodes.

### 3.2 Self-contained SLURM launchers

The shared `launch-slurm.py` / `weathergen_slurm.sh` path does not work on this branch (it
calls a removed `run_train.py train` positional subcommand and triggers an MLflow
`ConfigKeyError`). Use the self-contained launchers in `scripts/` instead — they call the
`train` entry point directly and bundle the platform fixes below:

- `scripts/launch_gmm_training.sh` — training.
- `scripts/launch_gmm_inference.sh` + `scripts/gmm_inference.py` — extraction.

These launchers already contain:

- **`WEATHERGEN_PRIVATE_CONF` export** — `platform-env.py` host detection only knows Clariden
  `nid006*` nodes and dies on newer `nid007*` nodes; exporting the private-config path
  bypasses host detection at every config call site.
- **uenv session unset** — `unset ${!UENV@}` before `uenv run`, so a nested uenv session does
  not refuse to start.
- **Options baked into the inner script** — SLURM `--export` truncates values at the first
  space, so the full option string is expanded in the launcher, not passed through the
  environment.
- **`--time 12:00:00`** — the `normal` partition caps walltime at 12 h.

### 3.3 Private config

`WeatherGenerator-private/hpc/alps-clariden/config/paths.yml` must define the data paths and a
`secrets:` block (object-store credentials for data access). Point `WEATHERGEN_PRIVATE_CONF`
at this file. **Do not commit credentials in plaintext** — move them to a real secret store
before sharing.

---

## 4. Training

The model is trained **end-to-end as a single network, with nothing frozen**
(`freeze_modules: ""` in both configs). There are two runs: an autoencoder from scratch, then
a forecast model warm-started from it. Both run on one Clariden node.

### 4.1 Autoencoder (masking reconstruction) — `default_config.yml`

- **Init:** from scratch (`load_chkpt: {}`).
- **Task:** masking autoencoder. HEALPix masking at level 5, `rate: 0.64`, `hl_mask: 4`;
  `forecast.num_steps` is 0, so the emitted mixtures are a **step-0 reconstruction** (this is
  why the autoencoder's weights are near one-hot — expected, not a bug).
- **Schedule:** `num_mini_epochs: 32`, `samples_per_mini_epoch: 4096`; LR `1e-6 → 1e-5` with
  512-step warmup and cooldown; loss `gmm_nll`.

Launch:

```bash
bash scripts/launch_gmm_training.sh gmm_era5_32ep_v3
```

This run reached mini-epoch 31 (~11.7 h) with a healthy **negative** validation NLL (≈ −0.40)
before an NCCL watchdog fired on teardown; judge success by the last saved checkpoint
(`chkpt00031`), which is the autoencoder used below. (The NCCL multi-rank teardown timeout is
a known open issue — one rank stalls on a collective at the mini-epoch boundary and the
600 s watchdog tears the whole job down — but it fires *after* the checkpoint is written,
so it does not affect the artifact.)

### 4.2 Forecast (+48 h rollout) — `gmm_forecast_config.yml`

- **Init:** warm-started from the autoencoder (`load_chkpt.run_id: gmm_era5_32ep_v3`,
  `mini_epoch: 31`; the forecast-engine blocks initialise fresh).
- **Rollout:** `training_config.forecast` with `offset: 1`, `time_step: 06:00:00`,
  `num_steps: 8` → leads **+6 h … +48 h**.
- **Schedule:** same data windows; `samples_per_mini_epoch: 4096`; loss `gmm_nll`.

The forecast checkpoint used in the report was produced in **two runs** (a node failure ended
the first):

```bash
# Run 1: warm-start from the autoencoder. Submitted as 16 mini-epochs; a node
# failure ended it during mini-epoch 6, leaving me5 as the last usable checkpoint
# (gmm_fc48_v1 me5 = 6 mini-epochs) — the "6-epoch" column in the report.
GMM_BASE_CONFIG=config/gmm_forecast_config.yml \
bash scripts/launch_gmm_training.sh gmm_fc48_v1

# Run 2: warm-start from run 1 (set load_chkpt.run_id=gmm_fc48_v1, mini_epoch=5),
# +8 mini-epochs → gmm_fc48_v2 me7. Total 6 + 8 = 14 mini-epochs — the "14-epoch"
# (final) column. Validation NLL ~flat across the second run.
GMM_BASE_CONFIG=config/gmm_forecast_config.yml \
GMM_EXTRA_OPTS="load_chkpt.run_id=gmm_fc48_v1 load_chkpt.mini_epoch=5" \
bash scripts/launch_gmm_training.sh gmm_fc48_v2
```

The two checkpoints (6-epoch `gmm_fc48_v1` me5 and 14-epoch `gmm_fc48_v2` me7) are the two
forecast columns compared in the report. The 14 mini-epochs are two separate runs, each with
its own LR warmup/cooldown — a `load_chkpt` weight load, not a single continuous trajectory.

---

## 5. Extraction (running inference to emit the mixtures)

Extraction runs the trained model once and saves the per-location mixtures. It is single-GPU
and takes ~1 minute per initialisation.

```bash
GMM_FORECAST_STEPS=8 \
GMM_TARGET_DATETIME=2023-11-01T00:00 \
GMM_OUTPUT_PATH=/path/to/gmm_params_gmm_fc48_v2_me7_2t_f8.pt \
bash scripts/launch_gmm_inference.sh gmm_fc48_v2 7
```

Key points, all handled by `scripts/gmm_inference.py`:

- **Full-grid prediction in one pass.** The target mask uses **keep-rate 1.0**, so every
  HEALPix cell is a prediction target and a single forward pass yields all 40,320 locations.
  (`rate: 0.0` would keep *zero* targets and predict nothing.)
- **Source masking.** For step-0 autoencoder extraction, forcing full-grid targets would make
  the default complement source empty, so the source is generated independently as
  `healpix(hl_mask=4, rate=0.36)` via
  `target_source_correspondence = {0: {0: "independent"}}`. Forecast extraction keeps the
  forecast source setup from training and only forces the targets to the full grid.
- **Forecast rollout.** `--forecast-steps 8` rolls the model out 8 steps from the
  initialisation; step *k* = lead +6*k* h. Set `--forecast-steps 0` to emit a step-0
  reconstruction from an autoencoder checkpoint.

**Output file** `gmm_params_<run>_me<k>_2t_f8.pt` (~717 MB) is a nested dict containing, per
lead: mixture weights `pi [N, K]`, means `mu [N, K, C]`, spreads `sigma [N, K, C]`,
selected-channel `2t` marginals `[N, K]`, the target `latlons`, `valid_datetime`, and the
de-standardisation constants `norm_mean` / `norm_std`. Each emitted file was audited
before use: no NaNs or infinities, component spreads in a physical range, and mixture
weights summing to one at single precision.

### Scoring initialisations outside the Oct–Dec window

The validation window is **baked into the checkpoint**, so editing the YAML does nothing.
To score initialisations elsewhere in 2023 (e.g. the 12 first-of-month inits used for the
across-init robustness check), set the env-gated override — unset, behaviour is byte-identical:

```bash
GMM_VAL_START_DATE=2023-01-01T00:00 \
GMM_TARGET_DATETIME=2023-03-01T00:00 \
GMM_FORECAST_STEPS=8 \
GMM_OUTPUT_PATH=/path/to/out.pt \
bash scripts/launch_gmm_inference.sh gmm_fc48_v2 7
```

---

## 6. Handoff back to this repository

Convert the emitted `.pt` into the per-lead marginal files the sampler reads:

```bash
python scripts/convert_real_gmm_pt_to_npz.py   # WeatherGenerator .pt -> phase_4_fc48_step{k}_2t.npz
```

From here the sampler methods, evaluation, and figures are entirely within this repository;
see [`README.md`](../README.md). Everything downstream is deterministic given the emitted
files: the regularised-MAP restarts derive from a fixed base seed, the independent-draw
baseline uses a fixed seed list, and extraction itself is deterministic.

---

## Summary of runs

| Run | Config | Init | Schedule | Checkpoint used |
| --- | --- | --- | --- | --- |
| Autoencoder | `default_config.yml` | from scratch | 32 mini-epochs (reached 31), masking level-5 rate 0.64 | `gmm_era5_32ep_v3` me31 |
| Forecast run 1 | `gmm_forecast_config.yml` | warm-start from AE me31 | ended by node failure at mini-epoch 6 | `gmm_fc48_v1` me5 (6-epoch column) |
| Forecast run 2 | `gmm_forecast_config.yml` | warm-start from run 1 me5 | +8 mini-epochs | `gmm_fc48_v2` me7 (14-epoch, final) |

All numbers the report quotes from these artifacts are emitted and regression-tested by
`scripts/emit_report_results.py`.
