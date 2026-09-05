# PTB-XL Challenge

ECG classification work against the [PTB-XL dataset](https://physionet.org/content/ptb-xl/), developed mostly in
numbered Jupyter notebooks that import reusable signal-processing, feature-extraction, and
experiment-tracking logic from the `ptbxl` package in `src/`.

## Quickstart

```bash
uv sync --all-extras       # creates .venv, installs the pinned environment (see uv.lock)
bash scripts/data_download.sh  # download PTB-XL records100 + metadata into data/raw (~1.8 GB, resumable)
uv run pytest               # run the test suite
uv run invoke lab           # launch Jupyter Lab
```

See [docs/install.md](docs/install.md) for the full setup guide, including how the data
download works and notebook diffing (`nbdime`)/Plotly/JupyterLab extension setup.

## What's implemented

The `ptbxl` package separates framework-agnostic ML core logic (grouped by pipeline stage)
from the one MLOps-specific adapter (MLflow), composed by a thin `pipelines/` layer:

- **`data/`** — dataset loading (`ingestion.py`, wraps `wfdb`), validation (`validation.py`),
  and PTB-XL's own `strat_fold`-based train/val/test split (`splitting.py`).
- **`preprocessing/`** — mean removal, baseline wander removal, band-pass filtering, a
  Pan-Tompkins pipeline, and min-max normalization (`filtering.py`), composed into config-driven
  stages in `pipeline.py`.
- **`features/`** — power spectral density, dominant frequency, spectral entropy, and
  per-wave (P/QRS/T) spectral descriptors (`spectral.py`); time-domain and heart-rate features
  (`temporal.py`); wavelet decomposition features (`wavelet.py`); composed in `extraction.py`.
- **`selection/`** — sklearn-based consensus feature selection (`selectors.py`, `selection.py`).
- **`modeling/`** — a config-driven model factory (`models.py`), training/hyperparameter search
  (`optimize.py`), prediction wrappers (`predict.py`), and MLflow-backed experiment tracking
  (`tracking.py` — SQLite store + local artifact directory under `tracking/`).
- **`evaluation/`** — pure classification-metric computation (`metrics.py`) and an `evaluate()`
  entrypoint (`evaluation.py`).
- **`pipelines/`** — composes the above into `feature_engineering_pipeline(config)` and
  `training_pipeline(config)`, driven by `config/dev.yml` / `config/prod.yml`.
- **`utils/paths.py`** — project-root-relative path helpers (`data_dir()`, `tracking_dir()`,
  etc.), so nothing hardcodes paths relative to the current working directory.

## What's still a stub

`src/ptbxl/visualization/visualize.py` and `app/main.py` are currently empty — the actual
app entry point hasn't been built yet.

## Project organization

See the full directory tree and description in [docs/project_structure.md](docs/project_structure.md).

## Contributing

No formal contribution process yet — this is a personal challenge project.
