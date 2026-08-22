# PTB-XL Challenge

ECG classification work against the [PTB-XL dataset](https://physionet.org/content/ptb-xl/), developed mostly in
numbered Jupyter notebooks that import reusable signal-processing, feature-extraction, and
experiment-tracking logic from the `ptbxl` package in `src/`.

## Quickstart

```bash
uv sync --all-extras   # creates .venv, installs the pinned environment (see uv.lock)
uv run pytest          # run the test suite
uv run invoke lab      # launch Jupyter Lab
```

See [docs/install.md](docs/install.md) for the full setup guide, including notebook diffing
(`nbdime`) and Plotly/JupyterLab extension setup.

## What's implemented

- **`SignalPreprocessing`** (`src/ptbxl/data/signal_preprocessing.py`) — mean removal, baseline
  wander removal, band-pass filtering, a Pan-Tompkins pipeline, and min-max normalization for
  `(records, samples, leads)` ECG arrays.
- **`BuildFeatures`** (`src/ptbxl/features/build_features.py`) — power spectral density,
  dominant frequency, spectral entropy, and wavelet-based features.
- **`ExperimentTracking`** (`src/ptbxl/models/tracking.py`) — MLflow-backed experiment tracking
  (SQLite store + local artifact directory under `tracking/`), logging params, metrics, a
  confusion matrix, and the trained model itself.
- **`src/ptbxl/data/make_dataset.py`** — loads raw PTB-XL waveform records via `wfdb`.
- **`src/ptbxl/utils/paths.py`** — project-root-relative path helpers (`data_dir()`,
  `tracking_dir()`, etc.), so nothing hardcodes paths relative to the current working directory.

## What's still a stub

`src/ptbxl/models/train_model.py`, `src/ptbxl/models/predict_model.py`,
`src/ptbxl/visualization/visualize.py`, and `app/main.py` are currently empty — the actual
training/prediction pipeline and app entry point haven't been built yet.

## Project organization

See the full directory tree and description in [docs/project_structure.md](docs/project_structure.md).

## Contributing

No formal contribution process yet — this is a personal challenge project.
