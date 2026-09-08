# Project structure

.
├── LICENSE
├── README.md              <- The top-level README for developers using this project.
├── CHANGELOG.md           <- A changelog to track project updates and versions.
├── pyproject.toml         <- Project configuration file (replaces setup.py and requirements.txt).
├── .gitignore             <- Specifies intentionally untracked files to ignore.
├── Makefile               <- Automate common tasks like testing, running, or setting up.
├── .env                   <- Environment variables (ignored by git).
├── .pre-commit-config.yaml <- Pre-commit hooks for linting/formatting.
├── app                    <- Main application code (if applicable).
│   └── main.py            <- Entry point for the application.
├── config                 <- Configuration files for the project.
│   ├── dev.yml            <- Development environment configuration.
│   └── prod.yml           <- Production environment configuration.
├── data
│   ├── external           <- Data from third party sources.
│   ├── interim            <- Intermediate data that has been transformed.
│   ├── processed          <- The final, canonical data sets for modeling.
│   ├── raw                <- The original, immutable data dump.
│   └── .dvc/              <- DVC meta-files for data tracking.
├── docs                   <- Project documentation.
│   ├── index.md           <- Main index or README for the documentation.
│   ├── project_structure.md    <- Project structure tree.
│   ├── install.md         <- Detailed instructions to set up this project.
│   ├── api.md             <- API documentation.
│   ├── user_guide.md      <- User guide for the project.
│   └── developer_guide.md <- Guide for developers contributing to the project.
├── logs                   <- Log files.
├── models                 <- Trained and serialized models, model predictions, or model summaries.
├── notebooks              <- Jupyter notebooks. Naming convention is a number (for ordering),
│                             the creator's initials, and a short `-` delimited description, e.g.
│                             `01-AGE90-initial_data_exploration`.
├── references             <- Data dictionaries, manuals, and all other explanatory materials.
├── reports                <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures            <- Generated graphics and figures to be used in reporting.
├── scripts                <- Utility scripts for project management, data processing, etc.
│   ├── data_download.sh   <- Script to download raw data.
│   └── setup_env.sh       <- Script to set up the development environment.
├── src
│   └── ptbxl  <- Source code for use in this project.
│       ├── __init__.py    <- Makes ptbxl a Python module.
│       ├── config.py      <- Typed config dataclasses + YAML loader for pipeline stages.
│       ├── data            <- Dataset loading, validation, and train/val/test splitting.
│       │   ├── ingestion.py
│       │   ├── validation.py
│       │   └── splitting.py
│       ├── preprocessing   <- Signal filtering + config-driven preprocessing stages.
│       │   ├── filtering.py
│       │   └── stages.py
│       ├── features        <- Feature extractors, grouped by domain, plus composition.
│       │   ├── spectral.py
│       │   ├── temporal.py
│       │   ├── wavelet.py
│       │   └── extraction.py
│       ├── selection       <- Consensus feature selection.
│       │   ├── selectors.py
│       │   └── selection.py
│       ├── modeling         <- Model factory, training/optimization, prediction, MLflow tracking.
│       │   ├── models.py
│       │   ├── optimize.py
│       │   ├── predict.py
│       │   └── tracking.py
│       ├── evaluation       <- Metric computation and evaluation entrypoint.
│       │   ├── metrics.py
│       │   └── evaluation.py
│       ├── pipelines        <- Composes the stages above into runnable pipelines.
│       │   ├── feature_engineering.py
│       │   └── training.py
│       ├── utils           <- Scripts to help with common tasks.
│       │   └── paths.py   <- Helper functions for relative file referencing across project.
│       └── visualization  <- Scripts to create exploratory and results oriented visualizations.
│           └── visualize.py
└── tests                  <- Test files should mirror the structure of `src`.
    ├── __init__.py
    ├── conftest.py        <- Shared pytest fixtures.
    ├── e2e/               <- End-to-end or integration tests.
    └── unit/              <- Unit tests, mirroring src structure.
