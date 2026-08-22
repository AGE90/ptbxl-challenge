# PTB-XL Challenge Installation Guide

Welcome to the **PTB-XL Challenge** installation guide! This guide will walk you through setting up the environment, installing necessary dependencies, and configuring essential tools to ensure a smooth development experience.

---

## Prerequisites

Make sure you have the following installed before proceeding:

- **[uv](https://docs.astral.sh/uv/)**: manages the Python version, the virtual environment, and all dependencies for this project.

---

## 1. Sync the Environment

Navigate to your project directory and let `uv` create the virtual environment and install every dependency (runtime, dev, and notebook tooling like Jupyter/JupyterLab):

```bash
uv sync --all-extras
```

This reads `.python-version` (pinned to 3.11), downloads that interpreter if needed, creates `.venv/`, and installs everything pinned in `uv.lock`. Re-run it any time `pyproject.toml` changes.

`uv run <command>` runs a command inside that environment without needing to `source .venv/bin/activate` first (all examples below use this form). If you prefer an activated shell, `source .venv/bin/activate` still works after `uv sync`.

---

## 2. Project's Module

`uv sync` already installs the `ptbxl` package itself in **editable** mode as part of the project's own dependencies, so changes to `src/ptbxl/` are picked up immediately — no separate `pip install -e .` step is needed.

### Use the Module Inside Jupyter Notebooks

To ensure that your changes in the `ptbxl` module are automatically reloaded in Jupyter notebooks, add `%autoreload` at the top of your notebook:

```python
%load_ext autoreload
%autoreload 2
```

### Example of Module Usage

```python
from ptbxl.utils.paths import data_dir
data_dir()
```

---

## 3. Set Up Git Diff for Jupyter Notebooks

To efficiently manage and track changes in Jupyter notebooks, we recommend using **[nbdime](https://nbdime.readthedocs.io/en/stable/index.html)** for diffing and merging. `nbdime` is already installed as part of `uv sync --all-extras`'s dev tooling; run it via `uv run`.

### Configure Git for nbdime

```bash
uv run nbdime config-git --enable
```

### Enable nbdime extensions

To enable the Jupyter extensions for diffing notebooks:

```bash
uv run nbdime extensions --enable --sys-prefix
```

Alternatively, if you need more granular control, you can manually enable the extensions with:

```bash
uv run jupyter serverextension enable --py nbdime --sys-prefix
uv run jupyter nbextension install --py nbdime --sys-prefix
uv run jupyter nbextension enable --py nbdime --sys-prefix
uv run jupyter labextension install nbdime-jupyterlab
```

If needed, rebuild the JupyterLab extensions with:

```bash
uv run jupyter lab build
```

---

## 4. Set Up Plotly for JupyterLab

Plotly requires some additional steps to work correctly with JupyterLab.

### Install Required Extensions

Run the following commands to install the necessary JupyterLab extensions for Plotly:

```bash
uv run jupyter labextension install @jupyter-widgets/jupyterlab-manager@0.36 --no-build
uv run jupyter labextension install plotlywidget@0.2.1 --no-build
uv run jupyter labextension install @jupyterlab/plotly-extension@0.16 --no-build
uv run jupyter lab build
```

**Note:** There can be version conflicts between JupyterLab and Plotly extensions, so always check the [latest Plotly documentation](https://github.com/plotly/plotly.py#installation-of-plotlypy-version-3) to ensure compatibility.

---

## 5. Managing Project Tasks with Invoke

We use **[Invoke](http://www.pyinvoke.org/)** as a task runner for common project management tasks. You can view available tasks and manage them from a single entry point. `make lab` / `make notebook` wrap the most common ones (see the project [Makefile](../Makefile)); for anything else, run `invoke` directly via `uv run`.

### List Available Tasks

```bash
uv run invoke -l
```

For example, you might see:

```text
Available tasks:

  lab     Launch Jupyter lab
```

### Get Help on a Specific Task

```bash
uv run invoke --help lab
```

The output might look like:

```text
Usage: inv[oke] [--core-opts] lab [--options] [other tasks here ...]

Docstring:
  Launch Jupyter Lab.

Options:
  -i STRING, --ip=STRING   IP to listen on, defaults to *
  -p, --port               Port to listen on, defaults to 8888
```

### Adding Custom Tasks

To add your own tasks, edit the `tasks.py` file. This file contains the definition of each task. You can create custom tasks based on your project's requirements.

---

### Final Notes

- Prefix project commands with `uv run` (or activate `.venv` first) so they use the project's environment.
- Explore `tasks.py` to customize and extend the task automation for your needs.

You're now all set to start developing with **PTB-XL Challenge**!
