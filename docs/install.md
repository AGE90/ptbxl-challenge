# PTB-XL Challenge Installation Guide

Welcome to the **PTB-XL Challenge** installation guide! This guide will walk you through setting up the environment, installing necessary dependencies, and configuring essential tools to ensure a smooth development experience.

---

## Prerequisites

Make sure you have the following installed before proceeding:

- **Python**: Version >= 3.9

---

## 1. Create and Activate a Virtual Environment

Navigate to your project directory and create a virtual environment:

```bash
python -m venv .venv
```

### Activate the virtual environment

#### On Unix/MacOS

```bash
source .venv/bin/activate
```

#### On Windows

```bash
. .venv/Scripts/activate
```

Once activated, your shell prompt should change to indicate that you're working inside the virtual environment.

---

## 2. Install Required Packages

First, upgrade `pip` to the latest version:

```bash
python -m pip install --upgrade pip
```

Then, install the required packages from `requirements.txt`:

```bash
pip install -r requirements.txt --no-cache-dir
```

### Install Jupyter and JupyterLab (Optional)

If you plan to use Jupyter or JupyterLab, install them with the following commands:

```bash
pip install jupyter
pip install jupyterlab
```

Your project dependencies are now installed within the virtual environment.

**Note:** The following sections assume that your virtual environment is active.

---

## 3. Set Up Project’s Module

To move beyond notebook prototyping, reusable code should reside in the `ptbxl/` package. To work with this package in development mode, you can install it in **editable** mode. This allows you to make changes to the module and use them without reinstalling the package.

Run the following command in your project root:

```bash
pip install -e .[dev]
```

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

## 4. Set Up Git Diff for Jupyter Notebooks

To efficiently manage and track changes in Jupyter notebooks, we recommend using **[nbdime](https://nbdime.readthedocs.io/en/stable/index.html)** for diffing and merging.

### Install nbdime

```bash
pip install nbdime
```

### Configure Git for nbdime

```bash
nbdime config-git --enable
```

### Enable nbdime extensions

To enable the Jupyter extensions for diffing notebooks:

```bash
nbdime extensions --enable --sys-prefix
```

Alternatively, if you need more granular control, you can manually enable the extensions with:

```bash
jupyter serverextension enable --py nbdime --sys-prefix
jupyter nbextension install --py nbdime --sys-prefix
jupyter nbextension enable --py nbdime --sys-prefix
jupyter labextension install nbdime-jupyterlab
```

If needed, rebuild the JupyterLab extensions with:

```bash
jupyter lab build
```

---

## 5. Set Up Plotly for JupyterLab

Plotly requires some additional steps to work correctly with JupyterLab.

### Install Required Extensions

Run the following commands to install the necessary JupyterLab extensions for Plotly:

```bash
jupyter labextension install @jupyter-widgets/jupyterlab-manager@0.36 --no-build
jupyter labextension install plotlywidget@0.2.1 --no-build
jupyter labextension install @jupyterlab/plotly-extension@0.16 --no-build
jupyter lab build
```

**Note:** There can be version conflicts between JupyterLab and Plotly extensions, so always check the [latest Plotly documentation](https://github.com/plotly/plotly.py#installation-of-plotlypy-version-3) to ensure compatibility.

---

## 6. Managing Project Tasks with Invoke

We use **[Invoke](http://www.pyinvoke.org/)** as a task runner for common project management tasks. You can view available tasks and manage them from a single entry point.

### List Available Tasks

```bash
invoke -l
```

For example, you might see:

```text
Available tasks:

  lab     Launch Jupyter lab
```

### Get Help on a Specific Task

```bash
invoke --help lab
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

- Ensure that your virtual environment is activated when running any project-related commands.
- Explore `tasks.py` to customize and extend the task automation for your needs.
  
You're now all set to start developing with **PTB-XL Challenge**!
