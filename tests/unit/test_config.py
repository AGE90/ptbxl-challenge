from pathlib import Path

from ptbxl.config import load_config


def test_load_config_round_trips_values(tmp_path: Path):
    yaml_text = """
    data:
      sampling_rate: 250
    preprocessing:
      lowcut: 1.0
      highcut: 40.0
    selection:
      max_features: 5
      sample_size: null
      methods: [relief, anova_f]
    training:
      model: logistic_regression
    optimization:
      n_trials: 7
      metric: f1
    """
    config_path = tmp_path / "test.yml"
    config_path.write_text(yaml_text)

    config = load_config(config_path)

    assert config.data.sampling_rate == 250
    assert config.preprocessing.lowcut == 1.0
    assert config.preprocessing.highcut == 40.0
    assert config.selection.max_features == 5
    assert config.selection.sample_size is None
    assert config.selection.methods == ["relief", "anova_f"]
    assert config.training.model == "logistic_regression"
    assert config.optimization.n_trials == 7
    assert config.optimization.metric == "f1"

    # Unspecified sections/fields fall back to dataclass defaults.
    assert config.features.spectral is True
    assert config.preprocessing.filter_order == 4


def test_load_config_defaults_on_empty_file(tmp_path: Path):
    config_path = tmp_path / "empty.yml"
    config_path.write_text("")

    config = load_config(config_path)

    assert config.data.sampling_rate == 100
    assert config.training.model == "xgboost"
