"""Typed config objects for pipeline stages, loaded from YAML (see `config/dev.yml`)."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    sampling_rate: int = 100
    trim_seconds: float = 1.0


@dataclass
class PreprocessingConfig:
    lowcut: float = 5.0
    highcut: float = 12.0
    filter_order: int = 4
    window_ms: int = 150


@dataclass
class FeaturesConfig:
    spectral: bool = True
    temporal: bool = True
    wavelet: bool = True


@dataclass
class SelectionConfig:
    method: str = "consensus"
    max_features: int = 10
    sample_size: int | None = 2000


@dataclass
class TrainingConfig:
    model: str = "xgboost"
    random_state: int = 42


@dataclass
class OptimizationConfig:
    n_trials: int = 20
    metric: str = "roc_auc"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)


def load_config(path: str | Path) -> Config:
    """Loads a YAML file shaped like `config/dev.yml` into a `Config`."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    return Config(
        data=DataConfig(**raw.get("data", {})),
        preprocessing=PreprocessingConfig(**raw.get("preprocessing", {})),
        features=FeaturesConfig(**raw.get("features", {})),
        selection=SelectionConfig(**raw.get("selection", {})),
        training=TrainingConfig(**raw.get("training", {})),
        optimization=OptimizationConfig(**raw.get("optimization", {})),
    )
