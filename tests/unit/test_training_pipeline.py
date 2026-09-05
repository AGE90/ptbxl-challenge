import numpy as np
import pandas as pd
import pytest

from ptbxl.config import Config, SelectionConfig, TrainingConfig
from ptbxl.data.ingestion import LEADS, Dataset
from ptbxl.modeling.optimize import TrainResult
from ptbxl.pipelines.training import training_pipeline


def _synthetic_dataset(n_per_fold_class: int = 2, n_samples: int = 200) -> Dataset:
    """~2 records per (fold, class) combination -> every strat_fold has both classes."""
    rng_demo = np.random.default_rng(1)
    rows = [
        {
            "strat_fold": fold,
            "target": target,
            "age": rng_demo.uniform(20, 80),
            "sex": rng_demo.integers(0, 2),
        }
        for fold in range(1, 11)
        for target in (0.0, 1.0)
        for _ in range(n_per_fold_class)
    ]
    metadata = pd.DataFrame(rows, index=pd.RangeIndex(len(rows), name="ecg_id"))

    rng = np.random.default_rng(0)
    signals = rng.standard_normal((len(rows), n_samples, len(LEADS)))

    return Dataset(metadata=metadata, signals=signals)


@pytest.fixture(autouse=True)
def _patch_load_dataset(monkeypatch):
    monkeypatch.setattr(
        "ptbxl.pipelines.feature_engineering.load_dataset",
        lambda config: _synthetic_dataset(),
    )


def test_training_pipeline_runs_end_to_end():
    config = Config(
        selection=SelectionConfig(max_features=3),
        training=TrainingConfig(model="logistic_regression"),
    )

    result, evaluation = training_pipeline(config)

    assert isinstance(result, TrainResult)
    assert hasattr(result.model, "predict")

    expected_metric_keys = {"precision", "recall", "f1", "roc_auc"}
    assert expected_metric_keys.issubset(evaluation.metrics)
    for value in evaluation.metrics.values():
        assert np.isfinite(value)
