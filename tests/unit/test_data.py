import numpy as np
import pandas as pd
import pytest

from ptbxl.data.ingestion import Dataset, build_target
from ptbxl.data.splitting import split_dataset
from ptbxl.data.validation import validate_dataset


def _sample_metadata():
    return pd.DataFrame(
        {
            "scp_codes": [{"NORM": 100.0}, {"AFIB": 80.0}, {"NORM": 100.0}],
            "static_noise": [np.nan, np.nan, 1.0],
            "burst_noise": [np.nan, np.nan, np.nan],
            "baseline_drift": [np.nan, np.nan, np.nan],
            "electrodes_problems": [np.nan, np.nan, np.nan],
        },
        index=pd.Index([1, 2, 3], name="ecg_id"),
    )


def test_build_target_labels_and_drops_quality_flagged_records():
    metadata = _sample_metadata()
    rhythm_codes = {"AFIB"}

    result = build_target(metadata, rhythm_codes)

    # Record 3 (NORM, but quality-flagged) is dropped.
    assert list(result.index) == [1, 2]
    assert result.loc[1, "target"] == 0.0  # NORM, no rhythm code
    assert result.loc[2, "target"] == 1.0  # AFIB is a rhythm code


def test_validate_dataset_passes_on_well_formed_dataset():
    metadata = pd.DataFrame({"target": [0.0, 1.0, 0.0]})
    signals = np.random.default_rng(0).standard_normal((3, 10, 2))

    dataset = validate_dataset(Dataset(metadata=metadata, signals=signals))

    assert len(dataset.metadata) == 3


@pytest.mark.parametrize(
    "metadata_kwargs,signals_shape",
    [
        ({"target": [0.0, 1.0]}, (3, 10, 2)),  # shape mismatch
        ({"target": [np.nan, 1.0, 0.0]}, (3, 10, 2)),  # NaN target
        ({"target": [0.0, 0.0, 0.0]}, (3, 10, 2)),  # single class
        ({"other": [0.0, 1.0, 0.0]}, (3, 10, 2)),  # missing target column
    ],
)
def test_validate_dataset_raises_on_bad_input(metadata_kwargs, signals_shape):
    metadata = pd.DataFrame(metadata_kwargs)
    signals = np.random.default_rng(0).standard_normal(signals_shape)

    with pytest.raises(ValueError):
        validate_dataset(Dataset(metadata=metadata, signals=signals))


def test_validate_dataset_raises_on_nan_signals():
    metadata = pd.DataFrame({"target": [0.0, 1.0, 0.0]})
    signals = np.random.default_rng(0).standard_normal((3, 10, 2))
    signals[0, 0, 0] = np.nan

    with pytest.raises(ValueError):
        validate_dataset(Dataset(metadata=metadata, signals=signals))


def test_split_dataset_partitions_by_strat_fold():
    n = 30
    metadata = pd.DataFrame({"strat_fold": [(i % 10) + 1 for i in range(n)]})
    X = pd.DataFrame({"f0": np.arange(n)})
    y = pd.Series(np.arange(n) % 2)

    splits = split_dataset(X, y, metadata)

    assert len(splits.X_train) == 24  # folds 1-8, 3 records each
    assert len(splits.X_val) == 3  # fold 9
    assert len(splits.X_test) == 3  # fold 10
    assert splits.cv.get_n_splits() == 8
