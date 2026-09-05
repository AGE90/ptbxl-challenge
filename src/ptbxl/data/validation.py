"""Minimal sanity checks on a loaded `Dataset` before it enters the pipeline.

ponytail: shape/NaN/label-presence checks only -- add schema/range validation
when a real bad-data incident actually needs it.
"""

import numpy as np

from ptbxl.data.ingestion import Dataset, TARGET_COLUMN


def validate_dataset(dataset: Dataset) -> Dataset:
    """Raises `ValueError` if `dataset` is empty, misaligned, or has no target."""
    n_records = len(dataset.metadata)

    if n_records == 0:
        raise ValueError("dataset is empty")

    n_signals = dataset.signals.shape[0]
    if n_signals != n_records:
        raise ValueError(f"signals has {n_signals} records, metadata has {n_records}")

    if np.isnan(dataset.signals).any():
        raise ValueError("signals contains NaN values")

    if TARGET_COLUMN not in dataset.metadata.columns:
        raise ValueError(f"metadata is missing the '{TARGET_COLUMN}' column")

    if dataset.metadata[TARGET_COLUMN].isna().any():
        raise ValueError(f"'{TARGET_COLUMN}' contains NaN values")

    if dataset.metadata[TARGET_COLUMN].nunique() < 2:
        raise ValueError(f"'{TARGET_COLUMN}' has fewer than 2 classes")

    return dataset
