"""Pipeline A: preprocessing -> feature engineering -> feature selection."""

import pandas as pd

from ptbxl.config import Config
from ptbxl.data.ingestion import DEMOGRAPHIC_COLUMNS, load_dataset
from ptbxl.data.validation import validate_dataset
from ptbxl.features.extraction import extract_features
from ptbxl.preprocessing.stages import derive_qrs_envelope, preprocess_ecg
from ptbxl.selection.selection import select_features


def feature_engineering_pipeline(
    config: Config,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Loads and validates the dataset, extracts features, and picks a consensus subset.

    Demographics (`age`/`sex`) are concatenated in after selection, unconditionally --
    matching notebook 04, they aren't subject to the consensus vote themselves.

    Returns
    -------
    X : pd.DataFrame
        Selected signal features + demographics, indexed like `metadata` (ecg_id).
    y : pd.Series
        Binary target, same index.
    metadata : pd.DataFrame
        Full record metadata (e.g. for `data.splitting.split_dataset`'s `strat_fold`).
    """
    dataset = load_dataset(config.data)
    dataset = validate_dataset(dataset)

    fs = config.data.sampling_rate
    signal = preprocess_ecg(dataset.signals, fs, config.preprocessing)
    qrs_envelope = (
        derive_qrs_envelope(signal, fs, config.preprocessing)
        if config.features.temporal
        else None
    )

    features = extract_features(signal, fs, config.features, qrs_envelope=qrs_envelope)
    features.index = dataset.metadata.index

    y = dataset.metadata["target"].astype(int)
    selected = select_features(features, y, config.selection)

    demographics = dataset.metadata[DEMOGRAPHIC_COLUMNS]
    X = pd.concat([demographics, features[selected]], axis=1)

    return X, y, dataset.metadata
