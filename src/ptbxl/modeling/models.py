"""Config-driven model factory (see `ptbxl.config.TrainingConfig`)."""

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from ptbxl.config import TrainingConfig


def get_model(config: TrainingConfig, y: pd.Series | None = None) -> BaseEstimator:
    """Builds an unfitted estimator for `config.model`.

    Parameters
    ----------
    config : TrainingConfig
    y : pd.Series, optional
        Training labels; when given, xgboost gets `scale_pos_weight` set to the
        negative/positive class ratio (matches notebooks 04/05's imbalance handling).
    """
    if config.model == "logistic_regression":
        return LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=config.random_state
        )

    if config.model == "xgboost":
        scale_pos_weight = None
        if y is not None:
            n_pos = (y == 1).sum()
            scale_pos_weight = (y == 0).sum() / n_pos if n_pos else None
        return XGBClassifier(
            random_state=config.random_state, scale_pos_weight=scale_pos_weight
        )

    raise NotImplementedError(f"unsupported model: {config.model}")
