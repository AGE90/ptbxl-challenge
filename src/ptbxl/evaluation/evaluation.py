from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from ptbxl.evaluation.metrics import compute_classification_metrics
from ptbxl.modeling.predict import predict, predict_proba


@dataclass
class EvaluationResult:
    metrics: dict[str, float]
    y_pred: np.ndarray


def evaluate(
    model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series
) -> EvaluationResult:
    y_pred = predict(model, X_test)
    y_proba = predict_proba(model, X_test)
    metrics = compute_classification_metrics(y_test.to_numpy(), y_pred, y_proba)
    return EvaluationResult(metrics=metrics, y_pred=y_pred)
