import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


def predict(model: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(X))


def predict_proba(model: BaseEstimator, X: pd.DataFrame) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None
    return np.asarray(model.predict_proba(X))
