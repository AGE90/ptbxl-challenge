"""Training / hyperparameter-optimization stage (see notebook 05).

ponytail: `optimize_hyperparameters` only knows a param space for `XGBClassifier`
-- that's the only hyperparameter-search pattern that exists today (notebook 04's
PSO is for *feature selection*, not model hyperparameters). Add a branch when a
second model actually needs tuning.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd
from scipy.stats import randint, uniform
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from ptbxl.config import OptimizationConfig, TrainingConfig
from ptbxl.modeling.models import get_model


@dataclass
class TrainResult:
    model: BaseEstimator
    best_params: dict[str, Any]
    cv_score: float | None


def prepare_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame | None = None,
    X_test: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, ...]:
    """Median-imputes and standard-scales, fit on `X_train` only.

    Returns a tuple the same length as the non-`None` inputs given, in
    `(train, val, test)` order.
    """
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    def _fit_transform(X: pd.DataFrame) -> pd.DataFrame:
        values = scaler.fit_transform(imputer.fit_transform(X))
        return pd.DataFrame(values, columns=X.columns, index=X.index)

    def _transform(X: pd.DataFrame) -> pd.DataFrame:
        values = scaler.transform(imputer.transform(X))
        return pd.DataFrame(values, columns=X.columns, index=X.index)

    outputs = [_fit_transform(X_train)]
    for X in (X_val, X_test):
        if X is not None:
            outputs.append(_transform(X))

    return tuple(outputs)


def optimize_hyperparameters(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    config: OptimizationConfig,
    cv: Any,
) -> TrainResult:
    """Randomized hyperparameter search; only `XGBClassifier` has a param space."""
    if type(estimator).__name__ != "XGBClassifier":
        raise NotImplementedError(
            f"no hyperparameter search space defined for {type(estimator).__name__}"
        )

    neg, pos = (y == 0).sum(), (y == 1).sum()
    scale_pos_weight = neg / pos
    param_distributions = {
        "n_estimators": randint(100, 400),
        "max_depth": randint(3, 8),
        "learning_rate": uniform(0.01, 0.29),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "scale_pos_weight": uniform(scale_pos_weight * 0.7, scale_pos_weight * 0.6),
    }

    search = RandomizedSearchCV(
        estimator,
        param_distributions=param_distributions,
        n_iter=config.n_trials,
        cv=cv,
        scoring=config.metric,
        random_state=getattr(estimator, "random_state", None),
        n_jobs=-1,
    )
    search.fit(X, y)

    return TrainResult(
        model=search.best_estimator_,
        best_params=dict(search.best_params_),
        cv_score=float(search.best_score_),
    )


def optimize_and_train(
    X: pd.DataFrame,
    y: pd.Series,
    training_config: TrainingConfig,
    optimization_config: OptimizationConfig | None,
    cv: Any = None,
) -> TrainResult:
    """`logistic_regression` -> plain `.fit()`; `xgboost` -> tuned via search."""
    if training_config.model == "logistic_regression":
        model = get_model(training_config, y)
        model.fit(X, y)
        return TrainResult(
            model=model, best_params=dict(model.get_params()), cv_score=None
        )

    if training_config.model == "xgboost":
        if optimization_config is None or cv is None:
            raise ValueError("xgboost training requires optimization_config and cv")
        estimator = get_model(
            TrainingConfig(model="xgboost", random_state=training_config.random_state)
        )
        return optimize_hyperparameters(estimator, X, y, optimization_config, cv)

    raise NotImplementedError(f"unsupported model: {training_config.model}")
