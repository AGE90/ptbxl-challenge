"""Pipeline B: Pipeline A + split -> optimize/train -> evaluate.

Splits happen *after* preprocessing/feature engineering, not before: those
steps are stateless per-record transforms with no leakage risk, so splitting
first would only mean running expensive neurokit2 delineation three times for
no benefit. The only fit-sensitive steps (imputer/scaler, RandomizedSearchCV's
own cross-validation) only ever see `splits.X_train`.
"""

from ptbxl.config import Config
from ptbxl.data.splitting import split_dataset
from ptbxl.evaluation.evaluation import EvaluationResult, evaluate
from ptbxl.modeling.optimize import TrainResult, optimize_and_train, prepare_features
from ptbxl.pipelines.feature_engineering import feature_engineering_pipeline


def training_pipeline(config: Config) -> tuple[TrainResult, EvaluationResult]:
    X, y, metadata = feature_engineering_pipeline(config)
    splits = split_dataset(X, y, metadata)

    X_train, X_test = prepare_features(splits.X_train, X_test=splits.X_test)

    result = optimize_and_train(
        X_train, splits.y_train, config.training, config.optimization, cv=splits.cv
    )
    evaluation = evaluate(result.model, X_test, splits.y_test)

    return result, evaluation
