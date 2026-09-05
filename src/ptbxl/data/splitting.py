"""Train/val/test split on PTB-XL's own patient-disjoint `strat_fold` scheme.

ponytail: no `config` param -- the 1-8/9/10 fold convention is fixed to
PTB-XL's 10-fold scheme, nothing about it is configurable today.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import PredefinedSplit


@dataclass
class Splits:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    cv: PredefinedSplit  # folds 1-8, for hyperparameter search on X_train/y_train


def split_dataset(X: pd.DataFrame, y: pd.Series, metadata: pd.DataFrame) -> Splits:
    """Splits `X`/`y` (indexed like `metadata`) using `metadata['strat_fold']`.

    Folds 1-8 -> train (also the `cv` groups for hyperparameter search),
    fold 9 -> validation, fold 10 -> held-out test.
    """
    strat_fold = metadata["strat_fold"]
    train_mask = strat_fold.between(1, 8)
    val_mask = strat_fold == 9
    test_mask = strat_fold == 10

    cv = PredefinedSplit(test_fold=strat_fold[train_mask].to_numpy() - 1)

    return Splits(
        X_train=X[train_mask],
        y_train=y[train_mask],
        X_val=X[val_mask],
        y_val=y[val_mask],
        X_test=X[test_mask],
        y_test=y[test_mask],
        cv=cv,
    )
