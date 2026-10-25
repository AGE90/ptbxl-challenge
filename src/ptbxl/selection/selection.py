"""Config-driven feature selection (see notebook 04's `select_domain_features`).

`config.methods` picks selectors by name from `selectors.SELECTORS`. "pso" needs the
optional `metaheuristics` extra installed; it is imported only when requested.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from ptbxl.config import SelectionConfig
from ptbxl.selection.selectors import FeatureSelectionPipeline, build_selectors

_RANDOM_STATE = 42


def select_features(
    X: pd.DataFrame, y: pd.Series, config: SelectionConfig
) -> list[str]:
    """Consensus of the `config.methods` selectors, budgeted to `max_features`.

    Subsamples to `config.sample_size` (stratified) first if set, matching
    notebook 04's `SELECTION_SAMPLE_SIZE` behavior.
    """
    if config.method != "consensus":
        raise NotImplementedError(f"unsupported selection method: {config.method}")

    if config.sample_size is not None and config.sample_size < len(X):
        sample_idx, _ = train_test_split(
            X.index,
            train_size=config.sample_size,
            stratify=y,
            random_state=_RANDOM_STATE,
        )
        X, y = X.loc[sample_idx], y.loc[sample_idx]

    pipeline = FeatureSelectionPipeline(
        estimator=LogisticRegression(max_iter=1000, random_state=_RANDOM_STATE),
        cv=3,
        scoring="roc_auc",
    )
    selectors = build_selectors(config.max_features, _RANDOM_STATE, config.methods)

    _, votes, _ = pipeline.run(X, y, selectors)
    return pipeline.consensus_features(votes, len(selectors))
