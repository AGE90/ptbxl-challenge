from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureSelectionResult:
    """Outcome of one selector fit within a `FeatureSelectionPipeline` run."""

    method: str
    selected_features: list[str]
    cv_score: float


class FeatureSelectionPipeline:
    """
    Runs several sklearn-`SelectorMixin`-compatible feature selectors on one
    feature set and reports a consensus subset.

    Selector choice (which sklearn methods, whether to include a wrapper
    like `metaheuristics.feature_selection.MetaheuristicSelector`, and what
    feature budget to use) is left to the caller via the `selectors` dict
    passed to `run` -- this class only handles imputation, running each
    selector, scoring, and aggregating agreement/consensus across them.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        imputer: BaseEstimator | None = None,
        scaler: BaseEstimator | str | None = "auto",
        cv: int = 5,
        scoring: str = "roc_auc",
        consensus_threshold: float = 0.5,
    ) -> None:
        self.estimator = estimator
        self.imputer = (
            imputer if imputer is not None else SimpleImputer(strategy="median")
        )
        # Coefficient-/importance-based selectors (RFE, SelectFromModel) are
        # scale-sensitive: a feature with larger raw magnitude (e.g. wavelet
        # energy vs. std) gets an artificially larger coefficient, not a more
        # informative one. Default ("auto") standardizes after imputation;
        # pass `scaler=None` to disable, or a fitted-compatible transformer
        # of your own.
        self.scaler: BaseEstimator | None = (
            StandardScaler() if scaler == "auto" else scaler
        )
        self.cv = cv
        self.scoring = scoring
        self.consensus_threshold = consensus_threshold

    def impute(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fits the imputer on `X`; returns a same-shape DataFrame with no NaNs."""
        values = self.imputer.fit_transform(X)
        return pd.DataFrame(values, columns=X.columns, index=X.index)

    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        """Imputes, then scales (if a scaler is configured); same shape as `X`."""
        X_imp = self.impute(X)
        if self.scaler is None:
            return X_imp
        values = self.scaler.fit_transform(X_imp)
        return pd.DataFrame(values, columns=X.columns, index=X.index)

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        selectors: dict[str, BaseEstimator],
    ) -> tuple[list[FeatureSelectionResult], pd.Series, pd.DataFrame]:
        """
        Imputes and scales `X`, fits each selector in `selectors`, and
        cross-validates `self.estimator` on each selector's chosen subset.

        Returns
        -------
        results : list[FeatureSelectionResult]
            One entry per selector, in `selectors` iteration order.
        votes : pd.Series
            Per-feature count of how many selectors chose it, indexed by
            `X.columns`.
        jaccard : pd.DataFrame
            Pairwise Jaccard similarity of the selected-feature masks,
            indexed/columned by selector name.
        """
        X_proc = self.preprocess(X)

        results = []
        masks: dict[str, np.ndarray] = {}
        for name, selector in selectors.items():
            selector.fit(X_proc, y)
            mask = np.asarray(selector.get_support())
            masks[name] = mask

            features = X.columns[mask].tolist()
            score = cross_val_score(
                self.estimator,
                X_proc.loc[:, mask],
                y,
                cv=self.cv,
                scoring=self.scoring,
            ).mean()
            results.append(FeatureSelectionResult(name, features, float(score)))

        votes = pd.Series(
            np.sum(list(masks.values()), axis=0), index=X.columns, name="votes"
        )
        jaccard = _jaccard_matrix(masks)

        return results, votes, jaccard

    def consensus_features(self, votes: pd.Series, n_methods: int) -> list[str]:
        """Features voted for by >= `consensus_threshold` fraction of `n_methods`."""
        threshold = self.consensus_threshold * n_methods
        return [str(f) for f in votes[votes >= threshold].index]


def _jaccard_matrix(masks: dict[str, np.ndarray]) -> pd.DataFrame:
    names = list(masks)
    mat = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            union = np.logical_or(masks[a], masks[b]).sum()
            inter = np.logical_and(masks[a], masks[b]).sum()
            mat.loc[a, b] = inter / union if union else 1.0
    return mat
