from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
from skrebate import MultiSURF, ReliefF
from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    f_classif,
    mutual_info_classif,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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

    Selector choice (which sklearn methods and what feature budget to use) is
    left to the caller via the `selectors` dict passed to `run` -- this class
    only handles imputation, running each selector, scoring, and aggregating
    agreement/consensus across them.
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
            imputer
            if imputer is not None
            # keep_empty_features: an all-NaN column (e.g. a wave-spectral
            # feature with 0 detected beats) would otherwise be silently
            # dropped by sklearn, breaking impute()'s same-shape guarantee.
            else SimpleImputer(strategy="median", keep_empty_features=True)
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


def _relief_scores(X, y, estimator_cls, **kwargs) -> np.ndarray:
    """`SelectKBest` score_func adapter for skrebate's Relief-family estimators."""
    fitted = estimator_cls(**kwargs).fit(np.asarray(X), np.asarray(y))
    scores: np.ndarray = np.asarray(fitted.feature_importances_)
    return scores


def _logreg(random_state: int) -> LogisticRegression:
    return LogisticRegression(max_iter=1000, random_state=random_state)


def _pso(k: int, random_state: int) -> BaseEstimator:
    # Imported lazily: `metaheuristics` is an optional extra, only needed if
    # "pso" is requested. Ignores `k` -- the swarm picks its own subset size.
    from metaheuristics.algorithms.particle_swarm import ParticleSwarmOptimization
    from metaheuristics.feature_selection import MetaheuristicSelector

    return MetaheuristicSelector(
        estimator=_logreg(random_state),
        optimizer=ParticleSwarmOptimization(num_particles=10, max_iterations=8),
        alpha=0.1,
        cv=3,
        random_state=random_state,
    )


# name -> factory(k, random_state). Relief-family is O(n_samples^2 * n_features):
# fine on a few thousand rows, slow on the full dataset.
SELECTORS: dict[str, Callable[[int, int], BaseEstimator]] = {
    "anova_f": lambda k, rs: SelectKBest(f_classif, k=k),
    "mutual_info": lambda k, rs: SelectKBest(mutual_info_classif, k=k),
    "select_from_model": lambda k, rs: SelectFromModel(
        _logreg(rs), max_features=k, threshold=-np.inf
    ),
    "rfe": lambda k, rs: RFE(_logreg(rs), n_features_to_select=k, step=0.2),
    # Classic Relief: 1 nearest hit/miss (target is binary).
    "relief": lambda k, rs: SelectKBest(
        partial(_relief_scores, estimator_cls=ReliefF, n_neighbors=1), k=k
    ),
    "relieff": lambda k, rs: SelectKBest(
        partial(_relief_scores, estimator_cls=ReliefF, n_neighbors=10), k=k
    ),
    # ReliefF variant with an adaptive neighbour radius -- nothing to tune.
    "multisurf": lambda k, rs: SelectKBest(
        partial(_relief_scores, estimator_cls=MultiSURF), k=k
    ),
    # Non-linear embedded selector (select_from_model/rfe are linear).
    "extra_trees": lambda k, rs: SelectFromModel(
        ExtraTreesClassifier(n_estimators=200, random_state=rs, n_jobs=-1),
        max_features=k,
        threshold=-np.inf,
    ),
    "pso": _pso,
}

DEFAULT_METHODS = ["anova_f", "mutual_info", "select_from_model", "rfe"]


def build_selectors(
    k: int, random_state: int, methods: Iterable[str] | None = None
) -> dict[str, BaseEstimator]:
    """Selectors named in `methods` (default: `DEFAULT_METHODS`), budgeted to `k`."""
    methods = list(DEFAULT_METHODS if methods is None else methods)
    unknown = set(methods) - set(SELECTORS)
    if unknown:
        raise ValueError(
            f"unknown selection methods {sorted(unknown)}; "
            f"choose from {list(SELECTORS)}"
        )
    return {name: SELECTORS[name](k, random_state) for name in methods}


def _jaccard_matrix(masks: dict[str, np.ndarray]) -> pd.DataFrame:
    names = list(masks)
    mat = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            union = np.logical_or(masks[a], masks[b]).sum()
            inter = np.logical_and(masks[a], masks[b]).sum()
            mat.loc[a, b] = inter / union if union else 1.0
    return mat
