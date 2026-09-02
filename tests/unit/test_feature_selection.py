import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression

from ptbxl.features.feature_selection import FeatureSelectionPipeline

pipeline = FeatureSelectionPipeline(
    estimator=LogisticRegression(), cv=3, scoring="accuracy"
)


def _sample_data():
    rng = np.random.default_rng(0)
    n_samples, n_features = 60, 5
    X = pd.DataFrame(
        rng.standard_normal((n_samples, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    # f0 is the only feature correlated with the target; the rest is noise.
    y = pd.Series((X["f0"] > 0).astype(int), name="target")
    # Inject missing values so imputation has something real to do.
    X.loc[X.index[:5], "f1"] = np.nan
    return X, y


def test_impute_removes_missing_values_and_preserves_shape():
    X, _ = _sample_data()
    assert X.isna().any().any()

    X_imp = pipeline.impute(X)

    assert X_imp.shape == X.shape
    assert list(X_imp.columns) == list(X.columns)
    assert not X_imp.isna().any().any()


def test_preprocess_scales_by_default_and_can_be_disabled():
    X, _ = _sample_data()

    X_scaled = pipeline.preprocess(X)
    assert np.allclose(X_scaled.mean(), 0, atol=1e-8)
    assert np.allclose(X_scaled.std(ddof=0), 1, atol=1e-8)

    unscaled_pipeline = FeatureSelectionPipeline(
        estimator=LogisticRegression(), scaler=None
    )
    X_unscaled = unscaled_pipeline.preprocess(X)
    assert not np.allclose(X_unscaled.std(ddof=0), 1, atol=1e-8)


def test_run_returns_one_result_per_selector_and_consistent_votes():
    X, y = _sample_data()
    selectors = {
        "variance": VarianceThreshold(),
        "kbest": SelectKBest(f_classif, k=2),
    }

    results, votes, jaccard = pipeline.run(X, y, selectors)

    assert {r.method for r in results} == set(selectors)
    for result in results:
        assert set(result.selected_features).issubset(set(X.columns))
        assert np.isfinite(result.cv_score)

    assert list(votes.index) == list(X.columns)
    assert votes.sum() == sum(len(r.selected_features) for r in results)

    assert list(jaccard.index) == list(selectors)
    assert list(jaccard.columns) == list(selectors)
    assert np.allclose(np.diag(jaccard.values), 1.0)


def test_consensus_features_applies_threshold():
    # threshold = consensus_threshold(0.5) * n_methods(4) = 2 votes required.
    votes = pd.Series({"a": 2, "b": 1, "c": 0}, name="votes")

    consensus = pipeline.consensus_features(votes, n_methods=4)

    assert consensus == ["a"]
