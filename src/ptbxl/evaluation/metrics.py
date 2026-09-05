"""Pure classification-metric computation, extracted from `modeling.tracking`."""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """precision/recall/f1 (weighted) + roc_auc (ovr if >2 classes, else binary).

    All values rounded to 2dp.
    """
    metrics = {
        "precision": float(
            np.round(precision_score(y_true, y_pred, average="weighted"), 2)
        ),
        "recall": float(np.round(recall_score(y_true, y_pred, average="weighted"), 2)),
        "f1": float(np.round(f1_score(y_true, y_pred, average="weighted"), 2)),
    }

    if y_proba is not None and len(np.unique(y_true)) > 2:
        metrics["roc_auc"] = float(
            np.round(roc_auc_score(y_true, y_proba, multi_class="ovr"), 2)
        )
    elif y_proba is not None:
        metrics["roc_auc"] = float(np.round(roc_auc_score(y_true, y_proba[:, 1]), 2))

    return metrics
