import io
import logging
import os
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
from mlflow.models.signature import infer_signature
from sklearn.base import BaseEstimator
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)

from ptbxl.utils.paths import reports_dir, tracking_dir

sns.set_theme()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@contextmanager
def change_working_dir(new_dir):
    """
    Temporarily changes the working directory.
    """
    original_dir = os.getcwd()
    try:
        os.chdir(new_dir)
        yield
    finally:
        os.chdir(original_dir)


def log_confusion_matrix(y_test, y_pred, labels):
    """
    Log a confusion matrix directly to MLflow without saving it locally.

    Parameters:
    -----------
    y_test : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    labels : list
        List of label names for the confusion matrix.

    Returns:
    --------
    None
    """
    # Create confusion matrix plot
    cm_display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=labels,
        cmap="Blues"
    )

    # Use an in-memory bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)  # Move to the beginning of the buffer

    # Log the confusion matrix as an artifact
    mlflow.log_artifact(buf, artifact_path="confusion_matrix.png")

    # Close the buffer and the plot
    buf.close()
    plt.close()


class ExperimentTracking:
    """
    This class provides methods for tracking and logging experiment details, including model 
    performance and artifacts.
    """

    def __init__(self):

        # Initialize tracking URI and artifact root
        self.backend = "sqlite"
        store_path = Path(tracking_dir("mlflow.db"))
        artifacts_dir = Path(tracking_dir("mlruns"))

        # Ensure that artifacts path exist
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Set the MLflow tracking URI and artifacts location
        self.store_uri = f"{self.backend}:///{store_path.as_posix()}"
        self.artifacts_location = artifacts_dir.as_uri()

        # Set the MLflow tracking URI
        mlflow.set_tracking_uri(self.store_uri)
        # mlflow.set_artifact_uri(self.artifacts_location) # No available

        # # Set the tracking URI and artifact location globally
        # os.environ["MLFLOW_TRACKING_URI"] = self.store_uri
        # os.environ["MLFLOW_ARTIFACT_URI"] = self.artifacts_location

        logger.info("MLflow tracking URI: %s", self.store_uri)
        logger.info("MLflow artifacts root: %s", self.artifacts_location)
        logger.info(
            "Open the MLflow UI by running the following command:\n"
            "mlflow ui --backend-store-uri %s",
            mlflow.get_tracking_uri(),
        )

    def track_experiment(
        self,
        experiment_name: str,
        run_name: str,
        model_name: str,
        developer: str,
        model: BaseEstimator,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        labels: list,
        xgb: bool = False
    ):
        """
        Tracks an experiment in MLflow by logging parameters, metrics, and artifacts.

        Parameters
        ----------
        experiment_name : str
            The name of the experiment.
        run_name : str
            The name of the run.
        model_name : str
            The name of the model being tracked.
        developer : str
            The developer's name.
        model : BaseEstimator
            The machine learning model to track.
        X_train : np.ndarray
            Training feature set.
        y_train : np.ndarray
            Training target set.
        X_test : np.ndarray
            Test feature set.
        y_test : np.ndarray
            Test target set.
        """

        # Predict and evaluate
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
        else:
            y_proba = None

        # Metrics
        precision = np.round(precision_score(
            y_test, y_pred, average="weighted"), 2)
        recall = np.round(recall_score(
            y_test, y_pred, average="weighted"), 2)
        f1 = np.round(f1_score(y_test, y_pred, average="weighted"), 2)
        if y_proba is not None and len(np.unique(y_test)) > 2:
            roc_auc = np.round(roc_auc_score(
                y_test, y_proba, multi_class="ovr"), 2)
        elif y_proba is not None:
            roc_auc = np.round(roc_auc_score(y_test, y_proba[:, 1]), 2)
        else:
            roc_auc = None

        logger.info("Precision: %s", precision)
        logger.info("Recall: %s", recall)
        logger.info("F1: %s", f1)
        if roc_auc is not None:
            logger.info("ROC AUC: %s", roc_auc)

        # Model signature
        signature = infer_signature(X_train, model.predict(X_train))

        try:
            # Temporarily change working directory
            with change_working_dir(str(Path(tracking_dir("mlruns")).parent)):

                # Ensure experiment exists
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    # Create experiment with explicit artifact location
                    mlflow.create_experiment(
                        name=experiment_name,
                        artifact_location=self.artifacts_location
                    )

                # Set experiment
                mlflow.set_experiment(experiment_name)

                # Log experiment details
                with mlflow.start_run(run_name=run_name):
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_param("developer", developer)

                    # Log hyperparameters if available
                    if hasattr(model, "get_params"):
                        mlflow.log_params(model.get_params())

                    # Log metrics
                    mlflow.log_metric("precision", precision)
                    mlflow.log_metric("recall", recall)
                    mlflow.log_metric("f1", f1)
                    if roc_auc is not None:
                        mlflow.log_metric("roc_auc", roc_auc)

                    # Log confusion matrix directly
                    log_confusion_matrix(y_test, y_pred, labels=labels)

                    # Log confusion matrix as artifact
                    # mlflow.log_artifact(str(reports_path), artifact_path="confusion_matrix")

                    # if xgb:
                    #     mlflow.xgboost.log_model(
                    #         model,
                    #         artifact_path=f"model_{model_name}",
                    #         signature=signature,
                    #         model_format="pkl"
                    #     )
                    # else:
                    #     # Log model with signature
                    #     mlflow.sklearn.log_model(
                    #         model,
                    #         artifact_path=f"model_{model_name}",
                    #         signature=signature,
                    #     )

        except Exception as e:
            logger.error(
                "An error occurred during experiment tracking %s", e, exc_info=True)
