import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentTracking:
    
    def __init__(self, tracking_dir: Path):
        
        
        self.backend = "sqlite"
        self.store_uri = f"{self.backend}:////{str(tracking_dir('mlflow.db')).replace(os.sep, '/')}"
        mlflow.set_tracking_uri(self.store_uri)
        
        # Specify artifacts location
        self.artifacts_root = f"file:///{str(tracking_dir('artifacts')).replace(os.sep, '/')}"
        
        print("Open the MLflow UI by running the following command into your terminal, and navigate to the URL:")
        print(f"mlflow ui --backend-store-uri {mlflow.get_tracking_uri()}")
        
        
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
    ):
        
        
        
        y_pred = model.predict(X_test)
        
        # Evaluate
        precision = np.round(precision_score(y_test, y_pred), 2)
        recall = np.round(recall_score(y_test, y_pred), 2)
        f1 = np.round(f1_score(y_test, y_pred), 2)
        roc_auc = np.round(roc_auc_score(y_test, y_pred), 2)
        
        logger.info(f"Precision: {precision}")
        logger.info(f"Recall: {recall}")
        logger.info(f"F1: {f1}")
        logger.info(f"ROC AUC: {roc_auc}")
        
        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=self.artifacts_root,
        )
        
        # Create an experiment
        experiment = mlflow.set_experiment(
            experiment_name=experiment_name,   
        )
        
        
        # Start an MLflow run
        with mlflow.start_run(run_name=run_name):

            # Log the model
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("developer", developer)
            
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("roc_auc", roc_auc)
            # Log the confusion matrix
            mlflow.log_metric("confusion_matrix", confusion_matrix(y_test, y_pred))
            # Log the model
            mlflow.sklearn.log_model(
                model,
                f"model_{model_name}",
                
                
            )
            

# ####
# import numpy as np
# import logging
# import os
# import mlflow
# from pathlib import Path
# from sklearn.base import BaseEstimator
# from sklearn.metrics import (
#     ConfusionMatrixDisplay,
#     confusion_matrix,
#     f1_score,
#     precision_score,
#     recall_score,
#     roc_auc_score,
# )


# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


class ExperimentTracking2:
    def __init__(self, tracking_dir: Path):
        # Ensure directory exists
        tracking_dir.mkdir(parents=True, exist_ok=True)

        self.backend = "sqlite"
        self.store_uri = f"{self.backend}:////{str(tracking_dir('mlflow.db')).replace(os.sep, '/')}"
        mlflow.set_tracking_uri(self.store_uri)

        # Specify artifacts location
        self.artifacts_root = f"file:///{str(tracking_dir('artifacts')).replace(os.sep, '/')}"
        (tracking_dir / 'artifacts').mkdir(parents=True, exist_ok=True)

        logger.info(
            "Open the MLflow UI by running the following command into your terminal, and navigate to the URL:"
        )
        logger.info(f"mlflow ui --backend-store-uri {mlflow.get_tracking_uri()}")

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
        custom_metrics: dict = None,
    ):
        try:
            # Predict
            y_pred = model.predict(X_test)

            # Handle multi-class or binary classification for roc_auc
            if len(np.unique(y_test)) > 2:
                roc_auc = np.round(roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr"), 2)
            else:
                roc_auc = np.round(roc_auc_score(y_test, y_pred), 2)

            # Evaluate
            precision = np.round(precision_score(y_test, y_pred, average="weighted"), 2)
            recall = np.round(recall_score(y_test, y_pred, average="weighted"), 2)
            f1 = np.round(f1_score(y_test, y_pred, average="weighted"), 2)

            logger.info(f"Precision: {precision}")
            logger.info(f"Recall: {recall}")
            logger.info(f"F1: {f1}")
            logger.info(f"ROC AUC: {roc_auc}")

            # Log experiment
            mlflow.set_experiment(experiment_name=experiment_name)

            with mlflow.start_run(run_name=run_name):
                # Log parameters
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("developer", developer)
                if hasattr(model, "get_params"):
                    mlflow.log_params(model.get_params())

                # Log metrics
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1", f1)
                mlflow.log_metric("roc_auc", roc_auc)

                # Log confusion matrix as artifact
                cm = confusion_matrix(y_test, y_pred)
                cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
                fig, ax = plt.subplots(figsize=(8, 8))
                cm_display.plot(ax=ax)
                cm_artifact_path = Path(self.artifacts_root) / f"{run_name}_confusion_matrix.png"
                plt.savefig(cm_artifact_path)
                plt.close(fig)
                mlflow.log_artifact(str(cm_artifact_path), artifact_path="confusion_matrix")

                # Log custom metrics if provided
                if custom_metrics:
                    for metric_name, metric_value in custom_metrics.items():
                        mlflow.log_metric(metric_name, metric_value)

                # Log the model
                mlflow.sklearn.log_model(model, artifact_path=f"model_{model_name}")

        except Exception as e:
            logger.error(f"An error occurred during experiment tracking: {e}", exc_info=True)
