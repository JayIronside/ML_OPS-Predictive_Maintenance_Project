import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

class Evaluation(ABC):
    @abstractmethod
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass

class ClassificationEvaluation(Evaluation):
    """
    Class for evaluating classification models.
    """
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            # Generate classification report
            report = classification_report(y_true, y_pred, output_dict=True)
            logging.info("Classification Report:")
            logging.info(report)

            # Generate confusion matrix
            conf_matrix = confusion_matrix(y_true, y_pred)
            logging.info("Confusion Matrix:")
            logging.info(conf_matrix)

            return {
                "classification_report": report,
                "confusion_matrix": conf_matrix
            }
        except Exception as e:
            logging.error(f"Error calculating metrics: {e}")
            raise e

