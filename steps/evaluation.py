import logging
import pandas as pd
import numpy as np
import joblib
from zenml import step
from src.evaluation import ClassificationEvaluation
from sklearn.base import ClassifierMixin
from typing import Tuple

@step
def evaluate_model(model: ClassifierMixin, X_test: np.ndarray, y_test: pd.Series) -> Tuple[dict, np.ndarray]:
    try:
        # Ensure X_test is a NumPy array
        if not isinstance(X_test, np.ndarray):
            raise ValueError("X_test must be a NumPy array.")

        # Generate predictions on the test set
        y_test_pred = model.predict(X_test)

        # Initialize the evaluation class
        evaluator = ClassificationEvaluation()

        # Calculate metrics
        metrics = evaluator.calculate_metrics(y_test, y_test_pred)
        classification_report = metrics["classification_report"]
        confusion_matrix = metrics["confusion_matrix"]

        logging.info("Evaluation completed successfully.")
        return classification_report, confusion_matrix

    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        raise e