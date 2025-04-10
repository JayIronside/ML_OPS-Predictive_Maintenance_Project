import logging
import pandas as pd
import os
import numpy as np
from xgboost import XGBClassifier
from src.model_dev import ModelDevelopment, Oversampler  
from src.data_cleaning_and_preparation import DataScalingAndEncodingStrategy
from sklearn.base import ClassifierMixin
from sklearn.decomposition import PCA  
from .config import ModelNameConfig
from zenml import step
import joblib
import mlflow
from zenml.client import Client




# Create the config object with the appropriate save directory
config = ModelNameConfig(model_name="xgboost", save_dir="../saved_model")


experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig
) -> ClassifierMixin:
    try:
        model_name = config.model_name.lower()  # Normalize model name to lowercase

        if model_name == "xgboost":
            mlflow.sklearn.autolog()  # Enable autologging for XGBoost
            # Ensure all data is numeric before training
            if not np.issubdtype(X_train.dtype, np.number):
                raise ValueError("X_train contains non-numeric values. Ensure proper preprocessing.")
            

            # Initialize the oversampler
            oversampler = Oversampler(random_state=42)
            X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

            # Initialize the model development class
            model_dev = ModelDevelopment(pca_model_path="pca.pkl", random_state=42)

            # Train the model
            trained_model = model_dev.train_xgboost(X_train_resampled, y_train_resampled)

            # Save the trained model
            model_path = os.path.join(config.save_dir, "xgb_classifier_pca.joblib")
            joblib.dump(trained_model, model_path)

            logging.info(f"XGBoost classifier trained and saved as {model_path}")
            return trained_model
        else:
            raise ValueError(f"Model '{config.model_name}' is not supported. Supported models: ['xgboost']")
    except Exception as e:
        logging.error(f"Error in train_model step: {e}")
        raise e
