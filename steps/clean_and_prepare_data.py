import logging
import pandas as pd
from zenml import step
from src.data_cleaning_and_preparation import (
    DataPreProcessStrategy,
    DataDivideStrategy,
    DataScalingAndEncodingStrategy
)
from typing import Tuple
import numpy as np

@step
def clean_df(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    try:
        if df is None or df.empty:
            raise ValueError("The input DataFrame is empty or None.")

        logging.info(f"DataFrame shape before preprocessing: {df.shape}")

        # Step 1: Preprocess the data (drop columns etc.)
        preprocess_strategy = DataPreProcessStrategy()
        df = preprocess_strategy.handle_data(df)

        logging.info(f"DataFrame shape after preprocessing: {df.shape}")

        # Step 2: Split into train/test sets
        divide_strategy = DataDivideStrategy()
        X_train, X_test, y_train, y_test = divide_strategy.handle_data(df)

        logging.info(f"Train/Test split completed. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

        # Step 3: Scale and encode the features
        scale_encode_strategy = DataScalingAndEncodingStrategy()
        X_train_processed, X_test_processed = scale_encode_strategy.handle_data(X_train, X_test)

        logging.info(f"Data processed. X_train_processed shape: {X_train_processed.shape}, X_test_processed shape: {X_test_processed.shape}")

        return X_train_processed, X_test_processed, y_train, y_test

    except Exception as e:
        logging.error(f"Error in clean_df step: {e}")
        raise e
