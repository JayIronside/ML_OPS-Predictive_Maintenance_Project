import logging
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from typing import Union
import joblib


class DataStrategy (ABC):
    """
    Abstract class defining for handling data
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreProcessStrategy (DataStrategy):
    """
    Strategy for preprocessing data
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data
        """
        try:
            # Drop the 'Vehicle_ID' column
            data = data.drop(columns=['Vehicle_ID'])
            return data
        except Exception as e:
            logging.error(f"Error preprocessing data: {e}")
            raise e
        
class DataDivideStrategy (DataStrategy):
    """
    Strategy for dividing data into train and test sets
    """
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            if data is None or data.empty:
                raise ValueError("The input DataFrame is empty or None.")

            logging.info(f"DataFrame shape before splitting: {data.shape}")

            target = 'Maintenance_Required'
            if target not in data.columns:
                raise ValueError(f"Target column '{target}' not found in the DataFrame.")

            X = data.drop(columns=[target])
            y = data[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            logging.info(f"Train/Test split shapes: X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error dividing data: {e}")
            raise e
        
class DataScalingAndEncodingStrategy (DataStrategy):
    """
    Strategy for scaling, encoding, and applying PCA to data
    """
    def handle_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        try:
            # Identify categorical and numerical columns
            categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
            numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

            # Log identified columns
            logging.info(f"Categorical columns: {categorical_cols}")
            logging.info(f"Numerical columns: {numerical_cols}")

            # Create a ColumnTransformer for scaling numerical features and one hot encoding categorical features
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_cols),
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
                ]
            )

            # Fit the preprocessor on the training set and transform both train and test sets
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            # Validate that all data is numeric
            if not np.issubdtype(X_train_processed.dtype, np.number):
                raise ValueError("X_train contains non-numeric values after preprocessing.")
            if not np.issubdtype(X_test_processed.dtype, np.number):
                raise ValueError("X_test contains non-numeric values after preprocessing.")

            logging.info(f"Processed training data shape: {X_train_processed.shape}")
            logging.info(f"Processed test data shape: {X_test_processed.shape}")

            # Dump the fitted preprocessor for later deployment
            joblib.dump(preprocessor, 'preprocessor.pkl')
            logging.info("Preprocessor dumped as preprocessor.pkl")

            return X_train_processed, X_test_processed

        except Exception as e:
            logging.error(f"Error scaling, encoding, or applying PCA: {e}")
            raise e
        
class DataCleaningAndPreparation:
    """
    Class for cleaning and preparing data using the strategy pattern
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle data using the specified strategy
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error handling data: {e}")
            raise e