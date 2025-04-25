import logging
import pandas as pd
from src.data_cleaning_and_preparation import DataPreProcessStrategy, DataScalingAndEncodingStrategy

def get_data_for_test():
    try:
        # Load the dataset from a CSV file
        df = pd.read_csv("data/logistics_dataset_with_maintenance_required.csv")
        logging.info(f"DataFrame shape before preprocessing: {df.shape}")

        # Apply preprocessing strategy
        preprocess_strategy = DataPreProcessStrategy()
        df = preprocess_strategy.handle_data(df)
        logging.info(f"DataFrame shape after preprocessing: {df.shape}")

        # Apply scaling and encoding strategy
        scale_encode_strategy = DataScalingAndEncodingStrategy()
        X_train_processed, _ = scale_encode_strategy.handle_data(df, df)  # Use the same data for both train and test
        logging.info(f"Data shape after scaling and encoding: {X_train_processed.shape}")

        # Convert the processed data to JSON format
        result = pd.DataFrame(X_train_processed).to_json(orient="split")
        return result
    except Exception as e:
        logging.error(f"Error in get_data_for_test: {e}")
        raise e

