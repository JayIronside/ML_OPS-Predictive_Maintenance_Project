import logging
import pandas as pd
from zenml import step

class IngestData:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def get_data(self):
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)
    
@step
def ingest_df(data_path: str) -> pd.DataFrame:
    try:
        logging.info(f"Ingesting data from {data_path}")
        df = pd.read_csv(data_path)
        if df is None or df.empty:
            raise ValueError("The ingested DataFrame is empty or None.")
        logging.info(f"Data ingested successfully with shape: {df.shape}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {data_path}")
        raise
    except Exception as e:
        logging.error(f"Error ingesting data: {e}")
        raise

