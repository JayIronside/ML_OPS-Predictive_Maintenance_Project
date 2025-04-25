from zenml import step
import json
import numpy as np
import pandas as pd
from zenml.integrations.mlflow.services import MLFlowDeploymentService
import logging

@step
def predictor(
    service: MLFlowDeploymentService,
    data: str
) -> np.ndarray:
    try:
        service.start(timeout=10)

        # Parse the input JSON
        data = json.loads(data)
        if not all(key in data["data"] for key in ["columns", "index", "data"]):
            raise ValueError("Input JSON is missing required keys: 'columns', 'index', or 'data'.")

        # Convert the data to a DataFrame
        columns_for_df = data["data"]["columns"]
        index_for_df = data["data"]["index"]
        data_for_df = data["data"]["data"]
        df = pd.DataFrame(data_for_df, columns=columns_for_df, index=index_for_df)

        # Convert DataFrame to NumPy array
        json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
        data = np.array(json_list)

        # Make predictions using the service
        prediction = service.predict(data)
        return prediction

    except Exception as e:
        logging.error(f"Error in predictor step: {e}")
        raise e

    finally:
        service.stop()