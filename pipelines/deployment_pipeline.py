import logging
import numpy as np
import pandas as pd
from zenml import step, pipeline
from zenml.config import DockerSettings
# from materializer.custom_materializer import cs_materializer
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import mlflow_deployment
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseStep
# from zenml.steps import BaseStepConfig, Output
from pydantic import BaseModel
from steps.clean_and_prepare_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model
from steps.config import ModelNameConfig
from pipelines.utils import get_data_for_test
import json
from zenml.integrations.mlflow.services import MLFlowDeploymentService

docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseModel):
    min_accuracy: float = 0.85


@step(enable_cache=False)
def dynamic_importer() -> str:
    data = get_data_for_test()
    return data

@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
):
    """Implements a model deploy trigger based on criteria"""
    return accuracy >= config.min_accuracy

class MLFlowDeploymentLoaderStepParameters(BaseModel):
    """Parameters for the MLFlow deployment loader step."""
    pipeline_name: str
    step_name: str
    running: bool = True

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowModelDeployer:
    """get the MLflow deployer stack component"""
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )
    if not existing_services:
        raise RuntimeError(
            f" No MLFlow deployment service found for pipeline '{pipeline_name}',"
            f" step '{pipeline_step_name}', and model '{model_name}' "
            f" pipeline for the'{model_name}' is currently "
            f" running."
        )
    return existing_services[0]

"""@step
def predictor(
    service: MLFlowModelDeployer,
    data: str
) -> np.ndarray:
    service.start(timeout = 10)
    data = json.loads(data)
    # Convert the data to a DataFrame
    columns_for_df = data["data"].pop("columns")
    index_for_df = data["data"].pop("index")
    data_for_df = data["data"].pop("data")
    df = pd.DataFrame(data_for_df, columns=columns_for_df, index=index_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    # Make predictions using the service
    prediction = service.predict(data)
    return prediction
"""

@step
def predictor(
    service: MLFlowModelDeployer,
    data: str
) -> np.ndarray:
    try:
        # Start the service
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


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(data_path: str, min_accuracy: float = 0.85, workers: int = 1,
                                   timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT):
    df = ingest_df(data_path=data_path)
    X_train, X_test, y_train, y_test = clean_df(df)

    # Instantiate the ModelNameConfig
    config = ModelNameConfig(model_name="xgboost")

    model = train_model(X_train, X_test, y_train, y_test, config=config)
    classification_report, confusion_matrix, accuracy = evaluate_model(model, X_test, y_test)

    # Instantiate the DeploymentTriggerConfig
    trigger_config = DeploymentTriggerConfig(min_accuracy=min_accuracy)

    deployment_decision = deployment_trigger(accuracy, config=trigger_config)

    # Add detailed logging to the deployment step
    logging.info("Starting MLFlow model deployment step...")
    mlflow_model_deployer_step(
        model=model,
        model_name="model",  # Ensure the model name is set
        workers=workers,
        timeout=timeout,
    )
    logging.info("MLFlow model deployment step completed.")

@step
def service_loader() -> MLFlowDeploymentService:
    """Load the MLFlowDeploymentService."""
    mlflow_model_deployer = MLFlowModelDeployer.get_active_model_deployer()
    existing_services = mlflow_model_deployer.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name="model",
        running=True
    )
    if not existing_services:
        # Provide detailed error message
        raise RuntimeError(
            "No running MLFlow deployment service found. Ensure that the deployment pipeline has been executed "
            "successfully and that a model is deployed. You can run the deployment pipeline using the following command: \n"
            "python run_deployment.py --config deploy"
        )
    return existing_services[0]

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(data: str):
    service = service_loader()
    predictor(
        service=service,
        data=data
    )
