import click
import logging
from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import mlflow_deployment, MLFlowDeploymentService
from pipelines.deployment_pipeline import continuous_deployment_pipeline, inference_pipeline
from typing import cast


DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"

@click.command()
@click.option("--config", "-c",
                type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]), default=DEPLOY_AND_PREDICT,
                help="Choose the deployment option.")

@click.option("--min_accuracy",
              default=0.85,
              help="Minimum accuracy required to deploy the model. Default is 0.85.")

def run_deployment(config: str, min_accuracy: float):
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting deployment process...")

    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT

    if deploy:
        logging.info("Running deployment pipeline...")
        continuous_deployment_pipeline(
            data_path="data/logistics_dataset_with_maintenance_required.csv",
            min_accuracy=min_accuracy,
            workers=3,
            timeout=60
        )
        logging.info("Deployment pipeline completed.")

    if predict:
        logging.info("Running prediction pipeline...")
        inference_pipeline(
            data="data/logistics_dataset_with_maintenance_required.csv"
        )
        logging.info("Prediction pipeline completed.")
        
        print(
            "You can run:\n" \
            f"[italic green] mlflow ui --backend-store-uri {get_tracking_uri()}"
            "[/italic green]\n ...to inspect your experiment runs within the MLflow"
             "UI. \n You can find your runs tracked within the"
              "`mlflow_example_pipeline` experiment. There you'll also be able to"
               "compare two or more runs. \n\n"
        )
        #fetch existing services with some pipeline name, step name and model name

        existing_services = mlflow_model_deployer_component.find_model_server(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
            model_name="model",
        )

        if existing_services:
            service = cast(MLFlowDeploymentService, existing_services[0])
            if service.is_running():
                print(f" The MLflow prediction server is running locally as a daemon"
                      f" process service and accepts inference requests at :\n"
                      f" {service.prediction_url} \n"
                      f" To stop the service, run"
                      f" [italic green] `zenml model_deployer models delete"
                      f" {str(service.wid)} `[/italic green]")
            elif service.is_failed:
                print(" The MLflow prediction server is in a failed state: \n "
                      f" Last state: `{service.status.state.value}` \n"
                      f" Last error: `{service.status.last_error}`")
        else:
            print(" The MLflow prediction server is currently running. The deployment\n"
                  f" pipeline must run first to train a model and deploy it. Execute\n"
                  f" the same command with the `--deploy` argument to deploy a model.")

if __name__ == "__main__":
    run_deployment()