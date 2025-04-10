"""
from pipelines.training_pipeline import training_pipeline

from zenml.client import Client

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    # Run the pipeline with the data path as an argument
    training_pipeline(data_path="data/logistics_dataset_with_maintenance_required.csv")
"""
from zenml.integrations.mlflow import MLFLOW_MODEL_EXPERIMENT_TRACKER_FLAVOR
from zenml.client import Client

if __name__ == "__main__":
    # Set the active ZenML project
    Client().set_active_project("default")

    # Retrieve the active experiment tracker from the stack
    experiment_tracker = Client().active_stack.experiment_tracker

    # Get the tracking URI from the registered tracker
    tracking_uri = experiment_tracker.get_tracking_uri()
    print(f"Tracking URI: {tracking_uri}")

    # Run the pipeline with the data path as an argument
    from pipelines.training_pipeline import training_pipeline
    training_pipeline(data_path="data/logistics_dataset_with_maintenance_required.csv")
    print("Pipeline executed successfully.")

