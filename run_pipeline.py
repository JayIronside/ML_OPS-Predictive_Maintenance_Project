from pipelines.training_pipeline import training_pipeline

from zenml.client import Client

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    # Run the pipeline with the data path as an argument
    training_pipeline(data_path="data/logistics_dataset_with_maintenance_required.csv")