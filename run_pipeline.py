from pipelines.training_pipeline import training_pipeline

if __name__ == "__main__":
    # Run the pipeline with the data path as an argument
    training_pipeline(data_path="data/logistics_dataset_with_maintenance_required.csv")