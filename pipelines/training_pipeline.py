from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_and_prepare_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model
from steps.config import ModelNameConfig

@pipeline(enable_cache=False)
def training_pipeline(data_path: str):
    """Pipeline to train the model."""
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)

    # Instantiate the ModelNameConfig
    config = ModelNameConfig(model_name="xgboost")

    model = train_model(X_train, X_test, y_train, y_test, config=config)
    classification_report, confusion_matrix, accuracy = evaluate_model(model, X_test, y_test)
    return classification_report, confusion_matrix, accuracy