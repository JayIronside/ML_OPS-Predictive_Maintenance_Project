import logging
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib
import numpy as np


class Oversampler:
    """
    Class for handling class imbalance using oversampling
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.ros = RandomOverSampler(random_state=self.random_state)

    def fit_resample(self, X, y):
        return self.ros.fit_resample(X, y)

class ModelDevelopment:
    """
    Class for training and saving the XGBoost classifier.
    PCA and oversampling are handled externally.
    """
    def __init__(self, pca_model_path: str = None, random_state: int = 42):
        self.pca_model_path = pca_model_path
        self.random_state = random_state
        self.xgb_clf = None
        self.pca = None

    def transform_with_pca(self, X_train_processed, X_test_processed):
        """Transform the data using the loaded PCA model."""
        if self.pca is None:
            raise ValueError("PCA model is not loaded. Call load_pca() first.")

        X_train_pca = self.pca.transform(X_train_processed)
        X_test_pca = self.pca.transform(X_test_processed)

        # Ensure PCA transformation is applied consistently
        if X_train_pca.shape[1] < 16 or X_test_pca.shape[1] < 16:
            raise ValueError("PCA transformation resulted in fewer than 16 components. Check PCA configuration.")

        # Select the first 16 principal components
        X_train_pca_16 = X_train_pca[:, :16]
        X_test_pca_16 = X_test_pca[:, :16]

        return X_train_pca_16, X_test_pca_16

    def oversample(self, X_train_pca_16, y_train):
        """Perform oversampling to balance the classes in the training set."""
        ros = RandomOverSampler(random_state=self.random_state)
        X_train_bal, y_train_bal = ros.fit_resample(X_train_pca_16, y_train)
        print("Class distribution after oversampling:", np.bincount(y_train_bal))
        return X_train_bal, y_train_bal

    def train_xgboost(self, X_train_bal, y_train_bal):
        """
        Train the XGBoost classifier on the balanced training set.
        """
        try:
            self.xgb_clf = XGBClassifier(
                random_state=self.random_state,
                use_label_encoder=False,
                enable_categorical=False
            )
            self.xgb_clf.fit(X_train_bal, y_train_bal)
            return self.xgb_clf
        except Exception as e:
            logging.error(f"Error training XGBoost model: {e}")
            raise e

    def save_model(self, model_path: str):
        """
        Save the trained XGBoost model to the specified path.
        """
        if self.xgb_clf is None:
            raise ValueError("XGBoost model is not trained yet. Call train_xgboost() first.")
        joblib.dump(self.xgb_clf, model_path)
        logging.info(f"XGBoost model saved to {model_path}")
