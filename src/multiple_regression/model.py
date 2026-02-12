"""
Multiple Feature Regression Model — Level 3: Real-World Application

This module wraps scikit-learn's LinearRegression with a clean interface
for training, prediction, and evaluation on multi-feature datasets.

Your task: Implement the TODO sections to complete the model class.
Run tests with: pytest tests/test_multiple_regression.py
"""

import numpy as np
from sklearn.linear_model import LinearRegression


class MultipleRegressionModel:
    """A multiple feature regression model wrapper for real-world data.

    Attributes:
        model: The underlying scikit-learn LinearRegression instance.
        feature_names: Names of the features used for training.
        is_fitted: Whether the model has been trained.
    """

    def __init__(self):
        self.model = LinearRegression()
        self.feature_names = None
        self.is_fitted = False

    def fit(self, X_train, y_train, feature_names=None):
        """Train the multiple feature regression model.

        Args:
            X_train (np.ndarray): Training features, shape (m, n_features).
            y_train (np.ndarray): Training targets, shape (m,).
            feature_names (list[str], optional): Names for each feature.

        Returns:
            self
        """
        # TODO: Implement model fitting
        # Step 1: Fit the model on X_train and y_train
        # Step 2: Store feature_names and set is_fitted to True
        # Step 3: Return self

        raise NotImplementedError("Implement the fit method")

    def predict(self, X):
        """Make predictions on new data.

        Args:
            X (np.ndarray): Input features, shape (m, n_features).

        Returns:
            np.ndarray: Predicted values, shape (m,).

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        # TODO: Implement prediction
        # Step 1: Check that the model is fitted (raise RuntimeError if not)
        # Step 2: Use the model to predict and return results

        raise NotImplementedError("Implement the predict method")

    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data.

        Args:
            X_test (np.ndarray): Test features, shape (m, n_features).
            y_test (np.ndarray): True target values, shape (m,).

        Returns:
            dict: A dictionary with evaluation metrics:
                - "mse": Mean Squared Error
                - "rmse": Root Mean Squared Error
                - "r2": R-squared score
                - "mae": Mean Absolute Error
        """
        # TODO: Implement evaluation
        # Step 1: Get predictions using self.predict()
        # Step 2: Compute MSE (use sklearn.metrics.mean_squared_error)
        # Step 3: Compute RMSE (square root of MSE)
        # Step 4: Compute R² (use sklearn.metrics.r2_score)
        # Step 5: Compute MAE (use sklearn.metrics.mean_absolute_error)
        # Step 6: Return a dict with keys: "mse", "rmse", "r2", "mae"

        raise NotImplementedError("Implement the evaluate method")

    def get_coefficients(self):
        """Get the model coefficients (weights and bias).

        Returns:
            dict: A dictionary with:
                - "weights": np.ndarray of feature weights
                - "bias": float bias term
                - "feature_importance": list of (feature_name, weight) tuples
                  sorted by absolute weight (descending)

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        # TODO: Implement coefficient extraction
        # Step 1: Check that the model is fitted
        # Step 2: Get weights from model.coef_ and bias from model.intercept_
        # Step 3: If feature_names exist, create a sorted list of
        #         (feature_name, weight) tuples sorted by |weight| descending
        # Step 4: Return the dict

        raise NotImplementedError("Implement the get_coefficients method")
