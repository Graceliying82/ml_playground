"""
Tests for Linear Regression — Level 3

These tests validate your implementation in src/linear_regression/.
Run with: pytest tests/test_linear_regression.py -v
"""

import numpy as np
import pytest

from src.linear_regression.model import LinearRegressionModel
from src.linear_regression.pipeline import load_data, preprocess_data, split_data


# ---------------------------------------------------------------------------
# Model Tests
# ---------------------------------------------------------------------------

class TestLinearRegressionModel:
    """Tests for the LinearRegressionModel class."""

    def setup_method(self):
        """Create simple test data: y = 2*x1 + 3*x2 + 5."""
        np.random.seed(42)
        m = 200
        self.X = np.random.rand(m, 2)
        self.y = 2 * self.X[:, 0] + 3 * self.X[:, 1] + 5 + np.random.randn(m) * 0.1
        self.feature_names = ["feature_1", "feature_2"]

    def test_fit_returns_self(self):
        model = LinearRegressionModel()
        result = model.fit(self.X, self.y)
        assert result is model, "fit() should return self"

    def test_is_fitted_after_training(self):
        model = LinearRegressionModel()
        assert not model.is_fitted
        model.fit(self.X, self.y)
        assert model.is_fitted

    def test_predict_shape(self):
        model = LinearRegressionModel()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        assert predictions.shape == self.y.shape, \
            f"Expected shape {self.y.shape}, got {predictions.shape}"

    def test_predict_before_fit_raises(self):
        model = LinearRegressionModel()
        with pytest.raises(RuntimeError):
            model.predict(self.X)

    def test_predictions_are_reasonable(self):
        model = LinearRegressionModel()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        # Predictions should be close to actual values (R² > 0.95)
        ss_res = np.sum((self.y - predictions) ** 2)
        ss_tot = np.sum((self.y - np.mean(self.y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.95, f"R² = {r2:.4f}, expected > 0.95"

    def test_evaluate_returns_all_metrics(self):
        model = LinearRegressionModel()
        model.fit(self.X, self.y)
        metrics = model.evaluate(self.X, self.y)
        expected_keys = {"mse", "rmse", "r2", "mae"}
        assert set(metrics.keys()) == expected_keys, \
            f"Expected keys {expected_keys}, got {set(metrics.keys())}"

    def test_evaluate_metrics_values(self):
        model = LinearRegressionModel()
        model.fit(self.X, self.y)
        metrics = model.evaluate(self.X, self.y)
        assert metrics["mse"] >= 0, "MSE should be non-negative"
        assert metrics["rmse"] >= 0, "RMSE should be non-negative"
        assert np.isclose(metrics["rmse"], np.sqrt(metrics["mse"])), \
            "RMSE should be sqrt(MSE)"
        assert metrics["r2"] > 0.95, f"R² should be > 0.95, got {metrics['r2']}"
        assert metrics["mae"] >= 0, "MAE should be non-negative"

    def test_get_coefficients_before_fit_raises(self):
        model = LinearRegressionModel()
        with pytest.raises(RuntimeError):
            model.get_coefficients()

    def test_get_coefficients_structure(self):
        model = LinearRegressionModel()
        model.fit(self.X, self.y, feature_names=self.feature_names)
        coeffs = model.get_coefficients()
        assert "weights" in coeffs
        assert "bias" in coeffs
        assert "feature_importance" in coeffs
        assert len(coeffs["weights"]) == 2
        assert isinstance(coeffs["bias"], float)

    def test_feature_importance_sorted_by_abs_weight(self):
        model = LinearRegressionModel()
        model.fit(self.X, self.y, feature_names=self.feature_names)
        coeffs = model.get_coefficients()
        importance = coeffs["feature_importance"]
        abs_weights = [abs(w) for _, w in importance]
        assert abs_weights == sorted(abs_weights, reverse=True), \
            "Feature importance should be sorted by |weight| descending"


# ---------------------------------------------------------------------------
# Pipeline Tests
# ---------------------------------------------------------------------------

class TestPipeline:
    """Tests for pipeline helper functions."""

    def test_load_data_shapes(self):
        X, y, feature_names = load_data()
        assert X.shape[0] == y.shape[0], "X and y should have same number of samples"
        assert X.shape[1] == len(feature_names), "Number of features should match feature_names"
        assert X.shape[0] > 10000, "California Housing should have >10k samples"

    def test_preprocess_data_standardized(self):
        X = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0], [4.0, 400.0]])
        X_scaled, means, stds = preprocess_data(X)
        # After standardization, each column should have ~0 mean and ~1 std
        np.testing.assert_array_almost_equal(X_scaled.mean(axis=0), [0, 0], decimal=10)
        np.testing.assert_array_almost_equal(X_scaled.std(axis=0), [1, 1], decimal=10)

    def test_preprocess_returns_means_and_stds(self):
        X = np.array([[10.0, 20.0], [30.0, 40.0]])
        _, means, stds = preprocess_data(X)
        np.testing.assert_array_almost_equal(means, [20.0, 30.0])
        assert stds.shape == (2,)

    def test_split_data_sizes(self):
        X = np.random.rand(100, 3)
        y = np.random.rand(100)
        X_train, X_test, y_train, y_test = split_data(X, y, test_ratio=0.2)
        assert X_train.shape[0] == 80
        assert X_test.shape[0] == 20
        assert y_train.shape[0] == 80
        assert y_test.shape[0] == 20

    def test_split_data_reproducible(self):
        X = np.random.rand(100, 3)
        y = np.random.rand(100)
        result1 = split_data(X, y, random_seed=42)
        result2 = split_data(X, y, random_seed=42)
        np.testing.assert_array_equal(result1[0], result2[0],
                                       err_msg="Same seed should give same split")
