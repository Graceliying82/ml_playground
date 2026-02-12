"""
Tests for Multiple Feature Regression — Level 3

These tests validate your implementation in src/multiple_regression/.
Run with: pytest tests/test_multiple_regression.py -v
"""

import numpy as np
import pytest

from src.multiple_regression.model import MultipleRegressionModel
from src.multiple_regression.pipeline import load_data, preprocess_data, split_data


# ---------------------------------------------------------------------------
# Model Tests
# ---------------------------------------------------------------------------

class TestMultipleRegressionModel:
    """Tests for the MultipleRegressionModel class."""

    def setup_method(self):
        """Create multi-feature test data: y = 1*x1 + 2*x2 + 3*x3 + 5."""
        np.random.seed(42)
        m = 300
        self.X = np.random.rand(m, 3)
        self.y = (1 * self.X[:, 0] + 2 * self.X[:, 1] + 3 * self.X[:, 2]
                  + 5 + np.random.randn(m) * 0.1)
        self.feature_names = ["feature_1", "feature_2", "feature_3"]

    def test_fit_returns_self(self):
        model = MultipleRegressionModel()
        result = model.fit(self.X, self.y)
        assert result is model, "fit() should return self"

    def test_is_fitted_after_training(self):
        model = MultipleRegressionModel()
        assert not model.is_fitted
        model.fit(self.X, self.y)
        assert model.is_fitted

    def test_predict_shape(self):
        model = MultipleRegressionModel()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        assert predictions.shape == self.y.shape, \
            f"Expected shape {self.y.shape}, got {predictions.shape}"

    def test_predict_before_fit_raises(self):
        model = MultipleRegressionModel()
        with pytest.raises(RuntimeError):
            model.predict(self.X)

    def test_predictions_are_reasonable(self):
        model = MultipleRegressionModel()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        ss_res = np.sum((self.y - predictions) ** 2)
        ss_tot = np.sum((self.y - np.mean(self.y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.95, f"R² = {r2:.4f}, expected > 0.95"

    def test_evaluate_returns_all_metrics(self):
        model = MultipleRegressionModel()
        model.fit(self.X, self.y)
        metrics = model.evaluate(self.X, self.y)
        expected_keys = {"mse", "rmse", "r2", "mae"}
        assert set(metrics.keys()) == expected_keys, \
            f"Expected keys {expected_keys}, got {set(metrics.keys())}"

    def test_evaluate_metrics_values(self):
        model = MultipleRegressionModel()
        model.fit(self.X, self.y)
        metrics = model.evaluate(self.X, self.y)
        assert metrics["mse"] >= 0, "MSE should be non-negative"
        assert metrics["rmse"] >= 0, "RMSE should be non-negative"
        assert np.isclose(metrics["rmse"], np.sqrt(metrics["mse"])), \
            "RMSE should be sqrt(MSE)"
        assert metrics["r2"] > 0.95, f"R² should be > 0.95, got {metrics['r2']}"
        assert metrics["mae"] >= 0, "MAE should be non-negative"

    def test_get_coefficients_before_fit_raises(self):
        model = MultipleRegressionModel()
        with pytest.raises(RuntimeError):
            model.get_coefficients()

    def test_get_coefficients_structure(self):
        model = MultipleRegressionModel()
        model.fit(self.X, self.y, feature_names=self.feature_names)
        coeffs = model.get_coefficients()
        assert "weights" in coeffs
        assert "bias" in coeffs
        assert "feature_importance" in coeffs
        assert len(coeffs["weights"]) == 3
        assert isinstance(coeffs["bias"], float)

    def test_feature_importance_sorted_by_abs_weight(self):
        model = MultipleRegressionModel()
        model.fit(self.X, self.y, feature_names=self.feature_names)
        coeffs = model.get_coefficients()
        importance = coeffs["feature_importance"]
        abs_weights = [abs(w) for _, w in importance]
        assert abs_weights == sorted(abs_weights, reverse=True), \
            "Feature importance should be sorted by |weight| descending"

    def test_handles_many_features(self):
        """Model should work with 10+ features (like the Diabetes dataset)."""
        np.random.seed(42)
        X_large = np.random.rand(200, 10)
        w_large = np.arange(1, 11, dtype=float)
        y_large = X_large @ w_large + 3.0 + np.random.randn(200) * 0.1
        names = [f"feat_{i}" for i in range(10)]

        model = MultipleRegressionModel()
        model.fit(X_large, y_large, feature_names=names)
        coeffs = model.get_coefficients()

        assert len(coeffs["weights"]) == 10
        assert len(coeffs["feature_importance"]) == 10


# ---------------------------------------------------------------------------
# Pipeline Tests
# ---------------------------------------------------------------------------

class TestPipeline:
    """Tests for pipeline helper functions."""

    def test_load_data_shapes(self):
        X, y, feature_names = load_data()
        assert X.shape[0] == y.shape[0], \
            "X and y should have same number of samples"
        assert X.shape[1] == len(feature_names), \
            "Number of features should match feature_names"
        assert X.shape[0] > 400, "Diabetes dataset should have >400 samples"
        assert X.shape[1] == 10, "Diabetes dataset should have 10 features"

    def test_load_data_feature_names(self):
        _, _, feature_names = load_data()
        assert "bmi" in feature_names, "Feature names should include 'bmi'"
        assert len(feature_names) == 10

    def test_preprocess_data_standardized(self):
        X = np.array([
            [1.0, 100.0, 10.0],
            [2.0, 200.0, 20.0],
            [3.0, 300.0, 30.0],
            [4.0, 400.0, 40.0],
        ])
        X_scaled, means, stds = preprocess_data(X)
        np.testing.assert_array_almost_equal(
            X_scaled.mean(axis=0), [0, 0, 0], decimal=10
        )
        np.testing.assert_array_almost_equal(
            X_scaled.std(axis=0), [1, 1, 1], decimal=10
        )

    def test_preprocess_returns_means_and_stds(self):
        X = np.array([[10.0, 20.0, 30.0], [30.0, 40.0, 50.0]])
        _, means, stds = preprocess_data(X)
        np.testing.assert_array_almost_equal(means, [20.0, 30.0, 40.0])
        assert stds.shape == (3,)

    def test_preprocess_handles_zero_std(self):
        """Constant feature (zero std) should not cause division by zero."""
        X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
        X_scaled, _, stds = preprocess_data(X)
        assert not np.any(np.isnan(X_scaled)), \
            "Scaling should not produce NaN (handle zero std)"
        assert not np.any(np.isinf(X_scaled)), \
            "Scaling should not produce Inf (handle zero std)"

    def test_split_data_sizes(self):
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        X_train, X_test, y_train, y_test = split_data(X, y, test_ratio=0.2)
        assert X_train.shape[0] == 80
        assert X_test.shape[0] == 20
        assert y_train.shape[0] == 80
        assert y_test.shape[0] == 20

    def test_split_data_preserves_features(self):
        """Split should preserve the number of features."""
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        X_train, X_test, _, _ = split_data(X, y)
        assert X_train.shape[1] == 5
        assert X_test.shape[1] == 5

    def test_split_data_reproducible(self):
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        result1 = split_data(X, y, random_seed=42)
        result2 = split_data(X, y, random_seed=42)
        np.testing.assert_array_equal(
            result1[0], result2[0],
            err_msg="Same seed should give same split"
        )
