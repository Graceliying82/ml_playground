"""
Multiple Feature Regression Pipeline — Level 3: Real-World Application

This module handles the end-to-end workflow:
1. Load a real dataset (Diabetes)
2. Explore and preprocess the data
3. Split into train/test sets
4. Train the model
5. Evaluate and report results

Your task: Implement the TODO sections.
Run tests with: pytest tests/test_multiple_regression.py
Run the full pipeline: python -m src.multiple_regression.pipeline
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from .model import MultipleRegressionModel
from .display import show_sample_predictions, plot_results


def load_data():
    """Load the Diabetes dataset.

    This dataset contains 442 samples with 10 features each.
    The target is a quantitative measure of disease progression
    one year after baseline.

    Features:
        - age: Age of the patient
        - sex: Sex of the patient
        - bmi: Body mass index
        - bp: Average blood pressure
        - s1: Total serum cholesterol (tc)
        - s2: Low-density lipoproteins (ldl)
        - s3: High-density lipoproteins (hdl)
        - s4: Total cholesterol / HDL (tch)
        - s5: Log of serum triglycerides level (ltg)
        - s6: Blood sugar level (glu)

    Returns:
        tuple: (X, y, feature_names)
            - X (np.ndarray): Features, shape (442, 10)
            - y (np.ndarray): Target values, shape (442,)
            - feature_names (list[str]): Names of the 10 features
    """
    # Step 1: Import load_diabetes from sklearn.datasets
    # Step 2: Call load_diabetes() to get the data
    # Step 3: Return X (data.data), y (data.target), feature_names (data.feature_names)

    data = load_diabetes()

    X = data.data
    y = data.target
    print(f"  Loaded Diabetes dataset with {X.shape[0]} samples and {X.shape[1]} features.")
    print(f"  Feature names: {data.feature_names}")
    
    names = data.feature_names

    return X, y, names


def explore_data(X, y, feature_names):
    """Print basic statistics about the dataset.

    Args:
        X (np.ndarray): Features, shape (m, n_features).
        y (np.ndarray): Targets, shape (m,).
        feature_names (list[str]): Feature names.
    """
    # TODO: Print a data exploration summary
    # Include:
    # - Dataset shape (number of samples and features)
    # - Feature names
    # - Target statistics (min, max, mean, std)
    # - Feature statistics (min, max, mean for each feature)
    # - Check for any missing values (np.isnan)

    print(f"  Dataset shape: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Feature names: {feature_names}")
    print(f"  Target statistics:")
    print(f"    Min: {y.min():.2f}")
    print(f"    Max: {y.max():.2f}")
    print(f"    Mean: {y.mean():.2f}")
    print(f"    Std: {y.std():.2f}")
    print(f"  Feature statistics:")
    for i, name in enumerate(feature_names):
        print(f"    {name}: min={X[:, i].min():.2f}, max={X[:, i].max():.2f}, mean={X[:, i].mean():.2f}")

    #Check for any missing values (np.isnan)
    if np.isnan(X).any():
        print("  Warning: Missing values detected in features.")

def preprocess_data(X):
    """Standardize features to zero mean and unit variance.

    Standardization formula: X_scaled = (X - mean) / std

    Args:
        X (np.ndarray): Raw features, shape (m, n_features).

    Returns:
        tuple: (X_scaled, means, stds)
            - X_scaled (np.ndarray): Standardized features
            - means (np.ndarray): Mean of each feature (for later use)
            - stds (np.ndarray): Std of each feature (for later use)
    """
    # TODO: Implement feature standardization
    # Step 1: Compute the mean of each feature (column)
    # Step 2: Compute the std of each feature (column)
    # Step 3: Handle zero std by replacing with 1.0
    # Step 4: Standardize: X_scaled = (X - means) / stds
    # Step 5: Return (X_scaled, means, stds)

    means = X.mean(axis=0)
    stds = X.std(axis=0)

    # Handle zero std by replacing with 1.0
    stds[stds == 0] = 1.0

    X_scaled = (X - means) / stds

    return X_scaled, means, stds


def split_data(X, y, test_ratio=0.2, random_seed=42):
    """Split data into training and test sets.

    Args:
        X (np.ndarray): Features, shape (m, n_features).
        y (np.ndarray): Targets, shape (m,).
        test_ratio (float): Fraction of data for testing (0.0 to 1.0).
        random_seed (int): Random seed for reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # TODO: Split the data
    # Step 1: Import train_test_split from sklearn.model_selection
    # Step 2: Split X and y with the given test_ratio and random_seed
    # Step 3: Return (X_train, X_test, y_train, y_test)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=random_seed)

    return X_train, X_test, y_train, y_test

def run_pipeline():
    """Run the full multiple feature regression pipeline.

    This is the main entry point that ties everything together.
    """

    print("=" * 60)
    print("  Multiple Regression Pipeline — Diabetes Dataset")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/5] Loading data...")
    X, y, feature_names = load_data()

    # Step 2: Explore
    print("\n[2/5] Exploring data...")
    explore_data(X, y, feature_names)

    # Step 3: Preprocess
    print("\n[3/5] Preprocessing data...")
    X_scaled, means, stds = preprocess_data(X)

    # Step 4: Split
    print("\n[4/5] Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    print(f"  Train set: {X_train.shape[0]} samples")
    print(f"  Test set:  {X_test.shape[0]} samples")

    # Step 5: Train and evaluate
    print("\n[5/5] Training and evaluating model...")
    model = MultipleRegressionModel()
    model.fit(X_train, y_train, feature_names=list(feature_names))

    # Evaluate on both sets
    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)

    print("\n--- Results ---")
    print(f"  {'Metric':<10} {'Train':>10} {'Test':>10}")
    print(f"  {'-'*30}")
    for key in ["mse", "rmse", "r2", "mae"]:
        print(
            f"  {key.upper():<10} {train_metrics[key]:>10.4f} {test_metrics[key]:>10.4f}"
        )

    # Feature importance
    coeffs = model.get_coefficients()
    print("\n--- Feature Importance (by |weight|) ---")
    for name, weight in coeffs["feature_importance"]:
        bar = "#" * int(abs(weight) * 5)
        print(f"  {name:<15} {weight:>8.4f}  {bar}")

    # --- Sample Predictions & Plots ---
    predictions = model.predict(X_test)
    show_sample_predictions(y_test, predictions)
    plot_results(y_test, predictions, coeffs["feature_importance"],
                 save_path="multiple_regression_results.png")

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
