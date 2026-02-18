"""
Multiple Regression Display Module

Visualization and display utilities for the multiple regression pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt


def show_sample_predictions(y_actual, y_predicted, n=10):
    """Print a table of sample predictions vs actual values.

    Args:
        y_actual (np.ndarray): True target values.
        y_predicted (np.ndarray): Model predictions.
        n (int): Number of samples to show.
    """
    print("\n--- Sample Predictions (first {} test patients) ---".format(n))
    print(f"  {'Patient':<10} {'Actual':>10} {'Predicted':>10} {'Error':>10}")
    print(f"  {'-'*40}")
    for i in range(min(n, len(y_actual))):
        error = y_predicted[i] - y_actual[i]
        print(f"  {'#' + str(i+1):<10} {y_actual[i]:>10.1f} {y_predicted[i]:>10.1f} {error:>+10.1f}")


def plot_results(y_actual, y_predicted, feature_importance, save_path=None):
    """Plot actual vs predicted, residuals, and feature importance.

    Args:
        y_actual (np.ndarray): True target values.
        y_predicted (np.ndarray): Model predictions.
        feature_importance (list[tuple]): List of (feature_name, weight) tuples.
        save_path (str, optional): If provided, save the figure to this path.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Actual vs Predicted
    axes[0].scatter(y_actual, y_predicted, alpha=0.6, edgecolors='k', linewidths=0.5)
    min_val = min(y_actual.min(), y_predicted.min())
    max_val = max(y_actual.max(), y_predicted.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    axes[0].set_xlabel('Actual Disease Progression')
    axes[0].set_ylabel('Predicted Disease Progression')
    axes[0].set_title('Actual vs Predicted')
    axes[0].legend()

    # Plot 2: Residuals
    residuals = y_actual - y_predicted
    axes[1].scatter(y_predicted, residuals, alpha=0.6, edgecolors='k', linewidths=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('Predicted Disease Progression')
    axes[1].set_ylabel('Residual (Actual - Predicted)')
    axes[1].set_title('Residual Plot')

    # Plot 3: Feature Importance
    names = [name for name, _ in feature_importance]
    weights = [w for _, w in feature_importance]
    colors = ['green' if w > 0 else 'red' for w in weights]
    axes[2].barh(names, weights, color=colors)
    axes[2].set_xlabel('Weight')
    axes[2].set_title('Feature Importance (green=increases, red=decreases progression)')
    axes[2].invert_yaxis()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"\n  Plot saved to {save_path}")

    plt.close(fig)
