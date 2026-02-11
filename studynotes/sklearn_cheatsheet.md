# scikit-learn Cheat Sheet

A quick reference for scikit-learn, focused on the workflow and classes used in machine learning.

---

## 1. Import Convention

```python
# Don't import the whole library — import what you need
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
```

## 2. The Universal API Pattern

Every scikit-learn model follows the same 3-step pattern:

```python
# Step 1: Create
model = SomeModel(hyperparameters)

# Step 2: Fit (train)
model.fit(X_train, y_train)

# Step 3: Predict
y_pred = model.predict(X_test)
```

This works for **every** algorithm — linear regression, logistic regression, decision trees, SVMs, etc. Learn this once, use it everywhere.

## 3. Data Format Rules

```python
# X must be 2D: shape (n_samples, n_features)
# y must be 1D: shape (n_samples,)

# Common fix for single-feature data:
X = data.reshape(-1, 1)    # (100,) → (100, 1)
```

## 4. Train/Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% for testing
    random_state=42      # Reproducible split
)
```

| Parameter | Purpose |
|-----------|---------|
| `test_size` | Fraction for test set (0.2 = 20%) |
| `random_state` | Seed for reproducibility |
| `shuffle` | Shuffle before splitting (default: True) |
| `stratify` | Preserve class proportions (for classification) |

## 5. Preprocessing

### Feature Scaling

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit + transform on train
X_test_scaled = scaler.transform(X_test)         # Only transform on test!
```

| Scaler | Formula | When to Use |
|--------|---------|-------------|
| `StandardScaler` | (x - mean) / std | Most algorithms (default choice) |
| `MinMaxScaler` | (x - min) / (max - min) | When you need [0, 1] range |
| `RobustScaler` | (x - median) / IQR | When data has outliers |

**Important**: Always `fit_transform()` on training data, then `transform()` on test data. Never fit on test data — that's data leakage.

### Encoding Categorical Variables

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Label encoding (for ordinal categories)
le = LabelEncoder()
y_encoded = le.fit_transform(["cat", "dog", "cat"])  # [0, 1, 0]

# One-hot encoding (for nominal categories)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
X_encoded = ohe.fit_transform(X_categorical)
```

## 6. Linear Models

### Linear Regression (Regression)

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

model.coef_          # Weights (one per feature)
model.intercept_     # Bias term
y_pred = model.predict(X_test)
```

### Logistic Regression (Classification)

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)           # Class labels
y_prob = model.predict_proba(X_test)     # Class probabilities
```

## 7. Other Common Models

```python
# Decision Tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Random Forest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Support Vector Machine
from sklearn.svm import SVC, SVR

# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# All follow the same API:
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## 8. Evaluation Metrics

### Regression Metrics

```python
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

mse  = mean_squared_error(y_test, y_pred)       # Lower is better
rmse = mean_squared_error(y_test, y_pred, squared=False)  # In original units
mae  = mean_absolute_error(y_test, y_pred)       # Lower is better
r2   = r2_score(y_test, y_pred)                  # 1.0 = perfect, 0 = useless
```

| Metric | What It Tells You |
|--------|-------------------|
| **MSE** | Average squared error (penalizes large errors) |
| **RMSE** | MSE in original units (more interpretable) |
| **MAE** | Average absolute error (robust to outliers) |
| **R²** | Fraction of variance explained (0 to 1) |

### Classification Metrics

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

# Full report
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
```

| Metric | Formula | When to Use |
|--------|---------|-------------|
| **Accuracy** | Correct / Total | Balanced classes |
| **Precision** | TP / (TP + FP) | Cost of false positives is high |
| **Recall** | TP / (TP + FN) | Cost of false negatives is high |
| **F1** | 2 * P * R / (P + R) | Imbalanced classes |

## 9. Cross-Validation

Instead of a single train/test split, evaluate across multiple folds:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring="r2")

print(f"R² scores: {scores}")
print(f"Mean: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

| Parameter | Purpose |
|-----------|---------|
| `cv=5` | 5-fold cross-validation |
| `scoring` | Metric to use ("r2", "accuracy", "neg_mean_squared_error") |

## 10. Pipelines

Chain preprocessing and model into a single object:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

pipe.fit(X_train, y_train)        # Scales then fits
y_pred = pipe.predict(X_test)     # Scales then predicts
```

Pipelines prevent data leakage and keep your code clean.

## 11. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10, None]
}

grid = GridSearchCV(
    RandomForestRegressor(),
    param_grid,
    cv=5,
    scoring="r2"
)
grid.fit(X_train, y_train)

print(grid.best_params_)     # Best hyperparameters
print(grid.best_score_)      # Best cross-validation score
best_model = grid.best_estimator_
```

## 12. Loading Built-in Datasets

```python
from sklearn.datasets import (
    fetch_california_housing,   # Regression: house prices
    load_iris,                  # Classification: 3 flower types
    load_digits,                # Classification: handwritten digits
    make_classification,        # Generate synthetic classification data
    make_regression             # Generate synthetic regression data
)

# Example
data = fetch_california_housing()
X, y = data.data, data.target
feature_names = data.feature_names
print(data.DESCR)              # Dataset description
```

## 13. Model Persistence (Save/Load)

```python
import joblib

# Save
joblib.dump(model, "model.pkl")

# Load
model = joblib.load("model.pkl")
```

## 14. Quick Reference: Complete Workflow

```python
# 1. Load data
X, y = load_data()

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. Preprocess
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Train
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict
y_pred = model.predict(X_test)

# 6. Evaluate
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")
print(f"R²:   {r2_score(y_test, y_pred):.4f}")
```
