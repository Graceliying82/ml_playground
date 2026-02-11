# NumPy Cheat Sheet

A quick reference for NumPy operations commonly used in machine learning.

---

## 1. Import

```python
import numpy as np
```

## 2. Creating Arrays

```python
# From Python lists
a = np.array([1, 2, 3])                  # 1D array
b = np.array([[1, 2], [3, 4]])            # 2D array (matrix)

# Common constructors
np.zeros((3, 4))                          # 3x4 matrix of zeros
np.ones((2, 3))                           # 2x3 matrix of ones
np.full((2, 2), 7)                        # 2x2 matrix filled with 7
np.eye(3)                                 # 3x3 identity matrix

# Sequences
np.arange(0, 10, 2)                       # [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)                      # [0, 0.25, 0.5, 0.75, 1.0]

# Random
np.random.rand(3, 2)                      # 3x2 uniform random [0, 1)
np.random.randn(3, 2)                     # 3x2 standard normal
np.random.randint(0, 10, size=(3, 2))     # 3x2 random integers [0, 10)
np.random.seed(42)                        # Set seed for reproducibility
```

## 3. Array Properties

```python
a.shape       # Dimensions, e.g. (3, 4)
a.ndim        # Number of dimensions, e.g. 2
a.size        # Total number of elements, e.g. 12
a.dtype       # Data type, e.g. float64
```

## 4. Reshaping

```python
a = np.arange(6)                          # [0, 1, 2, 3, 4, 5]

a.reshape(2, 3)                           # [[0,1,2], [3,4,5]]
a.reshape(-1, 1)                          # Column vector (6, 1) — useful for sklearn
a.reshape(1, -1)                          # Row vector (1, 6)
a.flatten()                               # Back to 1D
a.T                                       # Transpose
```

## 5. Indexing & Slicing

```python
a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

a[0, 1]                # Single element: 2
a[0]                   # First row: [1, 2, 3]
a[:, 0]                # First column: [1, 4, 7]
a[0:2, 1:]             # Subarray: [[2,3], [5,6]]
a[:, -1]               # Last column: [3, 6, 9]

# Boolean indexing
a[a > 5]               # [6, 7, 8, 9]
a[a % 2 == 0]          # [2, 4, 6, 8]
```

## 6. Element-wise Operations

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

a + b                   # [5, 7, 9]
a - b                   # [-3, -3, -3]
a * b                   # [4, 10, 18]   (element-wise, NOT dot product)
a / b                   # [0.25, 0.4, 0.5]
a ** 2                  # [1, 4, 9]
np.sqrt(a)              # [1.0, 1.414, 1.732]
np.exp(a)               # [2.718, 7.389, 20.086]
np.log(a)               # [0.0, 0.693, 1.099]
```

## 7. Linear Algebra (ML Essentials)

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Dot product — used everywhere in ML
np.dot(a, b)                    # 32  (1*4 + 2*5 + 3*6)
a @ b                           # 32  (same thing, cleaner syntax)

# Matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
A @ B                           # [[19,22], [43,50]]
np.matmul(A, B)                 # Same as above

# Other
np.linalg.inv(A)                # Matrix inverse
np.linalg.norm(a)               # Euclidean norm (L2)
np.linalg.det(A)                # Determinant
```

## 8. Aggregation Functions

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

np.sum(a)               # 21 (all elements)
np.sum(a, axis=0)       # [5, 7, 9]   (sum each column)
np.sum(a, axis=1)       # [6, 15]     (sum each row)

np.mean(a)              # 3.5
np.mean(a, axis=0)      # [2.5, 3.5, 4.5]  (mean of each column)

np.std(a)               # Standard deviation
np.var(a)               # Variance
np.min(a)               # 1
np.max(a)               # 6
np.argmin(a)            # Index of min (flattened)
np.argmax(a)            # Index of max (flattened)
```

### axis= Explained

```
axis=0 → operate DOWN columns (collapse rows)
axis=1 → operate ACROSS rows (collapse columns)

     col0  col1  col2
      ↓     ↓     ↓       axis=0: result has shape (3,)
row0 [1,    2,    3 ] →   axis=1: result has shape (2,)
row1 [4,    5,    6 ] →
```

## 9. Broadcasting

NumPy automatically expands smaller arrays to match larger ones in arithmetic.

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])     # Shape (2, 3)

# Scalar broadcast
A + 10                         # [[11,12,13], [14,15,16]]

# Vector broadcast (row)
row = np.array([10, 20, 30])   # Shape (3,)
A + row                        # [[11,22,33], [14,25,36]]

# Vector broadcast (column)
col = np.array([[10], [20]])   # Shape (2, 1)
A + col                        # [[11,12,13], [24,25,26]]
```

### Broadcasting Rules
1. Dimensions are compared from right to left.
2. Dimensions match if they are equal or one of them is 1.
3. If one array has fewer dimensions, it's padded with 1s on the left.

## 10. Common ML Patterns

### Vectorized prediction (linear regression)
```python
# Instead of looping:
# f_wb = w * X + b
predictions = np.dot(X, w) + b           # For multi-feature
predictions = w * X + b                   # For single feature
```

### Vectorized cost (MSE)
```python
errors = predictions - y
cost = np.sum(errors ** 2) / (2 * m)
```

### Vectorized gradient
```python
dj_dw = np.dot(errors, X) / m            # For single feature
dj_db = np.sum(errors) / m
```

### Feature standardization
```python
means = X.mean(axis=0)
stds = X.std(axis=0)
X_normalized = (X - means) / stds
```

### Train/test split (manual)
```python
indices = np.random.permutation(m)
split = int(0.8 * m)
train_idx, test_idx = indices[:split], indices[split:]
X_train, X_test = X[train_idx], X[test_idx]
```

## 11. Useful Utilities

```python
np.isnan(a).any()               # Check for NaN values
np.where(a > 3, a, 0)          # Replace elements ≤ 3 with 0
np.clip(a, 0, 10)              # Clamp values to [0, 10]
np.concatenate([a, b], axis=0) # Stack arrays
np.vstack([a, b])              # Vertical stack
np.hstack([a, b])              # Horizontal stack
np.unique(a)                   # Unique values
np.sort(a)                     # Sorted copy
np.copy(a)                     # Deep copy (avoid aliasing bugs)
```
