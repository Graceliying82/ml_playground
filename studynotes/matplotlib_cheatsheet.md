# Matplotlib Pyplot Cheat Sheet

A quick reference for creating visualizations commonly used in machine learning.

---

## 1. Import

```python
import matplotlib.pyplot as plt
import numpy as np
```

## 2. Basic Plot Structure

```python
# Quick plot (simplest)
plt.plot(x, y)
plt.show()

# Recommended structure (more control)
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title("Title")
ax.set_xlabel("X")
ax.set_ylabel("Y")
plt.show()
```

### plt vs ax — What's the Difference?

```
plt.plot(...)         # Uses the "current" figure/axes (quick and simple)
ax.plot(...)          # Uses a specific axes object (better for subplots)

# Rule of thumb:
# - Single plot → plt.plot() is fine
# - Multiple subplots → use fig, ax = plt.subplots()
```

## 3. Line Plot

```python
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)                             # Basic
plt.plot(x, y, color="red")               # Color
plt.plot(x, y, linewidth=2)               # Thickness
plt.plot(x, y, linestyle="--")            # Dashed
plt.plot(x, y, "r--")                     # Shorthand: red dashed
plt.plot(x, y, "bo-")                     # Blue circles with line
```

### Common Line Styles & Markers

| Style | Meaning | Marker | Meaning |
|-------|---------|--------|---------|
| `"-"` | Solid | `"o"` | Circle |
| `"--"` | Dashed | `"s"` | Square |
| `"-."` | Dash-dot | `"^"` | Triangle |
| `":"` | Dotted | `"x"` | X mark |

### Common Colors

| Short | Name |
|-------|------|
| `"b"` | Blue |
| `"r"` | Red |
| `"g"` | Green |
| `"k"` | Black |
| `"m"` | Magenta |
| `"c"` | Cyan |

## 4. Scatter Plot

```python
plt.scatter(x, y)                          # Basic
plt.scatter(x, y, alpha=0.5)              # Transparency (0-1)
plt.scatter(x, y, s=10)                   # Marker size
plt.scatter(x, y, c=colors, cmap="viridis")  # Color by value
plt.colorbar()                             # Show color scale
```

### ML Use Case: Data with Labels

```python
# Plot two classes with different colors
plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Class 0", alpha=0.6)
plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Class 1", alpha=0.6)
plt.legend()
```

## 5. Histogram

```python
plt.hist(data, bins=30)                    # Basic
plt.hist(data, bins=30, alpha=0.7)        # With transparency
plt.hist(data, bins=30, edgecolor="black") # With borders
plt.hist(data, density=True)              # Normalized (area = 1)

# Compare two distributions
plt.hist(data1, bins=30, alpha=0.5, label="Train")
plt.hist(data2, bins=30, alpha=0.5, label="Test")
plt.legend()
```

## 6. Bar Chart

```python
categories = ["A", "B", "C", "D"]
values = [23, 45, 12, 67]

plt.bar(categories, values)                # Vertical
plt.barh(categories, values)               # Horizontal
```

## 7. Labels, Title & Legend

```python
plt.title("My Plot", fontsize=14)
plt.xlabel("X axis label")
plt.ylabel("Y axis label")

# Legend (requires label= in each plot call)
plt.plot(x, y1, label="Training")
plt.plot(x, y2, label="Validation")
plt.legend()                               # Auto position
plt.legend(loc="upper right")             # Manual position
plt.legend(loc="lower left", fontsize=10)
```

### Legend Location Options

```
"best", "upper right", "upper left", "lower left",
"lower right", "center", "center left", "center right"
```

## 8. Axis Controls

```python
plt.xlim(0, 10)                # Set x-axis range
plt.ylim(-1, 1)               # Set y-axis range
plt.xticks([0, 2, 4, 6, 8])   # Custom tick positions
plt.yticks(np.arange(-1, 1.5, 0.5))
plt.grid(True)                 # Show grid
plt.grid(True, alpha=0.3)     # Subtle grid
```

## 9. Subplots

```python
# Basic: 1 row, 2 columns
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(x, y1)
axes[0].set_title("Plot 1")

axes[1].scatter(x, y2)
axes[1].set_title("Plot 2")

plt.tight_layout()             # Prevent overlap
plt.show()
```

```python
# 2 rows, 2 columns
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0, 0].plot(x, y)         # Top-left
axes[0, 1].scatter(x, y)      # Top-right
axes[1, 0].hist(data)         # Bottom-left
axes[1, 1].bar(cats, vals)    # Bottom-right

plt.tight_layout()
plt.show()
```

```python
# Share axes (useful for comparing)
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
```

## 10. Figure Size & Style

```python
# Set figure size
plt.figure(figsize=(10, 6))    # Width x Height in inches

# Built-in styles
plt.style.use("seaborn-v0_8")          # Clean style
plt.style.use("ggplot")                # R-like style
plt.style.use("default")              # Reset to default
print(plt.style.available)             # List all styles
```

## 11. Annotations & Text

```python
# Add text at a position
plt.text(x, y, "label", fontsize=12)

# Arrow annotation
plt.annotate("Important!", xy=(x, y), xytext=(x+1, y+1),
             arrowprops=dict(arrowstyle="->"))

# Horizontal/vertical lines
plt.axhline(y=0, color="gray", linestyle="--")    # Horizontal
plt.axvline(x=5, color="gray", linestyle="--")    # Vertical
```

## 12. Saving Figures

```python
plt.savefig("plot.png")                    # PNG (default)
plt.savefig("plot.png", dpi=150)          # Higher resolution
plt.savefig("plot.pdf")                    # Vector format
plt.savefig("plot.png", bbox_inches="tight")  # No extra whitespace

# IMPORTANT: call savefig() BEFORE show()
plt.savefig("plot.png")
plt.show()
```

## 13. Common ML Visualizations

### Cost Function Convergence

```python
plt.plot(range(len(cost_history)), cost_history)
plt.xlabel("Iteration")
plt.ylabel("Cost J")
plt.title("Gradient Descent Convergence")
plt.grid(True, alpha=0.3)
```

### Linear Regression Fit

```python
plt.scatter(X, y, alpha=0.5, label="Data")
plt.plot(X, predictions, color="red", linewidth=2, label="Fit")
plt.legend()
```

### Residual Plot

```python
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
```

### Predicted vs Actual

```python
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "r--", linewidth=2)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Predicted vs Actual")
```

### Feature Importance (Horizontal Bar)

```python
features = ["Income", "Age", "Rooms"]
weights = [0.85, 0.12, 0.45]

# Sort by absolute value
order = np.argsort(np.abs(weights))
plt.barh(np.array(features)[order], np.array(weights)[order])
plt.xlabel("Weight")
plt.title("Feature Importance")
```

### Heatmap (Correlation Matrix)

```python
import seaborn as sns

corr = df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Feature Correlation Matrix")
```

## 14. Quick Reference Table

| What | Code |
|------|------|
| New figure | `plt.figure(figsize=(w, h))` |
| Line plot | `plt.plot(x, y)` |
| Scatter | `plt.scatter(x, y)` |
| Histogram | `plt.hist(data, bins=30)` |
| Bar chart | `plt.bar(x, heights)` |
| Title | `plt.title("text")` |
| X label | `plt.xlabel("text")` |
| Y label | `plt.ylabel("text")` |
| Legend | `plt.legend()` |
| Grid | `plt.grid(True)` |
| Subplots | `fig, axes = plt.subplots(rows, cols)` |
| Tight layout | `plt.tight_layout()` |
| Save | `plt.savefig("file.png")` |
| Show | `plt.show()` |
