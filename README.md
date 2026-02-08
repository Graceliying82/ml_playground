# Machine Learning Playground

A hands-on practice repository for the **Machine Learning Specialization** course. Each topic is structured in three progressive levels to build deep understanding from scratch to real-world application.

## Learning Approach

Each algorithm module follows a **3-level progression**:

| Level | Goal | Format | What You Do |
|-------|------|--------|-------------|
| **Level 1 - From Scratch** | Understand the math | Jupyter Notebook | Implement the algorithm using only NumPy |
| **Level 2 - scikit-learn** | Learn the industry tool | Jupyter Notebook | Reimplement using scikit-learn and compare results |
| **Level 3 - Real-World** | Build production-style code | Python module + tests | Solve a real problem with proper project structure |

## Project Structure

```
ml_playground/
├── notebooks/                        # Level 1 & 2: Interactive learning
│   ├── 01_linear_regression/
│   │   └── linear_regression.ipynb
│   ├── 02_logistic_regression/       (coming soon)
│   └── ...
├── src/                              # Level 3: Production-style Python modules
│   ├── linear_regression/
│   │   ├── __init__.py
│   │   ├── model.py                  # Your model implementation
│   │   └── pipeline.py              # Data loading, training, evaluation
│   └── ...
├── tests/                            # Unit tests for src/ modules
│   ├── __init__.py
│   └── test_linear_regression.py
├── data/                             # Datasets
├── utils/                            # Shared helper functions
├── requirements.txt
└── README.md
```

## Topics Roadmap

### Supervised Learning
- [x] Linear Regression
- [ ] Logistic Regression
- [ ] Neural Networks
- [ ] Decision Trees
- [ ] Support Vector Machines

### Unsupervised Learning
- [ ] K-Means Clustering
- [ ] Principal Component Analysis (PCA)
- [ ] Anomaly Detection

### Special Topics
- [ ] Recommender Systems
- [ ] Reinforcement Learning

## Getting Started

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter (for Level 1 & 2)
jupyter notebook

# Run tests (for Level 3)
pytest tests/
```

## How to Practice

### Level 1 & 2 (Notebooks)
1. Read the **concept summary** at the top of each notebook.
2. Follow the instructions in each cell marked with `# TODO`.
3. Write your implementation where indicated.
4. Run the validation cells to check your work.
5. Compare your Level 1 (from-scratch) results with Level 2 (scikit-learn) results.

### Level 3 (src/ modules)
1. Read the docstrings and TODOs in the Python files under `src/`.
2. Implement the functions and classes.
3. Run `pytest tests/` to validate your implementation.
4. Use real-world data to train and evaluate your model.
