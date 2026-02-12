# ML Playground - Claude Code Context

## Project Overview
A hands-on ML learning repository following a 3-level progression per algorithm:
- **Level 1**: Implement from scratch with NumPy (Jupyter notebooks)
- **Level 2**: Reimplement with scikit-learn (Jupyter notebooks)
- **Level 3**: Production-style Python modules with tests (`src/` + `tests/`)

## Project Structure
- `notebooks/<topic>/` - Jupyter notebooks for Level 1 & 2
- `src/<topic>/model.py` - Model class with `fit`, `predict`, `evaluate`, `get_coefficients`
- `src/<topic>/pipeline.py` - Data loading, preprocessing, train/test split, full pipeline
- `tests/test_<topic>.py` - pytest test suites
- `studynotes/` - NumPy, scikit-learn, matplotlib cheat sheets
- `data/` - Datasets (large files excluded from git)

## Tech Stack
- **Language**: Python 3
- **Core libs**: numpy, pandas, matplotlib, scikit-learn, seaborn
- **Notebooks**: Jupyter
- **Testing**: pytest
- **Virtual env**: `.venv/` (activate with `source .venv/bin/activate`)

## Common Commands
```bash
# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_linear_regression.py -v

# Run a pipeline
python -m src.linear_regression.pipeline
python -m src.multiple_regression.pipeline

# Launch notebooks
jupyter notebook
```

## Conventions
- Each model class wraps scikit-learn and exposes: `fit()`, `predict()`, `evaluate()`, `get_coefficients()`
- `evaluate()` returns a dict with keys: `mse`, `rmse`, `r2`, `mae`
- Pipeline modules have: `load_data()`, `explore_data()`, `preprocess_data()`, `split_data()`, `run_pipeline()`
- Tests expect R² > 0.95 for fitted models
- Use `random_seed=42` for reproducibility
- Preprocessing: standardize features to zero mean and unit variance

## Current Topics
- `01_linear_regression` - Complete (single-feature, California Housing dataset)
- `02_multiple_regression` - In progress (multi-feature, Diabetes dataset)

## Model Escalation Strategy (Budget-First)

Default model is **Haiku** (cheapest). Escalate only when needed:

```
Haiku (default) → Sonnet → Opus (last resort)
```

| Step | Model | Switch Command | When to Use |
|------|-------|----------------|-------------|
| 1 | **Haiku** | _(default)_ | Simple edits, Q&A, running tests, file lookups, typo fixes |
| 2 | **Sonnet** | `/model sonnet` | First attempt didn't work, or task needs real coding (implement methods, write tests, debug) |
| 3 | **Opus** | `/model opus` | Sonnet wasn't good enough, or task is complex (multi-file refactor, architecture, tricky bugs) |

After finishing a complex task, switch back: `/model haiku`

## Cost-Saving Reminders
- When the conversation gets long (many back-and-forth exchanges), remind the user: **"This chat is getting long — consider starting a new chat (`/clear` or open a new session) to keep costs down. CLAUDE.md will carry over all the project context automatically."**
- One topic per chat session. If the user switches to a completely different task, suggest starting fresh.
- Prefer short, focused conversations over long sprawling ones — long contexts cost more tokens on every message.
