# GradeMiner

A student academic performance prediction tool built with Python and scikit-learn. Uses demographic, social, and academic data to predict a student's final grade and classify whether they will pass or fail.

Built using the [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance).

---

## What It Does

- **Regression** — Predicts a student's final grade (G3) using Linear Regression
- **Classification** — Classifies students as Pass/Fail using Logistic Regression, KNN, and Decision Tree
- **Feature Analysis** — Identifies which factors most influence academic performance
- **Model Comparison** — Evaluates and compares all models with appropriate metrics

---

## Project Structure

```
GradeMiner/
├── data/
│   ├── raw/                  # Original CSV files (student-mat.csv, student-por.csv)
│   └── processed/            # Reserved for processed data outputs
├── models/
│   ├── trained/              # Reserved for saved model files
│   └── results/              # Generated plots and results CSV
├── src/
│   ├── main.py               # Entry point — runs the full pipeline
│   ├── student.py            # Student dataclass
│   ├── data_preprocessor.py  # Loading, encoding, scaling, splitting
│   ├── grade_regressor.py    # Linear Regression for G3 prediction
│   ├── pass_fail_classifier.py  # 3 classifiers for Pass/Fail
│   ├── feature_analyzer.py   # Feature importance analysis & plots
│   └── model_evaluator.py    # Shared metrics and result formatting
├── .gitignore
└── README.md
```

---

## Setup

**Requirements:** Python 3.9+

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd GradeMiner

# 2. Create and activate a virtual environment
python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

# 3. Install dependencies
pip install pandas numpy scikit-learn matplotlib
```

---

## Running the Project

```bash
cd src
python main.py
```

Output files are saved to `models/results/`:
- `regression_actual_vs_predicted.png`
- `classifier_comparison.png`
- `feature_importance_regression.png`
- `feature_importance_logistic.png`
- `feature_importance_tree.png`
- `model_results_summary.csv`

---

## Configuration

Both settings live at the top of their respective files and are easy to change:

| Setting | File | Default |
|---|---|---|
| Switch dataset (Math vs Portuguese) | `main.py` → `DATASET` | `student-mat.csv` |
| Pass/Fail threshold | `data_preprocessor.py` → `PASS_THRESHOLD` | `10` |

---

## Dataset

**Source:** UCI Machine Learning Repository — Student Performance Data Set

Two datasets are included:
- `student-mat.csv` — Math course (395 students) ← default
- `student-por.csv` — Portuguese course (649 students)

**Target variable:** `G3` — Final grade (0–20 scale)

**Pass/Fail label:** G3 ≥ 10 = Pass, G3 < 10 = Fail

---

## Results (Math Dataset)

| Model | Metric | Score |
|---|---|---|
| Linear Regression | RMSE | 2.38 |
| Linear Regression | R² | 0.724 |
| Logistic Regression | F1 | 0.922 |
| KNN (k=5) | F1 | 0.825 |
| Decision Tree | F1 | 0.922 |

**Top predictors:** G2 (second period grade) was by far the strongest predictor across all models, followed by G1, age, absences, and study time.

---

## Tech Stack

- Python
- pandas
- numpy
- scikit-learn
- matplotlib

---

## Course

CMPSC 446 — Data Mining
