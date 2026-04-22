# -----------------------------------------------------------------------------------------------------------------------------
# GradeMiner
# CMPSC 446
# main.py | GradeMiner entry point.
""" 
  Runs:
  1. Data loading & preprocessing
  2. Grade regression (predict G3)
  3. Pass/Fail classification (compare 3 models)
  4. Feature importance analysis
  5. Saves results to models/results/
"""
# -----------------------------------------------------------------------------------------------------------------------------

import os
import sys
import csv

# Make sure src/ imports work when run from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_preprocessor import DataPreprocessor, PASS_THRESHOLD
from grade_regressor import GradeRegressor
from pass_fail_classifier import PassFailClassifier
from feature_analyzer import FeatureAnalyzer

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "models", "results")

# Dataset to use — change to "student-por.csv" to switch datasets
DATASET = "student-mat.csv"

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def save_results_csv(reg_metrics: dict, clf_results: list[dict],
                     best_clf: str, output_dir: str) -> None:
    """Saves a simple CSV summary of all model metrics."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "model_results_summary.csv")

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)

        # Regression
        writer.writerow(["=== REGRESSION ==="])
        writer.writerow(["Model", "RMSE", "R2"])
        writer.writerow(["Linear Regression", reg_metrics["RMSE"], reg_metrics["R2"]])
        writer.writerow([])

        # Classification
        writer.writerow(["=== CLASSIFICATION ==="])
        writer.writerow(["Model", "Accuracy", "Precision", "Recall", "F1"])
        for r in clf_results:
            writer.writerow([r["Model"], r["Accuracy"], r["Precision"], r["Recall"], r["F1"]])
        writer.writerow([])
        writer.writerow(["Best Classifier", best_clf])

    print(f"[main] Results saved → {filepath}")


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

def main():
    print("\n" + "=" * 55)
    print("  GradeMiner — Student Performance Prediction")
    print("=" * 55)
    print(f"  Dataset       : {DATASET}")
    print(f"  Pass threshold: G3 >= {PASS_THRESHOLD}")
    print()

    # ----------------------------------------------------------------
    # 1. Preprocessing
    # ----------------------------------------------------------------
    print(">>> Step 1: Loading & Preprocessing Data")
    preprocessor = DataPreprocessor(dataset=DATASET)
    preprocessor.load()

    counts = preprocessor.get_pass_fail_counts()
    print(f"  Class balance → Pass: {counts['Pass']}, Fail: {counts['Fail']}")

    # Regression split (uses G1, G2, all features; predicts G3)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = preprocessor.get_regression_data()
    feature_names_reg = preprocessor.get_feature_names()

    # Classification split (uses G1, G2; predicts Pass/Fail from G3 label)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = preprocessor.get_classification_data()
    feature_names_clf = preprocessor.get_feature_names()

    print(f"  Regression  → Train: {len(X_train_reg)}, Test: {len(X_test_reg)}")
    print(f"  Classification → Train: {len(X_train_clf)}, Test: {len(X_test_clf)}")

    # ----------------------------------------------------------------
    # 2. Regression — predict G3
    # ----------------------------------------------------------------
    print("\n>>> Step 2: Grade Regression (predicting G3)")
    regressor = GradeRegressor()
    regressor.train(X_train_reg, y_train_reg)
    regressor.predict(X_test_reg)
    reg_metrics = regressor.evaluate(y_test_reg)
    regressor.save_actual_vs_predicted_plot(y_test_reg, RESULTS_DIR)

    # ----------------------------------------------------------------
    # 3. Classification — Pass/Fail
    # ----------------------------------------------------------------
    print("\n>>> Step 3: Pass/Fail Classification")
    classifier = PassFailClassifier()
    classifier.train_all(X_train_clf, y_train_clf)
    clf_results = classifier.evaluate_all(X_test_clf, y_test_clf)
    classifier.save_comparison_bar_chart(RESULTS_DIR)

    # ----------------------------------------------------------------
    # 4. Feature Analysis
    # ----------------------------------------------------------------
    print("\n>>> Step 4: Feature Importance Analysis")

    # We need the raw model objects for coefficient extraction
    lr_model  = classifier.trained_models["Logistic Regression"]
    dt_model  = classifier.trained_models["Decision Tree"]

    # Regression uses its own feature set; for feature analysis we use
    # the classification feature names (same dataset, same columns minus G3)
    analyzer = FeatureAnalyzer(feature_names=feature_names_clf)
    analyzer.run_full_analysis(
        regressor=regressor.model,
        classifier_lr=lr_model,
        classifier_dt=dt_model,
        output_dir=RESULTS_DIR
    )

    # ----------------------------------------------------------------
    # 5. Save summary CSV
    # ----------------------------------------------------------------
    print("\n>>> Step 5: Saving Results Summary")
    save_results_csv(reg_metrics, clf_results, classifier.best_model_name, RESULTS_DIR)

    # ----------------------------------------------------------------
    # Done
    # ----------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  GradeMiner Complete!")
    print(f"  Results saved to: {RESULTS_DIR}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
