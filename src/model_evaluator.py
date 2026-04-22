# -----------------------------------------------------------------------------------------------------------------------------
# GradeMiner
# CMPSC 446
# model_evaluator.py | Centralized metric calculation and result formatting for GradeMiner.
# -----------------------------------------------------------------------------------------------------------------------------

import numpy as np
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)


class ModelEvaluator:
    """
    Calculates and formats evaluation metrics for both regression
    and classification models.
    """

    @staticmethod
    def regression_metrics(y_true, y_pred) -> dict:
        """
        Returns RMSE and R² for a regression model.
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return {"RMSE": round(rmse, 4), "R2": round(r2, 4)}

    @staticmethod
    def classification_metrics(y_true, y_pred, model_name: str = "") -> dict:
        """
        Returns Accuracy, Precision, Recall, and F1 for a classifier.
        Uses 'binary' averaging since this is a two-class problem.
        """
        return {
            "Model": model_name,
            "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
            "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
            "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        }

    @staticmethod
    def print_regression_results(metrics: dict, model_name: str = "Linear Regression") -> None:
        print(f"\n{'='*45}")
        print(f"  Regression Results — {model_name}")
        print(f"{'='*45}")
        print(f"  RMSE : {metrics['RMSE']}")
        print(f"  R²   : {metrics['R2']}")
        print(f"{'='*45}")

    @staticmethod
    def print_classification_comparison(results: list[dict]) -> None:
        """Pretty-print a comparison table for multiple classifiers."""
        print(f"\n{'='*65}")
        print(f"  Classification Model Comparison")
        print(f"{'='*65}")
        header = f"  {'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}"
        print(header)
        print(f"  {'-'*60}")
        for r in results:
            print(
                f"  {r['Model']:<22} {r['Accuracy']:>9.4f} "
                f"{r['Precision']:>10.4f} {r['Recall']:>8.4f} {r['F1']:>8.4f}"
            )
        print(f"{'='*65}")

    @staticmethod
    def best_classifier(results: list[dict]) -> dict:
        """Return the result dict with the highest F1 score."""
        return max(results, key=lambda r: r["F1"])
