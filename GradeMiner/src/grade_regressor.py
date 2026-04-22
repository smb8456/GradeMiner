# -----------------------------------------------------------------------------------------------------------------------------
# GradeMiner
# CMPSC 446
# grade_regressor.py | Trains a Linear Regression model to predict the final grade (G3).
# -----------------------------------------------------------------------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from model_evaluator import ModelEvaluator


class GradeRegressor:
    """
    Trains a Linear Regression model on student data to predict G3.
    Evaluates with RMSE and R², and optionally saves an actual vs predicted plot.
    """

    def __init__(self):
        self.model = LinearRegression()
        self.metrics: dict = {}
        self.y_pred = None

    def train(self, X_train, y_train) -> None:
        """Fit the linear regression model on training data."""
        self.model.fit(X_train, y_train)
        print("[GradeRegressor] Model trained.")

    def predict(self, X_test) -> np.ndarray:
        """Generate predictions for the test set."""
        self.y_pred = self.model.predict(X_test)
        return self.y_pred

    def evaluate(self, y_test) -> dict:
        """Compute RMSE and R² against the true test labels."""
        if self.y_pred is None:
            raise ValueError("Call predict() before evaluate().")
        self.metrics = ModelEvaluator.regression_metrics(y_test, self.y_pred)
        ModelEvaluator.print_regression_results(self.metrics)
        return self.metrics

    def save_actual_vs_predicted_plot(self, y_test, output_dir: str) -> None:
        """
        Saves a scatter plot of actual vs predicted G3 grades.
        A perfect model would show all points along the diagonal line.
        """
        if self.y_pred is None:
            raise ValueError("Call predict() before saving the plot.")

        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, "regression_actual_vs_predicted.png")

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(y_test, self.y_pred, alpha=0.6, color="steelblue", edgecolors="white", s=60)

        # Perfect prediction line
        min_val = min(y_test.min(), self.y_pred.min())
        max_val = max(y_test.max(), self.y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Perfect fit")

        ax.set_xlabel("Actual G3", fontsize=12)
        ax.set_ylabel("Predicted G3", fontsize=12)
        ax.set_title(
            f"Linear Regression — Actual vs Predicted G3\n"
            f"RMSE: {self.metrics.get('RMSE', '?')}  |  R²: {self.metrics.get('R2', '?')}",
            fontsize=11
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        print(f"[GradeRegressor] Plot saved → {filepath}")
