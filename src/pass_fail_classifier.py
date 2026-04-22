# -----------------------------------------------------------------------------------------------------------------------------
# GradeMiner
# CMPSC 446
# pass_fail_classifier.py | Trains and compares three classifiers for Pass/Fail prediction.
# -----------------------------------------------------------------------------------------------------------------------------

import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from model_evaluator import ModelEvaluator


# Model definitions — easy to add more later
CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "KNN (k=5)":           KNeighborsClassifier(n_neighbors=5),
    "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
}


class PassFailClassifier:
    """
    Trains Logistic Regression, KNN, and Decision Tree classifiers.
    Evaluates each with Accuracy, Precision, Recall, and F1.
    Identifies the best-performing model by F1 score.
    """

    def __init__(self):
        self.trained_models: dict = {}   # name -> fitted model
        self.results: list[dict] = []    # list of metric dicts
        self.best_model_name: str = ""

    def train_all(self, X_train, y_train) -> None:
        """Fit every classifier on the training set."""
        for name, clf in CLASSIFIERS.items():
            clf.fit(X_train, y_train)
            self.trained_models[name] = clf
        print(f"[PassFailClassifier] Trained {len(self.trained_models)} models.")

    def evaluate_all(self, X_test, y_test) -> list[dict]:
        """
        Generate predictions and compute metrics for each model.
        Returns a list of metric dicts, one per classifier.
        """
        self.results = []
        for name, clf in self.trained_models.items():
            y_pred = clf.predict(X_test)
            metrics = ModelEvaluator.classification_metrics(y_test, y_pred, model_name=name)
            self.results.append(metrics)

        # Print comparison table
        ModelEvaluator.print_classification_comparison(self.results)

        # Identify best model by F1
        best = ModelEvaluator.best_classifier(self.results)
        self.best_model_name = best["Model"]
        print(f"\n  ★  Best Classifier: {self.best_model_name}  (F1 = {best['F1']})")

        return self.results

    def get_best_model(self):
        """Return the fitted model object with the highest F1 score."""
        return self.trained_models.get(self.best_model_name)

    def save_comparison_bar_chart(self, output_dir: str) -> None:
        """
        Saves a grouped bar chart comparing Accuracy, Precision, Recall, F1
        across all trained classifiers.
        """
        if not self.results:
            raise ValueError("Run evaluate_all() before saving the chart.")

        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, "classifier_comparison.png")

        metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1"]
        model_names = [r["Model"] for r in self.results]
        n_models = len(model_names)
        n_metrics = len(metrics_to_plot)

        import numpy as np
        x = np.arange(n_metrics)
        bar_width = 0.22
        colors = ["steelblue", "seagreen", "tomato"]

        fig, ax = plt.subplots(figsize=(9, 5))
        for i, (name, color) in enumerate(zip(model_names, colors)):
            values = [self.results[i][m] for m in metrics_to_plot]
            bars = ax.bar(x + i * bar_width, values, width=bar_width,
                          label=name, color=color, alpha=0.85)
            # Add value labels on bars
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{bar.get_height():.2f}",
                        ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x + bar_width * (n_models - 1) / 2)
        ax.set_xticklabels(metrics_to_plot, fontsize=11)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title("Pass/Fail Classifier Comparison", fontsize=13)
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        print(f"[PassFailClassifier] Chart saved → {filepath}")
