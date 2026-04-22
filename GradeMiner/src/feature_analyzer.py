# -----------------------------------------------------------------------------------------------------------------------------
# GradeMiner
# CMPSC 446
# data_preprocessor.py | Identifies and visualizes the most important features for academic performance.
"""
Uses three sources of feature importance:
  1. Linear Regression coefficients  (for regression)
  2. Logistic Regression coefficients (for classification)
  3. Decision Tree feature importances
"""
# -----------------------------------------------------------------------------------------------------------------------------


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class FeatureAnalyzer:
    """
    Extracts and plots the top N most important features from trained models.
    """

    def __init__(self, feature_names: list):
        self.feature_names = feature_names

    # ------------------------------------------------------------------
    # Importance extraction
    # ------------------------------------------------------------------

    def from_linear_model(self, model, top_n: int = 15) -> pd.DataFrame:
        """
        Extracts feature importances from a Linear or Logistic Regression
        model using the absolute value of the coefficients.
        """
        coefs = model.coef_.flatten() if hasattr(model.coef_, "flatten") else model.coef_
        df = pd.DataFrame({
            "Feature": self.feature_names,
            "Importance": np.abs(coefs)
        }).sort_values("Importance", ascending=False).head(top_n)
        return df

    def from_tree_model(self, model, top_n: int = 15) -> pd.DataFrame:
        """
        Extracts feature importances from a Decision Tree (Gini-based).
        """
        df = pd.DataFrame({
            "Feature": self.feature_names,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False).head(top_n)
        return df

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_top_features(self, importance_df: pd.DataFrame,
                          title: str, output_dir: str,
                          filename: str = "feature_importance.png") -> None:
        """
        Saves a horizontal bar chart of the top features.
        """
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)

        fig, ax = plt.subplots(figsize=(9, max(4, len(importance_df) * 0.45)))

        colors = plt.cm.RdYlGn(
            np.linspace(0.85, 0.25, len(importance_df))
        )
        ax.barh(importance_df["Feature"][::-1],
                importance_df["Importance"][::-1],
                color=colors, edgecolor="white")

        ax.set_xlabel("Importance Score", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.tick_params(axis="y", labelsize=9)
        fig.tight_layout()
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        print(f"[FeatureAnalyzer] Plot saved → {filepath}")

    def run_full_analysis(self, regressor, classifier_lr, classifier_dt,
                          output_dir: str) -> dict:
        """
        Runs feature importance analysis for all three models and saves plots.

        Parameters:
            regressor     : trained LinearRegression model
            classifier_lr : trained LogisticRegression model
            classifier_dt : trained DecisionTreeClassifier model
            output_dir    : where to save the plots

        Returns a dict of DataFrames: {model_name: importance_df}
        """
        results = {}

        # 1. Linear Regression coefficients
        lr_importance = self.from_linear_model(regressor, top_n=15)
        results["Linear Regression (G3 prediction)"] = lr_importance
        self.plot_top_features(
            lr_importance,
            title="Top Features — Linear Regression (G3 Prediction)",
            output_dir=output_dir,
            filename="feature_importance_regression.png"
        )

        # 2. Logistic Regression coefficients
        log_importance = self.from_linear_model(classifier_lr, top_n=15)
        results["Logistic Regression (Pass/Fail)"] = log_importance
        self.plot_top_features(
            log_importance,
            title="Top Features — Logistic Regression (Pass/Fail)",
            output_dir=output_dir,
            filename="feature_importance_logistic.png"
        )

        # 3. Decision Tree importances
        dt_importance = self.from_tree_model(classifier_dt, top_n=15)
        results["Decision Tree (Pass/Fail)"] = dt_importance
        self.plot_top_features(
            dt_importance,
            title="Top Features — Decision Tree (Pass/Fail)",
            output_dir=output_dir,
            filename="feature_importance_tree.png"
        )

        # Print top 10 from each
        for model_name, df in results.items():
            print(f"\n  Top features for {model_name}:")
            for _, row in df.head(10).iterrows():
                print(f"    {row['Feature']:<35} {row['Importance']:.4f}")

        return results
