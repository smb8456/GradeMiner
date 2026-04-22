# -----------------------------------------------------------------------------------------------------------------------------
# GradeMiner
# CMPSC 446
# data_preprocessor.py | Loads, cleans, encodes, and splits the UCI Student Performance dataset.
# -----------------------------------------------------------------------------------------------------------------------------


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Default pass threshold — students with G3 >= this value are labeled Pass
PASS_THRESHOLD = 10

# Columns that are binary yes/no — mapped to 1/0
BINARY_COLS = [
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic"
]

# Columns with more than two categories — one-hot encoded
CATEGORICAL_COLS = ["school", "sex", "address", "famsize", "Pstatus",
                    "Mjob", "Fjob", "reason", "guardian"]


class DataPreprocessor:
    """
    Handles all preprocessing steps:
      - Loading raw CSV
      - Binary and one-hot encoding
      - Feature/target separation
      - Normalization (StandardScaler on numeric features)
      - Pass/Fail label creation
      - Train/test splitting
    """

    def __init__(self, dataset: str = "student-mat.csv",
                 data_dir: str = None,
                 pass_threshold: int = PASS_THRESHOLD,
                 test_size: float = 0.2,
                 random_state: int = 42):

        if data_dir is None:
            # Default: data/raw/ relative to this file's location
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base, "data", "raw")

        self.filepath = os.path.join(data_dir, dataset)
        self.pass_threshold = pass_threshold
        self.test_size = test_size
        self.random_state = random_state

        self.df_raw: pd.DataFrame = None        # Raw loaded data
        self.df_encoded: pd.DataFrame = None    # After encoding
        self.feature_names: list = []           # Feature column names
        self.scaler = StandardScaler()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """Read the CSV file (semicolon-separated) and store as df_raw."""
        self.df_raw = pd.read_csv(self.filepath, sep=";")
        print(f"[DataPreprocessor] Loaded {len(self.df_raw)} records from '{self.filepath}'")
        return self.df_raw

    def preprocess(self) -> pd.DataFrame:
        """
        Full preprocessing pipeline.
        Returns the encoded DataFrame (df_encoded).
        """
        if self.df_raw is None:
            self.load()

        df = self.df_raw.copy()

        # 1. Encode binary yes/no columns
        for col in BINARY_COLS:
            if col in df.columns:
                df[col] = df[col].map({"yes": 1, "no": 0})

        # 2. One-hot encode categorical columns
        df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

        # 3. Convert all bool columns (from get_dummies) to int
        bool_cols = df.select_dtypes(include="bool").columns
        df[bool_cols] = df[bool_cols].astype(int)

        self.df_encoded = df
        print(f"[DataPreprocessor] After encoding: {df.shape[1]} columns")
        return df

    def get_regression_data(self):
        """
        Returns (X_train, X_test, y_train, y_test) for G3 regression.
        Features are scaled; G1 and G2 are included since they are strong predictors.
        """
        if self.df_encoded is None:
            self.preprocess()

        df = self.df_encoded.copy()
        X = df.drop(columns=["G3"])
        y = df["G3"]

        self.feature_names = list(X.columns)
        return self._split_and_scale(X, y)

    def get_classification_data(self, include_grades: bool = True):
        """
        Returns (X_train, X_test, y_train, y_test) for Pass/Fail classification.
        y is 1 (Pass) or 0 (Fail) based on G3 >= pass_threshold.

        include_grades: if False, drops G1 and G2 so the model predicts
                        purely from demographic/social features.
        """
        if self.df_encoded is None:
            self.preprocess()

        df = self.df_encoded.copy()

        # Create binary target BEFORE dropping G3
        y = (df["G3"] >= self.pass_threshold).astype(int)

        # Drop G3 (target); optionally drop G1/G2
        drop_cols = ["G3"]
        if not include_grades:
            drop_cols += ["G1", "G2"]

        X = df.drop(columns=drop_cols)
        self.feature_names = list(X.columns)
        return self._split_and_scale(X, y)

    def get_feature_names(self) -> list:
        """Return feature names from the last get_*_data() call."""
        return self.feature_names

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_and_scale(self, X: pd.DataFrame, y: pd.Series):
        """Split into train/test, then StandardScale the features."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        # Fit scaler on training data only — avoids data leakage
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Wrap back into DataFrames so feature names are accessible
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

        return X_train_scaled, X_test_scaled, y_train.reset_index(drop=True), y_test.reset_index(drop=True)

    def get_pass_fail_counts(self) -> dict:
        """Quick look at class balance for the Pass/Fail target."""
        if self.df_raw is None:
            self.load()
        counts = (self.df_raw["G3"] >= self.pass_threshold).value_counts()
        return {"Pass": int(counts.get(True, 0)), "Fail": int(counts.get(False, 0))}
