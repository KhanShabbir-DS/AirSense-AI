"""
=========================================================
AirSense AI
Regression Analysis
=========================================================
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

from data_loader import DataLoader
from preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer

from config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_REGRESSION,
    MODEL_DIR,
    OUTPUT_DIR
)


class RegressionAnalysis:

    def __init__(self):

        print("="*70)
        print("AirSense AI - Regression Analysis")
        print("="*70)

        loader = DataLoader()
        raw = loader.load()

        clean = DataPreprocessor(raw).preprocess()

        self.df = FeatureEngineer(clean).build()

        self.model = joblib.load(
            MODEL_DIR / "best_model.pkl"
        )

    # --------------------------------------------------

    def prepare_data(self):

        self.X = self.df[
            NUMERICAL_FEATURES +
            CATEGORICAL_FEATURES
        ]

        self.y = self.df[TARGET_REGRESSION]

        self.predictions = self.model.predict(self.X)

    # --------------------------------------------------

    def actual_vs_predicted(self):

        print("Generating Actual vs Predicted Plot...")

        plt.figure(figsize=(8,8))

        plt.scatter(
            self.y,
            self.predictions,
            alpha=0.35
        )

        mn = min(self.y.min(), self.predictions.min())
        mx = max(self.y.max(), self.predictions.max())

        plt.plot(
            [mn, mx],
            [mn, mx],
            "r--",
            linewidth=2
        )

        plt.xlabel("Actual AQI")
        plt.ylabel("Predicted AQI")
        plt.title("Actual vs Predicted AQI")

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "actual_vs_predicted.png",
            dpi=300
        )

        plt.close()

    # --------------------------------------------------

    def residual_plot(self):

        print("Generating Residual Plot...")

        residuals = self.y - self.predictions

        plt.figure(figsize=(8,6))

        plt.scatter(
            self.predictions,
            residuals,
            alpha=0.35
        )

        plt.axhline(
            0,
            linestyle="--"
        )

        plt.xlabel("Predicted AQI")
        plt.ylabel("Residual")

        plt.title("Residual Analysis")

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "residual_plot.png",
            dpi=300
        )

        plt.close()

    # --------------------------------------------------

    def error_distribution(self):

        print("Generating Error Distribution...")

        errors = self.y - self.predictions

        plt.figure(figsize=(8,6))

        plt.hist(
            errors,
            bins=40
        )

        plt.xlabel("Prediction Error")

        plt.ylabel("Frequency")

        plt.title("Prediction Error Distribution")

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "error_distribution.png",
            dpi=300
        )

        plt.close()

    # --------------------------------------------------

    def prediction_sample(self):

        sample = pd.DataFrame({

            "Actual AQI": self.y,

            "Predicted AQI": self.predictions,

            "Absolute Error": abs(
                self.y -
                self.predictions
            )

        })

        sample.to_csv(

            OUTPUT_DIR / "prediction_analysis.csv",

            index=False

        )

    # --------------------------------------------------

    def summary(self):

        print("\n" + "="*70)

        print("Regression Summary")

        print("="*70)

        score = r2_score(
            self.y,
            self.predictions
        )

        print(f"Overall R² : {score:.4f}")

        print("\nAnalysis Files Generated")

        print("✓ actual_vs_predicted.png")

        print("✓ residual_plot.png")

        print("✓ error_distribution.png")

        print("✓ prediction_analysis.csv")


if __name__ == "__main__":

    analysis = RegressionAnalysis()

    analysis.prepare_data()

    analysis.actual_vs_predicted()

    analysis.residual_plot()

    analysis.error_distribution()

    analysis.prediction_sample()

    analysis.summary()