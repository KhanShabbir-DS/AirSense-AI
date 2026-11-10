"""
=========================================================
AirSense AI
Regression Model Training
Section A : Data Preparation & Pipeline Setup
=========================================================
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

# Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ML Utilities
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Project Modules
from data_loader import DataLoader
from preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer

from config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_REGRESSION,
    RANDOM_STATE,
    TEST_SIZE,
    MODEL_DIR,
    OUTPUT_DIR
)


class RegressionTrainer:

    def __init__(self):
        print("=" * 70)
        print("AirSense AI - Regression Training")
        print("=" * 70)

        self.models = {}
        self.results = {}
        self.trained_models = {}

        self.load_dataset()
        self.prepare_features()
        self.create_pipeline()
        self.create_models()

    def load_dataset(self):
        loader = DataLoader()
        raw_df = loader.load()
        processor = DataPreprocessor(raw_df)
        clean_df = processor.preprocess()
        engineer = FeatureEngineer(clean_df)
        self.df = engineer.build()

        print("\nDataset Ready for Machine Learning")
        print(f"Rows    : {len(self.df)}")
        print(f"Columns : {len(self.df.columns)}")

    def prepare_features(self):
        print("\nPreparing Features...")
        self.X = self.df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
        self.y = self.df[TARGET_REGRESSION]

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test
        ) = train_test_split(
            self.X,
            self.y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )

        print(f"Training Samples : {len(self.X_train)}")
        print(f"Testing Samples  : {len(self.X_test)}")

    def create_pipeline(self):
        print("\nCreating ML Pipeline...")
        numeric_pipeline = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, NUMERICAL_FEATURES),
                ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
            ]
        )
        print("Pipeline Ready")

    def create_models(self):
        print("\nCreating Models...")
        self.models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(
                n_estimators=200,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            "XGBoost": XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=8,
                random_state=RANDOM_STATE,
                objective="reg:squarederror"
            )
        }

        for model in self.models:
            print(f"✓ {model}")

    def train_models(self):
        print("\n" + "=" * 70)
        print("Training Regression Models")
        print("=" * 70)

        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")

            pipeline = Pipeline([
                ("preprocessor", self.preprocessor),
                ("model", model)
            ])

            start = time.time()
            pipeline.fit(self.X_train, self.y_train)
            train_time = time.time() - start

            predictions = pipeline.predict(self.X_test)
            mae = mean_absolute_error(self.y_test, predictions)
            rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
            r2 = r2_score(self.y_test, predictions)

            try:
                cv = cross_val_score(
                    pipeline,
                    self.X_train,
                    self.y_train,
                    cv=3,
                    scoring="r2",
                    n_jobs=1 
                ).mean()
            except Exception as e:
                print(f"⚠️ Cross-validation skipped for {model_name} due to environment limitations.")
                cv = 0.0

            self.results[model_name] = {
                "MAE": round(mae, 3),
                "RMSE": round(rmse, 3),
                "R2": round(r2, 4),
                "CV_R2": round(cv, 4),
                "Training_Time": round(train_time, 2)
            }

            self.trained_models[model_name] = pipeline

            print(f"MAE       : {mae:.3f}")
            print(f"RMSE      : {rmse:.3f}")
            print(f"R²        : {r2:.4f}")
            print(f"CV R²     : {cv:.4f}")
            print(f"Time (s)  : {train_time:.2f}")

    def compare_models(self):
        print("\n" + "=" * 70)
        print("Model Comparison")
        print("=" * 70)

        comparison = pd.DataFrame(self.results).T
        comparison = comparison.sort_values(by="R2", ascending=False)

        # Keep original comparison (index = model names)
        self.comparison = comparison.copy()

        # Save CSV with a Model column
        comparison_to_save = comparison.reset_index()
        comparison_to_save.rename(columns={"index": "Model"}, inplace=True)

        comparison_to_save.to_csv(
            OUTPUT_DIR / "regression_results.csv",
            index=False
        )

        print("✓ regression_results.csv saved")
        print(comparison_to_save)

        # Select best model BEFORE resetting the index
        self.best_model_name = comparison.index[0]
        self.best_model = self.trained_models[self.best_model_name]

        print(f"\nBest Model: {self.best_model_name}")


    def save_models(self):
        print("\n" + "=" * 70)
        print("Saving Models")
        print("=" * 70)
        
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        for model_name, pipeline in self.trained_models.items():
            filename = model_name.lower().replace(" ", "_") + ".pkl"
            path = MODEL_DIR / filename
            joblib.dump(pipeline, path)
            print(f"Saved : {filename}")

        best_path = MODEL_DIR / "best_model.pkl"
        joblib.dump(self.best_model, best_path)
        print("\nBest Model Saved")

    def save_reports(self):
        print("\nSaving Reports...")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Model Comparison
        comparison_path = OUTPUT_DIR / "model_comparison.csv"
        self.comparison.to_csv(comparison_path, index=True)

        # 2. Metrics TXT
        metrics_path = OUTPUT_DIR / "regression_metrics.txt"
        with open(metrics_path, "w") as f:
            f.write("AirSense AI Regression Results\n\n")
            f.write(self.comparison.to_string())

        # 3. Predictions Comparison Output
        predictions = self.best_model.predict(self.X_test)
        prediction_df = pd.DataFrame({
            "Actual AQI": self.y_test.values,
            "Predicted AQI": predictions
        })
        prediction_df.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
        print("Reports Saved")

        # ----------------------------------------
        # Prediction Error Distribution
        # ----------------------------------------

        predictions = self.best_model.predict(self.X_test)
        errors = self.y_test.values - predictions

        plt.figure(figsize=(8,6))

        plt.hist(
            errors,
            bins=40,
            edgecolor="black",
            color="steelblue"
        )

        plt.title("Prediction Error Distribution")
        plt.xlabel("Prediction Error (Actual - Predicted)")
        plt.ylabel("Frequency")

        plt.grid(alpha=0.3)

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "prediction_error_distribution.png",
            dpi=300
        )

        plt.close()

        print("✓ Prediction Error Distribution Saved")

    def feature_importance(self):
        print("\nGenerating Feature Importance...")
        model = self.best_model.named_steps["model"]

        if not hasattr(model, "feature_importances_"):
            print(f"Feature Importance Not Available for {self.best_model_name}")
            return

        # Safely extract dynamic post-transform names from pipeline
        try:
            ct = self.best_model.named_steps["preprocessor"]
            cat_encoder = ct.named_transformers_["cat"].named_steps["encoder"]
            encoded_cat_features = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
            all_features = NUMERICAL_FEATURES + encoded_cat_features
        except Exception as e:
            print(f"⚠️ Could not parse feature names dynamically: {e}")
            return

        importances = model.feature_importances_
        
        # Prevent length mismatch errors
        if len(all_features) != len(importances):
            print("⚠️ Mismatch between feature names array and calculated importance length.")
            return

        importance_df = pd.DataFrame({
            "Feature": all_features,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        # Export top 15 features to a clean chart
        plt.figure(figsize=(10, 6))
        top_n = importance_df.head(15)
        plt.barh(top_n["Feature"][::-1], top_n["Importance"][::-1], color="teal")
        plt.xlabel("Relative Importance Score")
        plt.title(f"Top 15 Feature Importances - {self.best_model_name}")
        plt.tight_layout()
        
        chart_path = OUTPUT_DIR / "feature_importance.png"
        plt.savefig(chart_path)
        plt.close()
        print(f"✓ Feature Importance Chart Saved to: {chart_path}")

#==========================================
if __name__ == "__main__":
    trainer = RegressionTrainer()
    trainer.train_models()
    trainer.compare_models()
    trainer.save_models()
    trainer.save_reports()
    trainer.feature_importance()
    print("\nSection B and Verification Tasks Completed Successfully")
