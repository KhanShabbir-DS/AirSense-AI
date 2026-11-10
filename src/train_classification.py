"""
=========================================================
AirSense AI
Classification Model Training
Section A : Data Preparation & Pipeline Setup
=========================================================
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ML Utilities
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    LabelEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split
)

# Project Modules
from data_loader import DataLoader
from preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import time
from config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_CLASSIFICATION,
    RANDOM_STATE,
    TEST_SIZE,
    MODEL_DIR,
    OUTPUT_DIR
)


class ClassificationTrainer:

    def __init__(self):

        print("=" * 70)
        print("AirSense AI - Classification Training")
        print("=" * 70)

        self.models = {}
        self.results = {}
        self.trained_models = {}

        self.load_dataset()
        self.prepare_features()
        self.create_pipeline()
        self.create_models()

    # =====================================================

    def load_dataset(self):

        loader = DataLoader()

        raw_df = loader.load()

        processor = DataPreprocessor(raw_df)

        clean_df = processor.preprocess()

        engineer = FeatureEngineer(clean_df)

        self.df = engineer.build()

        print("\nDataset Ready for Classification")

        print(f"Rows    : {len(self.df)}")

        print(f"Columns : {len(self.df.columns)}")

    # =====================================================

    def prepare_features(self):

        print("\nPreparing Features...")

        self.X = self.df[
            NUMERICAL_FEATURES +
            CATEGORICAL_FEATURES
        ]

        self.label_encoder = LabelEncoder()

        self.y = self.label_encoder.fit_transform(
            self.df[TARGET_CLASSIFICATION]
        )

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test
        ) = train_test_split(

            self.X,

            self.y,

            test_size=TEST_SIZE,

            random_state=RANDOM_STATE,

            stratify=self.y

        )

        print(f"Training Samples : {len(self.X_train)}")

        print(f"Testing Samples  : {len(self.X_test)}")

        print(f"Classes          : {len(self.label_encoder.classes_)}")

        print("\nAQI Categories:")

        for label in self.label_encoder.classes_:

            print(f"✓ {label}")
        print("\nClass Distribution")

        distribution = self.df[TARGET_CLASSIFICATION].value_counts()

        print(distribution)

        print("\nPercentage Distribution")
        print(
            round(
                distribution / len(self.df) * 100,
                2
            )
        ) 

    # =====================================================

    def create_pipeline(self):

        print("\nCreating ML Pipeline...")

        numeric_pipeline = Pipeline(

            steps=[

                (
                    "imputer",
                    SimpleImputer(strategy="median")
                )

            ]

        )

        categorical_pipeline = Pipeline(

            steps=[

                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),

                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )

            ]

        )

        self.preprocessor = ColumnTransformer(

            transformers=[

                (
                    "num",
                    numeric_pipeline,
                    NUMERICAL_FEATURES
                ),

                (
                    "cat",
                    categorical_pipeline,
                    CATEGORICAL_FEATURES
                )

            ]

        )

        print("Pipeline Ready")

    # =====================================================

    def create_models(self):

        print("\nCreating Classification Models...")

        self.models = {

            "Logistic Regression": LogisticRegression(

                max_iter=1000,

                random_state=RANDOM_STATE

            ),

            "Random Forest": RandomForestClassifier(

                n_estimators=300,

                random_state=RANDOM_STATE,

                n_jobs=-1

            ),

            "XGBoost": XGBClassifier(

                n_estimators=300,

                learning_rate=0.05,

                max_depth=8,

                random_state=RANDOM_STATE,

                objective="multi:softprob",

                eval_metric="mlogloss"

            )

        }

        print()

        for model in self.models:

            print(f"✓ {model}")

    # Fixed Indentation: Moved method inside class scope
    def train_models(self):

        print("\n" + "=" * 70)
        print("Training Classification Models")
        print("=" * 70)

        for model_name, model in self.models.items():

            print(f"\nTraining {model_name}...")

            pipeline = Pipeline([

                ("preprocessor", self.preprocessor),

                ("model", model)

            ])

            start = time.time()

            pipeline.fit(

                self.X_train,

                self.y_train

            )

            elapsed = time.time() - start

            predictions = pipeline.predict(

                self.X_test

            )

            accuracy = accuracy_score(

                self.y_test,

                predictions

            )

            precision = precision_score(

                self.y_test,

                predictions,

                average="weighted"

            )

            recall = recall_score(

                self.y_test,

                predictions,

                average="weighted"

            )

            f1 = f1_score(

                self.y_test,

                predictions,

                average="weighted"

            )

            self.results[model_name] = {

                "Accuracy": round(accuracy, 4),

                "Precision": round(precision, 4),

                "Recall": round(recall, 4),

                "F1": round(f1, 4),

                "Time": round(elapsed, 2)

            }

            self.trained_models[model_name] = pipeline

            print(f"Accuracy  : {accuracy:.4f}")

            print(f"Precision : {precision:.4f}")

            print(f"Recall    : {recall:.4f}")

            print(f"F1 Score  : {f1:.4f}")

            print(f"Time (s)  : {elapsed:.2f}")

    # Fixed Indentation: Moved method inside class scope
    def compare_models(self):

        print("\n" + "=" * 70)

        print("Classification Model Comparison")

        print("=" * 70)

        comparison = pd.DataFrame(

            self.results

        ).T

        comparison = comparison.sort_values(

            by="Accuracy",

            ascending=False

        )

        self.comparison = comparison

        print(comparison)

        self.best_model_name = comparison.index[0]

        self.best_model = self.trained_models[

            self.best_model_name

        ]

        print("\nBest Model")

        print(self.best_model_name)

    # =====================================================

    def save_models(self):

        print("\n" + "=" * 70)
        print("Saving Classification Models")
        print("=" * 70)

        for model_name, pipeline in self.trained_models.items():

            filename = model_name.lower().replace(" ", "_") + "_classifier.pkl"

            joblib.dump(
                pipeline,
                MODEL_DIR / filename
            )

            print(f"✓ {filename}")

        joblib.dump(
            self.best_model,
            MODEL_DIR / "best_classifier.pkl"
        )

        joblib.dump(
            self.label_encoder,
            MODEL_DIR / "label_encoder.pkl"
        )

        print("\n✓ Best Classifier Saved")
        print("✓ Label Encoder Saved")

    # =====================================================

    def generate_reports(self):

        print("\nGenerating Reports...")

        predictions = self.best_model.predict(
            self.X_test
        )

        # -------------------------
        # Classification Report
        # -------------------------

        report = classification_report(

            self.y_test,

            predictions,

            target_names=self.label_encoder.classes_
        )

        with open(

            OUTPUT_DIR / "classification_report.txt",

            "w"

        ) as f:

            f.write(report)

        # -------------------------
        # Results CSV
        # -------------------------

        self.comparison.to_csv(

            OUTPUT_DIR / "classification_results.csv"

        )

        # -------------------------
        # Prediction CSV
        # -------------------------

        prediction_df = pd.DataFrame({

            "Actual":

            self.label_encoder.inverse_transform(
                self.y_test
            ),

            "Predicted":

            self.label_encoder.inverse_transform(
                predictions
            )

        })

        prediction_df.to_csv(

            OUTPUT_DIR / "classification_predictions.csv",

            index=False

        )

        print("✓ Reports Generated")

    # =====================================================

    def generate_confusion_matrix(self):

        print("\nGenerating Confusion Matrix...")

        predictions = self.best_model.predict(
            self.X_test
        )

        cm = confusion_matrix(

            self.y_test,

            predictions

        )

        cm_df = pd.DataFrame(

            cm,

            index=self.label_encoder.classes_,

            columns=self.label_encoder.classes_
        )

        cm_df.to_csv(

            OUTPUT_DIR / "confusion_matrix.csv"

        )

        fig, ax = plt.subplots(
            figsize=(8, 8)
        )

        disp = ConfusionMatrixDisplay(

            confusion_matrix=cm,

            display_labels=self.label_encoder.classes_
        )

        disp.plot(

            cmap="Blues",

            ax=ax,

            xticks_rotation=45,

            colorbar=False

        )

        plt.title("Confusion Matrix")

        plt.tight_layout()

        plt.savefig(

            OUTPUT_DIR / "confusion_matrix.png",

            dpi=300

        )

        plt.close()

        print("✓ Confusion Matrix Saved")


if __name__ == "__main__":

    trainer = ClassificationTrainer()
    trainer.train_models()
    trainer.compare_models()
    trainer.save_models()
    trainer.generate_reports()
    trainer.generate_confusion_matrix()

    print("\nClassification Engine Completed Successfully")
