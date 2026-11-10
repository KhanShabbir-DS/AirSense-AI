"""
=========================================================
AirSense AI
Prediction Engine
=========================================================

Author : Khan Shabbir

This module loads the trained machine learning models and
provides a unified prediction interface for:

1. AQI Regression
2. AQI Category Classification
3. Health Recommendations
4. Dashboard Metadata

=========================================================
"""

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime

import joblib
import pandas as pd

from config import (
    MODEL_DIR,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)


class AirSensePredictor:

    """
    Main Prediction Engine
    """

    def __init__(self):

        print("=" * 70)
        print("AirSense AI Prediction Engine")
        print("=" * 70)

        self.load_models()
        self.load_city_data()

        print("\nPrediction Engine Ready")

    # =====================================================

    def load_models(self):

        print("\nLoading Trained Models...")

        self.regression_model = joblib.load(
            MODEL_DIR / "best_model.pkl"
        )

        self.classification_model = joblib.load(
            MODEL_DIR / "best_classifier.pkl"
        )

        self.label_encoder = joblib.load(
            MODEL_DIR / "label_encoder.pkl"
        )

        print("[OK] Regression Model Loaded")
        print("[OK] Classification Model Loaded")
        print("[OK] Label Encoder Loaded")
    # =====================================================

    def load_city_data(self):
        """
        Load and preprocess the city dataset once for reuse.
        """

        from data_loader import DataLoader
        from preprocessing import DataPreprocessor

        loader = DataLoader()
        raw_df = loader.load()
        self.city_data = DataPreprocessor(raw_df).preprocess()
        self.city_data["City"] = self.city_data["City"].astype(str).str.strip()

    # =====================================================

    def validate_input(self, input_data):

        """
        Validate the user input before prediction.
        """

        required_features = (
            NUMERICAL_FEATURES +
            CATEGORICAL_FEATURES
        )

        missing_features = []

        for feature in required_features:

            if feature not in input_data:

                missing_features.append(feature)

        if len(missing_features) > 0:

            raise ValueError(

                f"Missing Features: {missing_features}"

            )

        return True


    # =====================================================

    def prepare_input(self, input_data):

        """
        Convert dictionary input into
        a Pandas DataFrame in the correct order.
        """

        self.validate_input(input_data)

        df = pd.DataFrame([input_data])

        expected_columns = (
            NUMERICAL_FEATURES +
            CATEGORICAL_FEATURES
        )

        df = df[expected_columns]

        return df


    # =====================================================

    def check_city(self, city):

        """
        Check whether the entered city
        exists in the trained dataset.
        """

        allowed_cities = [

            "Ahmedabad",
            "Aizawl",
            "Amaravati",
            "Amritsar",
            "Bengaluru",
            "Bhopal",
            "Brajrajnagar",
            "Chandigarh",
            "Chennai",
            "Coimbatore",
            "Delhi",
            "Ernakulam",
            "Gurugram",
            "Guwahati",
            "Hyderabad",
            "Jaipur",
            "Kochi",
            "Kolkata",
            "Lucknow",
            "Mumbai",
            "Patna",
            "Shillong",
            "Talcher",
            "Thiruvananthapuram",
            "Visakhapatnam",
            "Pune"

        ]

        if city not in allowed_cities:

            raise ValueError(

                f"City '{city}' is not available."

            )

        return True


    # =====================================================

    def check_season(self, season):

        """
        Validate season.
        """

        allowed = [
            "Winter",
            "Summer",
            "Monsoon",
            "PostMonsoon",
        ]

        if season not in allowed:

            raise ValueError(

                f"Invalid Season: {season}"

            )

        return True


    # =====================================================

    def sanitize_input(self, input_data):

        """
        Perform final validation.
        """

        self.check_city(

            input_data["City"]

        )

        self.check_season(

            input_data["Season"]

        )

        return self.prepare_input(

            input_data

        )

    # =====================================================

    def get_health_information(self, category):
        """
        Return health metadata based on AQI category.
        """

        health_info = {

            "Good": {
                "Color": "#2ECC71",
                "Icon": "🟢",
                "Severity": "Low",
                "Health Advice": "Air quality is excellent. Enjoy outdoor activities.",
                "Mask": "Not Required",
                "Outdoor Activity": "Excellent",
                "Sensitive Groups": "No health risk."
            },

            "Satisfactory": {
                "Color": "#8BC34A",
                "Icon": "🟢",
                "Severity": "Low",
                "Health Advice": "Air quality is acceptable for most people.",
                "Mask": "Optional",
                "Outdoor Activity": "Good",
                "Sensitive Groups": "Minor discomfort for highly sensitive individuals."
            },

            "Moderate": {
                "Color": "#F1C40F",
                "Icon": "🟡",
                "Severity": "Moderate",
                "Health Advice": "Sensitive individuals should reduce prolonged outdoor activity.",
                "Mask": "Optional",
                "Outdoor Activity": "Moderate",
                "Sensitive Groups": "Children, elderly and asthma patients should be careful."
            },

            "Poor": {
                "Color": "#E67E22",
                "Icon": "🟠",
                "Severity": "High",
                "Health Advice": "Limit prolonged outdoor exposure.",
                "Mask": "Recommended",
                "Outdoor Activity": "Limited",
                "Sensitive Groups": "People with respiratory diseases should avoid long exposure."
            },

            "Very Poor": {
                "Color": "#E74C3C",
                "Icon": "🔴",
                "Severity": "Very High",
                "Health Advice": "Avoid unnecessary outdoor activities.",
                "Mask": "N95 Recommended",
                "Outdoor Activity": "Avoid",
                "Sensitive Groups": "Everyone may experience health effects."
            },

            "Severe": {
                "Color": "#8E44AD",
                "Icon": "🟣",
                "Severity": "Hazardous",
                "Health Advice": "Stay indoors as much as possible.",
                "Mask": "N95 Mandatory",
                "Outdoor Activity": "Not Recommended",
                "Sensitive Groups": "Serious health effects for the entire population."
            }

        }

        return health_info.get(category, {})


    # =====================================================

    def get_aqi_category(self, aqi):
        """
        Convert a predicted AQI value into an AQI category using the
        standard Indian AQI scale.
        """

        if aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Satisfactory"
        elif aqi <= 200:
            return "Moderate"
        elif aqi <= 300:
            return "Poor"
        elif aqi <= 400:
            return "Very Poor"
        else:
            return "Severe"

    # =====================================================

    def predict(self, input_data):
        """
        Predict AQI value and AQI category.
        """

        df = self.sanitize_input(input_data)

        # -----------------------------
        # Regression Prediction
        # -----------------------------

        predicted_aqi = float(
            self.regression_model.predict(df)[0]
        )

        # -----------------------------
        # Classification Prediction
        # -----------------------------

        predicted_class = self.classification_model.predict(df)

        ml_category = self.label_encoder.inverse_transform(
            predicted_class
        )[0]

        predicted_category = self.get_aqi_category(predicted_aqi)

        # -----------------------------
        # Confidence
        # -----------------------------

        confidence = None

        if hasattr(
            self.classification_model,
            "predict_proba"
        ):

            probabilities = self.classification_model.predict_proba(df)[0]

            confidence = float(probabilities.max())

        # -----------------------------
        # Health Information
        # -----------------------------

        health = self.get_health_information(
            predicted_category
        )

        # -----------------------------
        # Final Response
        # -----------------------------

        result = {

            "Prediction": {

                "AQI": round(
                    predicted_aqi,
                    2
                ),

                "Category": predicted_category,

                "Confidence": None if confidence is None else round(
                    confidence,
                    4
                )

            },

            "Status": {

                "Severity": health["Severity"],

                "Color": health["Color"],

                "Icon": health["Icon"]

            },

            "Health": {

                "Advice": health["Health Advice"],

                "Mask": health["Mask"],

                "Outdoor Activity": health["Outdoor Activity"],

                "Sensitive Groups": health["Sensitive Groups"]

            },

            "Technical": {

                "Regression Model": "Random Forest Regressor",

                "Classification Model": "Random Forest Classifier",

                "Prediction Time": datetime.now().strftime(
                    "%d-%m-%Y %H:%M:%S"
                )

            }

        }

        city_information = self.get_city_insights(
            input_data["City"]
        )

        result["City Insights"] = city_information

        return result

    # =====================================================

    def get_city_insights(self, city):

        """
        Generate historical insights for the selected city.
        """

        try:

            city = str(city).strip()
            clean_df = self.city_data.copy()
            clean_df["Date"] = pd.to_datetime(clean_df["Date"], errors="coerce")
            clean_df["City"] = clean_df["City"].astype(str).str.strip()

            city_df = clean_df[
                clean_df["City"].str.lower() == city.lower()
            ].dropna(subset=["Date"])

            if city_df.empty:

                return {
                    "City": city,
                    "Status": "No historical data available for this city"
                }

            average_aqi = round(city_df["AQI"].mean(), 2)

            maximum_aqi = round(city_df["AQI"].max(), 2)

            minimum_aqi = round(city_df["AQI"].min(), 2)

            latest_year = city_df["Date"].dt.year.max()

            oldest_year = city_df["Date"].dt.year.min()

            yearly = (

                city_df

                .groupby(

                    city_df["Date"].dt.year

                )["AQI"]

                .mean()

            )

            trend = "Stable"

            if len(yearly) >= 2:

                if yearly.iloc[-1] < yearly.iloc[0]:

                    trend = "Improving"

                elif yearly.iloc[-1] > yearly.iloc[0]:

                    trend = "Worsening"

            return {

                "City": city,

                "Historical Average AQI": average_aqi,

                "Highest AQI": maximum_aqi,

                "Lowest AQI": minimum_aqi,

                "Years Covered": f"{oldest_year}-{latest_year}",

                "Trend": trend

            }

        except Exception as e:

            print(f"City Insight Error: {e}")
            return {}


    # =====================================================

if __name__ == "__main__":

    predictor = AirSensePredictor()

    sample = {

        "PM2.5": 82,

        "PM10": 120,

        "NO": 18,

        "NO2": 32,

        "NOx": 60,

        "NH3": 28,

        "CO": 1.2,

        "SO2": 16,

        "O3": 44,

        "Benzene": 2.5,

        "Toluene": 5,

        "Xylene": 1,

        "Year": 2024,

        "Month": 7,

        "Quarter": 3,

        "Day": 15,

        "DayOfWeek": 1,

        "WeekOfYear": 29,

        "Weekend": 0,

        "Season": "Monsoon",

        "PM_Ratio": 0.68,

        "NOx_Ratio": 1.87,

        "Pollution_Index": 305,

        "City": "Pune"

    }

    result = predictor.predict(sample)

    print("\n" + "=" * 70)

    print("Prediction Result")

    print("=" * 70)

    print("\nPrediction")

    for key, value in result["Prediction"].items():

        print(f"{key:<20}: {value}")

    print("\nStatus")

    for key, value in result["Status"].items():

        print(f"{key:<20}: {value}")

    print("\nHealth")

    for key, value in result["Health"].items():

        print(f"{key:<20}: {value}")

    print("\nTechnical")

    for key, value in result["Technical"].items():

        print(f"{key:<20}: {value}")

    print("\nCity Insights")

    for key, value in result["City Insights"].items():

        print(f"{key:<25}: {value}")