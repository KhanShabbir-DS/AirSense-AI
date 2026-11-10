"""
=========================================================
AirSense AI
Feature Engineering
=========================================================
"""

import numpy as np

from data_loader import DataLoader
from preprocessing import DataPreprocessor


def get_season(month):
    """Return the season label used by the trained models."""
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Summer"
    if month in (6, 7, 8, 9):
        return "Monsoon"
    return "PostMonsoon"


def compute_pollution_features(pm25, pm10, no2, nox, co, so2):
    """Compute derived pollution features using the training pipeline formulas."""
    pm_ratio = pm25 / (pm10 + 1)
    nox_ratio = no2 / (nox + 1)
    pollution_index = (pm25 + pm10 + no2 + so2 + co * 100) / 5
    return pm_ratio, nox_ratio, pollution_index


class FeatureEngineer:

    def __init__(self, df):

        self.df = df.copy()

    # --------------------------------------------------

    def create_date_features(self):

        print("Creating Date Features...")

        self.df["Year"] = self.df["Date"].dt.year
        self.df["Month"] = self.df["Date"].dt.month
        self.df["Quarter"] = self.df["Date"].dt.quarter
        self.df["Day"] = self.df["Date"].dt.day
        self.df["DayOfWeek"] = self.df["Date"].dt.dayofweek
        self.df["WeekOfYear"] = self.df["Date"].dt.isocalendar().week.astype(int)

        self.df["Weekend"] = (
            self.df["DayOfWeek"] >= 5
        ).astype(int)

    # --------------------------------------------------

    def create_season_feature(self):

        print("Creating Season Feature...")
        self.df["Season"] = self.df["Month"].apply(get_season)

    # --------------------------------------------------

    def create_pollution_features(self):

        print("Creating Pollution Features...")

        ratios = self.df.apply(
            lambda row: compute_pollution_features(
                row["PM2.5"],
                row["PM10"],
                row["NO2"],
                row["NOx"],
                row["CO"],
                row["SO2"],
            ),
            axis=1,
            result_type="expand",
        )
        self.df["PM_Ratio"] = ratios[0]
        self.df["NOx_Ratio"] = ratios[1]
        self.df["Pollution_Index"] = ratios[2]

    # --------------------------------------------------

    def build(self):

        print("\n" + "=" * 60)
        print("Starting Feature Engineering")
        print("=" * 60)

        self.create_date_features()

        self.create_season_feature()

        self.create_pollution_features()

        print("\nFeature Engineering Completed")

        print("-" * 40)

        print(f"Total Rows     : {len(self.df)}")

        print(f"Total Columns  : {len(self.df.columns)}")

        return self.df


# =========================================================

if __name__ == "__main__":

    loader = DataLoader()

    raw_df = loader.load()

    processor = DataPreprocessor(raw_df)

    clean_df = processor.preprocess()

    engineer = FeatureEngineer(clean_df)

    final_df = engineer.build()

    print("\nFirst Five Rows\n")

    print(final_df.head())

    print("\nNew Columns Added\n")

    new_cols = [

        "Year",

        "Month",

        "Quarter",

        "Day",

        "DayOfWeek",

        "WeekOfYear",

        "Weekend",

        "Season",

        "PM_Ratio",

        "NOx_Ratio",

        "Pollution_Index"

    ]

    print(final_df[new_cols].head())