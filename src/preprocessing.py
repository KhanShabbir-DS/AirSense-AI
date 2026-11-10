"""
=========================================================
AirSense AI
Data Preprocessing
=========================================================
"""

import pandas as pd

from data_loader import DataLoader
from config import POLLUTANT_COLUMNS


class DataPreprocessor:

    def __init__(self, df):

        self.df = df.copy()

    def preprocess(self):

        print("\n" + "=" * 60)
        print("Starting Data Preprocessing")
        print("=" * 60)

        # -----------------------------
        # Convert Date
        # -----------------------------
        self.df["Date"] = pd.to_datetime(
            self.df["Date"],
            errors="coerce"
        )

        # Remove invalid dates
        self.df = self.df.dropna(subset=["Date"])

        # -----------------------------
        # Remove duplicates
        # -----------------------------
        before = len(self.df)

        self.df.drop_duplicates(inplace=True)

        duplicates = before - len(self.df)

        # -----------------------------
        # Remove rows with missing AQI
        # -----------------------------
        before = len(self.df)

        self.df = self.df.dropna(subset=["AQI"])

        removed_target = before - len(self.df)

        # -----------------------------
        # Fill missing pollutant values
        # City-wise median
        # -----------------------------
        for col in POLLUTANT_COLUMNS:

            self.df[col] = self.df.groupby("City")[col].transform(
                lambda x: x.fillna(x.median())
            )

        # Remaining missing → overall median
        self.df[POLLUTANT_COLUMNS] = self.df[
            POLLUTANT_COLUMNS
        ].fillna(
            self.df[POLLUTANT_COLUMNS].median()
        )

        # -----------------------------
        # Fill AQI Bucket
        # -----------------------------
        self.df["AQI_Bucket"] = self.df[
            "AQI_Bucket"
        ].fillna("Unknown")

        # -----------------------------
        # Sort
        # -----------------------------
        self.df = self.df.sort_values(
            ["City", "Date"]
        )

        self.df.reset_index(
            drop=True,
            inplace=True
        )

        print("\nPreprocessing Summary")
        print("-" * 40)

        print(f"Final Shape           : {self.df.shape}")

        print(f"Duplicates Removed    : {duplicates}")

        print(f"Rows Removed (No AQI) : {removed_target}")

        print(
            f"Remaining Missing     : {self.df.isnull().sum().sum()}"
        )

        return self.df


if __name__ == "__main__":

    loader = DataLoader()

    df = loader.load()

    processor = DataPreprocessor(df)

    clean_df = processor.preprocess()

    print("\nFirst Five Clean Rows\n")

    print(clean_df.head())