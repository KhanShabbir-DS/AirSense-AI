"""
=========================================================
AirSense AI
Data Loader
=========================================================
"""

import pandas as pd

from config import DATA_FILE


class DataLoader:

    def __init__(self):

        self.df = None

    def load(self):

        print("=" * 60)
        print("Loading Dataset...")
        print("=" * 60)

        self.df = pd.read_csv(DATA_FILE)

        print(f"Dataset Shape : {self.df.shape}")

        print(f"Cities        : {self.df['City'].nunique()}")

        print(f"Columns       : {self.df.shape[1]}")

        print("\nColumn Names")

        print(self.df.columns.tolist())

        return self.df


if __name__ == "__main__":

    loader = DataLoader()

    df = loader.load()

    print("\nFirst Five Rows\n")

    print(df.head())