from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/raw/credit_risk_dataset.csv")
PROCESSED_PATH = Path("data/processed/credit_risk_processed.csv")


def load_data():
    df = pd.read_csv(DATA_PATH)

    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nSummary Statistics:")
    print(df.describe(include="all"))

    print("\nMissing Values:")
    print(df.isnull().sum())

    return df


def clean_data(df):
    df = df.copy()

    # Remove duplicate records
    df = df.drop_duplicates()

    # Numeric columns
    numeric_columns = df.select_dtypes(include=np.number).columns

    # Fill missing numeric values using median
    for col in numeric_columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Categorical columns
    categorical_columns = df.select_dtypes(include="object").columns

    # Fill missing categorical values using mode
    for col in categorical_columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def add_binning(df):
    df = df.copy()

    if "person_age" in df.columns:
        df["age_group"] = pd.cut(
            df["person_age"],
            bins=[0, 25, 35, 50, 100],
            labels=["18-25", "26-35", "36-50", "50+"]
        )

    if "person_income" in df.columns:
        df["income_group"] = pd.qcut(
            df["person_income"],
            q=4,
            labels=["Low", "Medium", "High", "Very High"],
            duplicates="drop"
        )

    return df


def save_processed_data(df):
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print(f"\nProcessed dataset saved to {PROCESSED_PATH}")


if __name__ == "__main__":
    dataframe = load_data()
    dataframe = clean_data(dataframe)
    dataframe = add_binning(dataframe)
    save_processed_data(dataframe)