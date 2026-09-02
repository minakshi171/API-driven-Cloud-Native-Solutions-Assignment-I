from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


DATA_PATH = Path("data/processed/credit_risk_processed.csv")
CHART_DIR = Path("outputs/charts")


def run_eda():
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    print("\n===== EDA STARTED =====")

    # -------------------------------------------------
    # 1. Target distribution
    # -------------------------------------------------

    if "loan_status" in df.columns:
        counts = df["loan_status"].value_counts().sort_index()

        plt.figure(figsize=(6, 4))
        counts.plot(kind="bar")
        plt.title("Loan Default Distribution")
        plt.xlabel("Loan Status")
        plt.ylabel("Number of Applicants")
        plt.tight_layout()
        plt.savefig(CHART_DIR / "loan_status_distribution.png")
        plt.close()

    # -------------------------------------------------
    # 2. Age distribution
    # -------------------------------------------------

    if "person_age" in df.columns:
        plt.figure(figsize=(7, 4))
        plt.hist(df["person_age"], bins=20)
        plt.title("Applicant Age Distribution")
        plt.xlabel("Age")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(CHART_DIR / "age_distribution.png")
        plt.close()

    # -------------------------------------------------
    # 3. Income distribution
    # -------------------------------------------------

    if "person_income" in df.columns:
        plt.figure(figsize=(7, 4))
        plt.hist(df["person_income"], bins=30)
        plt.title("Applicant Income Distribution")
        plt.xlabel("Income")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(CHART_DIR / "income_distribution.png")
        plt.close()

    # -------------------------------------------------
    # 4. Loan amount vs default
    # -------------------------------------------------

    if {"loan_status", "loan_amnt"}.issubset(df.columns):
        grouped = df.groupby("loan_status")["loan_amnt"].mean()

        plt.figure(figsize=(6, 4))
        grouped.plot(kind="bar")
        plt.title("Average Loan Amount by Loan Status")
        plt.xlabel("Loan Status")
        plt.ylabel("Average Loan Amount")
        plt.tight_layout()
        plt.savefig(CHART_DIR / "loan_amount_vs_status.png")
        plt.close()

    # -------------------------------------------------
    # 5. Numeric correlations
    # -------------------------------------------------

    numeric_df = df.select_dtypes(include="number")

    correlations = numeric_df.corr()

    print("\nCorrelation Matrix:")
    print(correlations)

    correlations.to_csv(
        "outputs/reports/correlation_matrix.csv"
    )

    plt.figure(figsize=(10, 8))
    plt.imshow(correlations, aspect="auto")
    plt.colorbar()
    plt.xticks(
        range(len(correlations.columns)),
        correlations.columns,
        rotation=90
    )
    plt.yticks(
        range(len(correlations.columns)),
        correlations.columns
    )
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "correlation_matrix.png")
    plt.close()

    print("\nEDA charts successfully generated.")
    print("===== EDA COMPLETED =====")


if __name__ == "__main__":
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    run_eda()