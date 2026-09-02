from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/processed/credit_risk_processed.csv")
MODEL_PATH = Path("models/credit_risk_model.joblib")
REPORT_DIR = Path("outputs/reports")
CHART_DIR = Path("outputs/charts")


def train_model():
    df = pd.read_csv(DATA_PATH)

    target = "loan_status"

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    X = df.drop(columns=[target])
    y = df[target]

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    print("Numeric Features:")
    print(numeric_features)

    print("\nCategorical Features:")
    print(categorical_features)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
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
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced"
                )
            )
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining model...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    print("\nAccuracy:", accuracy)
    print("ROC AUC:", auc)

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(
        REPORT_DIR / "model_metrics.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(f"Accuracy: {accuracy}\n")
        file.write(f"ROC AUC: {auc}\n\n")
        file.write(
            classification_report(
                y_test,
                predictions
            )
        )

    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")

    return accuracy, auc


if __name__ == "__main__":
    train_model()