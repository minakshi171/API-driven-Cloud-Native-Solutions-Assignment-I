from datetime import timedelta

from prefect import flow, task, get_run_logger

from src.preprocess import (
    load_data,
    clean_data,
    add_binning,
    save_processed_data
)
from src.eda import run_eda
from src.train_model import train_model


@task(name="Data Ingestion")
def ingest_task():
    logger = get_run_logger()

    logger.info("Starting data ingestion...")

    df = load_data()

    logger.info(
        f"Dataset successfully loaded. "
        f"Rows={df.shape[0]}, Columns={df.shape[1]}"
    )

    return df


@task(name="Data Preprocessing")
def preprocessing_task(df):
    logger = get_run_logger()

    logger.info("Starting preprocessing.")

    df = clean_data(df)

    df = add_binning(df)

    save_processed_data(df)

    logger.info(
        "Missing values handled, duplicates removed "
        "and binning completed."
    )

    return df


@task(name="Exploratory Data Analysis")
def eda_task():
    logger = get_run_logger()

    logger.info("Starting EDA.")

    run_eda()

    logger.info(
        "Correlation and visualization analysis completed."
    )


@task(name="Machine Learning")
def model_task():
    logger = get_run_logger()

    logger.info("Starting model training.")

    accuracy, auc = train_model()

    logger.info(
        f"Model training completed. "
        f"Accuracy={accuracy:.4f}, ROC_AUC={auc:.4f}"
    )


@flow(name="Credit Risk Data Pipeline", log_prints=True)
def credit_risk_pipeline():

    df = ingest_task()

    preprocessing_task(df)

    eda_task()

    model_task()


if __name__ == "__main__":

    credit_risk_pipeline.serve(
        name="credit-risk-deployment",
        interval=timedelta(minutes=2)
    )