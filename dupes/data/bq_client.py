import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

load_dotenv()  # loads .env into os.environ

REQUIRED_ENV_VARS = ["GCP_PROJECT_ID", "BQ_DATASET", "BQ_TABLE", "GOOGLE_APPLICATION_CREDENTIALS"]


def _ensure_env() -> tuple[str, str, str, str]:
    """
    Validate required env vars and key file presence.
    Returns (project_id, dataset, table, credentials_path).
    Raises ValueError with a clear message if anything is missing.
    """
    missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing:
        raise ValueError(f"Missing required env vars for BigQuery: {', '.join(missing)}")

    project_id = os.getenv("GCP_PROJECT_ID")
    dataset = os.getenv("BQ_DATASET")
    table = os.getenv("BQ_TABLE")
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if not Path(credentials_path).is_file():
        raise ValueError(f"GOOGLE_APPLICATION_CREDENTIALS path does not exist: {credentials_path}")

    return project_id, dataset, table, credentials_path


def get_bq_client() -> bigquery.Client:
    project_id, _, _, _ = _ensure_env()

    return bigquery.Client(project=project_id)


def load_table_to_df(dataset: str | None = None, table: str | None = None):
    """
    Load the BigQuery table into a DataFrame.
    """

    project_id, env_dataset, env_table, _ = _ensure_env()

    dataset = dataset or env_dataset
    table = table or env_table

    client = get_bq_client()
    table_id = f"{project_id}.{dataset}.{table}"

    query = f"SELECT * FROM `{table_id}`"
    job = client.query(query)
    df = job.result().to_dataframe()

    return df
