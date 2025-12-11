import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery, storage

load_dotenv()  # loads .env into os.environ

REQUIRED_BQ_ENV_VARS = [
    "GCP_PROJECT",
    "BQ_DATASET",
    "BQ_TABLE",
    "GOOGLE_APPLICATION_CREDENTIALS",
]

def _ensure_bq_env() -> tuple[str, str, str, str]:
    """
    Validate required env vars and key file presence for BigQuery.
    Returns (project_id, dataset, table, credentials_path).
    """
    missing = [var for var in REQUIRED_BQ_ENV_VARS if not os.getenv(var)]
    if missing:
        raise ValueError(f"Missing required env vars for BigQuery: {', '.join(missing)}")

    project_id = os.getenv("GCP_PROJECT")
    dataset = os.getenv("BQ_DATASET")
    table = os.getenv("BQ_TABLE")
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if not Path(credentials_path).is_file():
        raise ValueError(f"GOOGLE_APPLICATION_CREDENTIALS path does not exist: {credentials_path}")

    return project_id, dataset, table, credentials_path


def get_bq_client() -> bigquery.Client:
    project_id, _, _, _ = _ensure_bq_env()
    return bigquery.Client(project=project_id)


def load_table_to_df(dataset: str | None = None, table: str | None = None) -> pd.DataFrame:
    """
    Load the BigQuery table into a DataFrame.
    """
    project_id, env_dataset, env_table, _ = _ensure_bq_env()

    dataset = dataset or env_dataset
    table = table or env_table
    client = get_bq_client()
    table_id = f"{project_id}.{dataset}.{table}"

    query = f"SELECT * FROM `{table_id}`"
    job = client.query(query)
    df = job.result().to_dataframe()

    return df

REQUIRED_GCS_ENV_VARS = [
    "GCP_PROJECT",
    "GCS_BUCKET_MODELS",
]

def _ensure_gcs_env() -> tuple[str, str, str | None]:
    """
    Validate env vars needed for GCS model storage.
    Returns (project_id, bucket_name, prefix).
    """
    missing = [var for var in REQUIRED_GCS_ENV_VARS if not os.getenv(var)]
    if missing:
        raise ValueError(f"Missing required env vars for GCS: {', '.join(missing)}")

    project_id = os.getenv("GCP_PROJECT")
    bucket_name = os.getenv("GCS_BUCKET_MODELS")
    prefix_raw = os.getenv("GCS_MODELS_PREFIX", "").rstrip("/")
    prefix = prefix_raw or None
    return project_id, bucket_name, prefix


def _build_blob_path(blob_name: str, prefix: str | None) -> str:
    return f"{prefix}/{blob_name}" if prefix else blob_name


def upload_model(local_path: str | Path, blob_name: str) -> str:
    """
    Upload a local file to the models bucket.
    Returns the full gs:// URI.
    """
    project_id, bucket_name, prefix = _ensure_gcs_env()
    blob_path = _build_blob_path(blob_name, prefix)

    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(str(local_path))

    return f"gs://{bucket_name}/{blob_path}"


def download_model(blob_name: str, dest_path: str | Path) -> Path:
    """
    Download a model blob from the models bucket to dest_path.
    """
    project_id, bucket_name, prefix = _ensure_gcs_env()
    blob_path = _build_blob_path(blob_name, prefix)

    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    if not blob.exists():
        raise FileNotFoundError(f"Not found: gs://{bucket_name}/{blob_path}")

    blob.download_to_filename(dest_path)
    return dest_path

if __name__ == "__main__":
    load_table_to_df()
