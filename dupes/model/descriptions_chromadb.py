import pandas as pd
from sentence_transformers import SentenceTransformer
from dupes.data.clean_data import clean_data
import os
import tarfile
from pathlib import Path
import chromadb

from dupes.data.gc_client import download_model, load_table_to_df, upload_model

model = SentenceTransformer("all-mpnet-base-v2")

CACHE_ROOT = Path(os.getenv("MODELS_CACHE_DIR", "models_cache"))
DESC_DIR = CACHE_ROOT / "descriptions"
CHROMA_DIR = DESC_DIR / "chroma"
CHROMA_ARCHIVE = DESC_DIR / "chroma.tar.gz"
GCS_CHROMA_BLOB = os.getenv("DESCRIPTION_CHROMA_BLOB", "descriptions/chroma.tar.gz")

# Make sure paths exist inside container
def _ensure_dirs() -> None:
    DESC_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def _archive_chroma(src_dir: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(src_dir, arcname=".")


def _extract_chroma(archive_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(dest_dir)

    # Check if files exist locally, if not downloads them
def ensure_description_artifacts() -> None:
    _ensure_dirs()
    
    chroma_db_path = CHROMA_DIR / "chroma.sqlite3"
    if not chroma_db_path.is_file():
        if CHROMA_ARCHIVE.is_file():
            _extract_chroma(CHROMA_ARCHIVE, CHROMA_DIR)
        else:
            download_model(GCS_CHROMA_BLOB, CHROMA_ARCHIVE)
            _extract_chroma(CHROMA_ARCHIVE, CHROMA_DIR)


def _get_client() -> chromadb.PersistentClient:
    ensure_description_artifacts()
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def embedding_description_get_data(df: pd.DataFrame):
    dropped = df[["product_id", "description"]]
    dropped = df.dropna(subset=["description"], axis=0)
    return dropped


def embedding_description_embed(dropped: pd.DataFrame):
    embeddings = model.encode(dropped["description"].values, show_progress_bar=True)
    return embeddings


def embedding_description_populate_chromadb(dropped: pd.DataFrame, embeddings, client=None):
    client = client or _get_client()
    collection = client.get_or_create_collection(name="description_embed")
    collection.add(ids=list(dropped["product_id"].values), embeddings=embeddings)
    return collection


def embedding_description_query_chromadb(query, n_results=5):
    collection = _get_client().get_collection(name="description_embed")
    query_embedding = model.encode(query, show_progress_bar=False)
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    results = results["ids"][0]
    df = load_table_to_df()
    product_names = [
        df.loc[df["product_id"] == product, ["product_name", "price_eur", "description", "volume_ml"]]
        for product in results
    ]

    return product_names


def embedding_description_get_recommendation():
    df = load_table_to_df()

    dropped_desc = embedding_description_get_data(df)
    embeddings_desc = embedding_description_embed(dropped_desc)
    embedding_description_populate_chromadb(dropped_desc, embeddings_desc)

    # Save the files into the bucket
    _archive_chroma(CHROMA_DIR, CHROMA_ARCHIVE)
    upload_model(CHROMA_ARCHIVE, GCS_CHROMA_BLOB)


# Load raw data
# This needs to be linked to the path of your csv


# # TODO REPLACE WITH THE BIGQUERY
# df = pd.read_csv('raw_data/products_clean_600_ingredients.csv')
# df_cleaned = clean_data(df)

def create_description_db(df: pd.DataFrame) -> None:
    """Create description ChromaDB collection from scratch"""
    _ensure_dirs()
    import shutil
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    dropped_desc = embedding_description_get_data(df)
    embeddings_desc = embedding_description_embed(dropped_desc)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_description_populate_chromadb(dropped_desc, embeddings_desc, client=client)

    # Save the files into the bucket
    _archive_chroma(CHROMA_DIR, CHROMA_ARCHIVE)
    upload_model(CHROMA_ARCHIVE, GCS_CHROMA_BLOB)
