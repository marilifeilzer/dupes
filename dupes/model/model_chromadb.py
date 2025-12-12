import chromadb # , EmbeddingFunction
import pandas as pd
from dupes.data.properties import encode_properties, use_encoder_load
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import os
import tarfile
import shutil
from pathlib import Path

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

from dupes.data.gc_client import download_model, upload_model
from dupes.data.properties import encode_properties, use_encoder_load

# Common instances
chroma_client = chromadb.PersistentClient(path="raw_data/")



# Import centralized paths
from dupes.model.model_paths import (
    MLB_INGREDIENTS_PATH as MLB_PATH,
    CHROMA_INGREDIENTS_DIR as CHROMA_DIR,
    CHROMA_INGREDIENTS_ARCHIVE as CHROMA_ARCHIVE,
    INGREDIENTS_DIR as INGR_DIR,
    GCS_MLB_BLOB,
    GCS_CHROMA_INGREDIENTS_BLOB as GCS_CHROMA_BLOB
)

# Checks for directories and internal paths
def _ensure_dirs() -> None:
    INGR_DIR.mkdir(parents=True, exist_ok=True)
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
def ensure_ingredients_artifacts() -> None:
    _ensure_dirs()

    if not MLB_PATH.is_file():
        download_model(GCS_MLB_BLOB, MLB_PATH)

    chroma_db_path = CHROMA_DIR / "chroma.sqlite3"
    if not chroma_db_path.is_file():
        if CHROMA_ARCHIVE.is_file():
            _extract_chroma(CHROMA_ARCHIVE, CHROMA_DIR)
        else:
            download_model(GCS_CHROMA_BLOB, CHROMA_ARCHIVE)
            _extract_chroma(CHROMA_ARCHIVE, CHROMA_DIR)


def _get_client() -> chromadb.PersistentClient:
    ensure_ingredients_artifacts()
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def embedding_ingredients_get_data(df: pd.DataFrame) -> pd.DataFrame:
    dropped = df[["product_id", "formula"]]
    dropped = dropped.dropna(subset=["formula"], axis=0)
    return dropped


def embedding_ingredients(df: pd.DataFrame, exist: bool = False):
    df.formula = df.formula.apply(lambda x: eval(str(x)))
    if exist:
        return use_encoder_load(df, col="formula")
    return encode_properties(df, col="formula")


def create_metadata_dictionairy_properties(
    df: pd.DataFrame, cols=["tipo_de_cabello", "color_de_cabello", "propiedad"]
):
    metadata_dict_encoded = []
    for _, row in df.iterrows():
        all_dict = {}
        for col in cols:
            tipo_values = row[col]
            tipo_dict = {}
            for i in tipo_values:
                i = i.lower().strip().replace(" ", "_")
                tipo_dict[i] = 1
            all_dict.update(tipo_dict)
        metadata_dict_encoded.append(all_dict)
    return metadata_dict_encoded


def create_metadata_dictionairy(
    df: pd.DataFrame, cols=["tipo_de_cabello", "color_de_cabello", "propiedad"]
):
    metadata_dict_encoded = []
    for col in cols:
        df[col] = df[col].apply(lambda x: x.split(","))

    for _, row in df.iterrows():
        all_dict = {}
        for col in cols:
            tipo_values = row[col]

            tipo_dict = {}
            for i in tipo_values:
                i = i.lower().strip().replace(" ", "_")
                tipo_dict[i] = 1
            all_dict.update(tipo_dict)
        metadata_dict_encoded.append(all_dict)
    return metadata_dict_encoded


def embedding_ingredients_populate_chromadb(
    dropped: pd.DataFrame, embeddings, properties_metadata, client=None
):
    client = client or _get_client()
    collection = client.get_or_create_collection(name="ingredients_embed_v2")
    collection.add(
        ids=list(dropped["product_id"].values),
        embeddings=embeddings.iloc[:, 1:].values,
        metadatas=properties_metadata,
    )
    return collection


def embedding_description_query_filtering_chromadb(collection, query, n_results, where=None):
    MODEL = SentenceTransformer("all-mpnet-base-v2")
    query_embedding = MODEL.encode(query)

    results = collection.query(
        query_embeddings=[query_embedding], n_results=n_results, where=where)

    return results


def query_chromadb_ingredients(collection, query_embedding, n_results, where=None):
    if len(where.items())> 1:
        and_list= []

        for each in where.items():
            and_list.append({each[0]: each[1]})

        filter= {'$and': and_list}

    query_embedding= query_embedding.iloc[:,:].to_numpy().flatten() #adjust this without product id

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=filter
    )

    return results

# Now recieves the dataframe instead of loading the file
def create_ingr_db(df: pd.DataFrame) -> None:
    _ensure_dirs()
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    dropped = embedding_ingredients_get_data(df)
    embed_ingredients = embedding_ingredients(dropped)

    # Get full data for metadata (not just the dropped version with only product_id and formula)
    full_data_for_metadata = df[df['product_id'].isin(dropped['product_id'])].copy()
    metadata_dict = create_metadata_dictionairy(full_data_for_metadata)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_ingredients_populate_chromadb(dropped, embed_ingredients, metadata_dict, client=client)

    # Save the files into the bucket
    _archive_chroma(CHROMA_DIR, CHROMA_ARCHIVE)
    upload_model(MLB_PATH, GCS_MLB_BLOB)
    upload_model(CHROMA_ARCHIVE, GCS_CHROMA_BLOB)

def main_res_product_id(product_id, df):
    # Use the proper client that points to models_cache
    ensure_ingredients_artifacts()  # Ensure ChromaDB is available
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name="ingredients_embed_v2")
    product = df.loc[df['product_id'] == product_id]
    embed_ex= embedding_ingredients(product, True)
    metadata_ex= create_metadata_dictionairy(product)
    results= query_chromadb_ingredients(collection, embed_ex, 5, where=metadata_ex[0])
    return results


if __name__ == "__main__":
    print("Building ChromaDB collection...")
    create_ingr_db()
    print("Done.")
