import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from dupes.data.clean_data import clean_data

# Load raw data
# This needs to be linked to the path of your csv

df = pd.read_csv('/home/marili/code/marilifeilzer/dupes/raw_data/products_data__0412 (3).csv')
df_cleaned = clean_data(df)

# Common instances
model = SentenceTransformer("all-mpnet-base-v2")
chroma_client = chromadb.PersistentClient(path="raw_data/")

# Helper functions
def embedding_description_get_data(df: pd.DataFrame):
    dropped = df[['product_id', 'description']]
    dropped = df.dropna(subset=['description'], axis=0)
    return dropped

def embedding_description_embed(dropped: pd.DataFrame):
    embeddings = model.encode(dropped['description'].values, show_progress_bar=True)
    return embeddings

def embedding_description_populate_chromadb(dropped: pd.DataFrame, embeddings):
    collection = chroma_client.get_or_create_collection(name="description_embed")
    collection.add(
        ids=list(dropped['product_id'].values),
        embeddings=embeddings
    )
    return collection

def embedding_description_query_chromadb(query, n_results=5):
    collection = chroma_client.get_collection(name="description_embed")
    query_embedding = model.encode(query, show_progress_bar=False)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results)

    results = results["ids"][0]
    product_names = [df_cleaned.loc[df_cleaned["product_id"]==product, ["product_name","price_eur", "description"]]for product in results]

    return product_names

# Main functionality
def embedding_description_get_recommendation():

    dropped_desc = embedding_description_get_data(df_cleaned)
    embeddings_desc = embedding_description_embed(dropped_desc)
    collection_desc = embedding_description_populate_chromadb(dropped_desc, embeddings_desc)
