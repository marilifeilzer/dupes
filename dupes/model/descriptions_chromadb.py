import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

# Load raw data
df = pd.read_csv('/Users/panamas/code/marili/dupes/raw_data/products_data.csv')

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
        embeddings=embeddings)
    return collection

def embedding_description_query_chromadb(collection, query, n_results):
    query_embedding = model.encode(query, show_progress_bar=False)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results)
    return results

# Main functionality
def embedding_description_get_recommendation(query, n_results = 5):
    dropped_desc = embedding_description_get_data(df)
    embeddings_desc = embedding_description_embed(dropped_desc)
    collection_desc = embedding_description_populate_chromadb(dropped_desc, embeddings_desc)
    recommendations_desc = embedding_description_query_chromadb(collection_desc, query, n_results)
    return recommendations_desc
