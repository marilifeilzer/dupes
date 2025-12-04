import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-mpnet-base-v2")
chroma_client = chromadb.PersistentClient(path="raw_data/")

# df = pd.read_csv('/Users/panamas/code/marili/dupes/raw_data/products_data.csv')

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

def embedding_description_query_chromadb(collection, query, n_results = 5):
    query_embedding = model.encode(query, show_progress_bar=False)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results)


    return results

# get_data
# embed
# populate_chromadb
# query_chromadb

if __name__ == "__main__":
    df = pd.read_csv('/Users/panamas/code/marili/dupes/raw_data/products_data.csv')
    dropped = embedding_description_get_data(df)
    embeddings = embedding_description_embed(dropped)
    collection = embedding_description_populate_chromadb(dropped, embeddings)

    #collection = chroma_client.get_or_create_collection(name="description_embed")

    breakpoint()
    result = embedding_description_query_chromadb(collection, 'Im looking for greasy and curly hair.'
)
