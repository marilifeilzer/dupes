import chromadb # , EmbeddingFunction
import pandas as pd
from dupes.data.properties import encode_properties
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

# Common instances
model = SentenceTransformer("all-mpnet-base-v2")
chroma_client = chromadb.PersistentClient(path="raw_data/")

def embedding_ingredients_get_data(df: pd.DataFrame):
    dropped = df[['product_id', 'ingredients_formula']]
    dropped = df.dropna(subset=['ingredients_formula'], axis=0)
    return dropped

def embedding_ingredients(df: pd.DataFrame, ):
    df= df.dropna(subset=["ingredients_formula"], axis=0)
    df.ingredients_formula = df.ingredients_formula.apply(lambda x: eval(x))
    encoded = encode_properties(df, col = 'ingredients_formula' )
    return encoded

def create_metadata_dictionairy(df: pd.DataFrame, cols=["tipo_de_cabello", "color_de_cabello", "propiedad"]):

    dropped= df.dropna(subset=cols, axis=0)
    properties_metadata = dropped[cols].to_dict(orient='records')
    return properties_metadata


def embedding_ingredients_populate_chromadb(dropped: pd.DataFrame, embeddings, properties_metadata):
    breakpoint()
    collection = chroma_client.get_or_create_collection(name="ingredients_embed")
    collection.add(
        ids=list(dropped['product_id'].values),
        embeddings=embeddings.iloc[:,1:].values,
        metadatas=properties_metadata
    )
    return collection

def embedding_description_query_filtering_chromadb(collection, query, n_results, where=None):
    query_embedding = model.encode(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where
    )

    return results

# Main functionality
def embedding_description_get_recommendation(df, query, where, n_results = 5):
    dropped= embedding_ingredients_get_data(df)
    embed_ingredients = embedding_ingredients(dropped)
    metadata_dict= create_metadata_dictionairy(dropped)
    collection= embedding_ingredients_populate_chromadb(dropped, embed_ingredients, metadata_dict)
    results= embedding_description_query_filtering_chromadb(collection, query,  n_results, where)
    return results

if __name__ == "__main__":
    df= pd.read_csv("/home/marili/code/marilifeilzer/dupes/raw_data/products_600_v3.csv")
    dropped= embedding_ingredients_get_data(df)
    breakpoint()
    embed_ingredients = embedding_ingredients(dropped)
    metadata_dict= create_metadata_dictionairy(dropped)
    collection= embedding_ingredients_populate_chromadb(dropped, embed_ingredients, metadata_dict)

    print(collection)
