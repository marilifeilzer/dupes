import chromadb # , EmbeddingFunction
import pandas as pd
from dupes.data.properties import encode_properties

def embedding_ingredients(df: pd.DataFrame, cols= ['product_id', 'ingredients_formula']):
    dropped= df.dropna(subset=[cols[1]], axis=0)
    dropped.ingredients_formula = dropped.ingredients_formula.apply(lambda x: eval(x))
    encoded = encode_properties(dropped, col = 'ingredients_formula' )
    encoded_feat_count = encoded.shape[1] + 1 - dropped.shape[1]
    embedding = encoded[[cols[0]] + list(encoded.columns[-encoded_feat_count:])]
    return embedding



# def chromadb_model_testing():
#     # client = chromadb.EphemeralClient()
#     collection = chroma_client.get_or_create_collection(name="my_collection")

#     client = chromadb.PersistentClient(path="/path/to/save/to")

#     collection.add(
#     ids=["id1", "id2"],
#     embeddings=
#     )

#     results = collection.query(
#         query_texts=["This is a query document about hawaii"], # Chroma will embed this for you
#         n_results=2 # how many results to return
#     )


if __name__ == "__main__":
    df = pd.read_csv('/home/marili/code/marilifeilzer/dupes/raw_data/products_600_v5.csv')
    result = embedding_ingredients(df=df, cols= ['product_id', 'ingredients_formula'])

    # Save result to raw data folder
    output_path = "/home/marili/code/marilifeilzer/dupes/raw_data/embedding.csv"
    result.to_csv(output_path, index=False)

    print(result)
    print(f"Saved to: {output_path}")
