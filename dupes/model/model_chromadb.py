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


chroma_client = chromadb.PersistentClient(path="raw_data/")


def chromadb_model_testing(embedding, test_ingredient_embedding, cols=['product_id', 'ingredients_formula']):
    # client = chromadb.EphemeralClient()
    embeds = embedding.iloc[:,1:].values
    collection = chroma_client.get_or_create_collection(name="ing_formula_embedding")
    collection.add(
    ids=list(embedding[cols[0]].values),

    embeddings=embeds
    )

    breakpoint()
    results= collection.query(
    query_embeddings=[embeds[0]],
    n_results=5,
)


    return results


if __name__ == "__main__":
    df = pd.read_csv('/home/marili/code/marilifeilzer/dupes/raw_data/products_600_v5.csv')
    df_withouttest = df
    embedding = embedding_ingredients(df=df_withouttest, cols= ['product_id', 'ingredients_formula'])
    test_ingredient_embedding = df.iloc[-1, 1:]
    test_ingredient_embedding= embedding_ingredients(df=df_withouttest, cols= ['product_id', 'ingredients_formula'])
    # Save embedding to raw data folder
    output_path = "/home/marili/code/marilifeilzer/dupes/raw_data/embedding.csv"
    embedding.to_csv(output_path, index=False)



    result= chromadb_model_testing(embedding, cols=['product_id', 'ingredients_formula'], test_ingredient_embedding= test_ingredient_embedding)
    print(result)
