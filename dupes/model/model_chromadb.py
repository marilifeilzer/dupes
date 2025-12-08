import chromadb # , EmbeddingFunction
import pandas as pd
from dupes.data.properties import encode_properties, use_encoder_load
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

# Common instances
model = SentenceTransformer("all-mpnet-base-v2")
chroma_client = chromadb.PersistentClient(path="raw_data/")

def embedding_ingredients_get_data(df: pd.DataFrame):
    dropped = df[['product_id', 'formula']]
    dropped = df.dropna(subset=['formula'], axis=0)
    return dropped

def embedding_ingredients(df: pd.DataFrame, exist=False):
    df.formula = df.formula.apply(lambda x: eval(x))
    if exist==True:
        encoded = use_encoder_load(df, col = 'formula')
    else:
        encoded= encode_properties(df, col= 'formula')



    return encoded

def create_metadata_dictionairy(df: pd.DataFrame, cols=["tipo_de_cabello", "color_de_cabello", "propiedad"]):

    dropped= df.dropna(subset=cols, axis=0)
    metadata_dict_encoded= []
    for col in cols:
        dropped[col]= dropped[col].apply(lambda x: x.split(','))


    for i, row in dropped.iterrows():
        all_dict= {}
        for col in cols:
            tipo_values= row[col]

            tipo_dict= {}
            for i in tipo_values:
                i=i.lower().strip().replace(' ', '_')
                tipo_dict[i]=1
            all_dict.update(tipo_dict)
        metadata_dict_encoded.append(all_dict)

    # properties_metadata = dropped[cols].to_dict(orient='records')
    return metadata_dict_encoded


def embedding_ingredients_populate_chromadb(dropped: pd.DataFrame, embeddings, properties_metadata):
    collection = chroma_client.get_or_create_collection(name="ingredients_embed_v2")
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

def query_chromadb_ingredients(collection, query_embedding, n_results, where=None):
    if len(where.items())> 1:
        and_list= []

        for each in where.items():
            and_list.append({each[0]: each[1]})

        filter= {'$and': and_list}

    query_embedding= query_embedding.iloc[:,1:].to_numpy().flatten() #added this line


    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=filter
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

def create_ingr_db() -> None:
    df= pd.read_csv("/home/marili/code/marilifeilzer/dupes/raw_data/products_clean_600_ingredients.csv")
    dropped= embedding_ingredients_get_data(df)
    embed_ingredients = embedding_ingredients(dropped)
    metadata_dict= create_metadata_dictionairy(dropped)
    embedding_ingredients_populate_chromadb(dropped, embed_ingredients, metadata_dict)

if __name__ == "__main__":
    df= pd.read_csv("/home/marili/code/marilifeilzer/dupes/raw_data/products_clean_ingredients_rank_2.csv")
    df= df.dropna(subset=["formula"], axis=0)
    product_example = df.iloc[50]
    product_example= product_example.to_frame().T
    embeddings= embedding_ingredients(df, False)
    metadata_dict= create_metadata_dictionairy(df)
    collection= embedding_ingredients_populate_chromadb(df, embeddings, metadata_dict)
    embed_ex= embedding_ingredients(product_example, True)
    metadata_ex= create_metadata_dictionairy(product_example)

    # res= query_chromadb_ingredients(collection, embeddings.iloc[50, 1:].astype(int).values, 5, where=metadata_dict[0])
    res= query_chromadb_ingredients(collection, embed_ex, 5, where=metadata_ex[0])
    breakpoint()
    print(res)
