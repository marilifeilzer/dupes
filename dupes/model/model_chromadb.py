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
    df.formula = df.formula.apply(lambda x: eval(str(x))) #added str
    if exist==True:
        encoded = use_encoder_load(df, col = 'formula')
    else:
        encoded= encode_properties(df, col= 'formula')



    return encoded

def create_metadata_dictionairy(df: pd.DataFrame, cols=["tipo_de_cabello", "color_de_cabello", "propiedad"]):
    # dropped= df.dropna(subset=cols, axis=0)
    metadata_dict_encoded= []
    print(f"This is df imput of create_metadata_dict {df}")
    ######COMMENDED THIS OUT IN ORDER FOR APIEND POINT TO WORK
    # for col in cols:
    #     df[col]= df[col].apply(lambda x: x.split(',')) #changed dropped to df


    for i, row in df.iterrows():
        all_dict= {}
        for col in cols:
            tipo_values= row[col]

            tipo_dict= {}
            for i in tipo_values:
                i=i.lower().strip().replace(' ', '_')
                tipo_dict[i]=1
            all_dict.update(tipo_dict)
        metadata_dict_encoded.append(all_dict)
    print(f"This is meta_dict_encoded {metadata_dict_encoded}")
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

    query_embedding= query_embedding.iloc[:,:].to_numpy().flatten() #adjust this without product id
    print(query_embedding)
    print(f"this is query_embedding: {query_embedding}")

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

# main_functionallity
def main_results(product):
    collection = chroma_client.get_collection(name="ingredients_embed_v2")
    embed_ex= embedding_ingredients(product, True)
    metadata_ex= create_metadata_dictionairy(product)
    results= query_chromadb_ingredients(collection, embed_ex, 5, where=metadata_ex[0])
    return results



if __name__ == "__main__":
    df= pd.read_csv("/home/marili/code/marilifeilzer/dupes/raw_data/products_clean_ingredients_rank_2.csv")
    df= df.dropna(subset=["formula"], axis=0)

    # product_example = df.iloc[50]
    # product_example= product_example.to_frame().T

    example2 = pd.DataFrame({
    "Unnamed: 0": [142],
    "product_id": [8734521901],
    "product_name": ["HydraGlow Nutritive Shampoo"],
    "manufacturer_name": ["BellezaVital Labs"],
    "price_eur": [12.99],
    "volume_ml": [250],
    "color_de_cabello": ["todos_los_colores_de_cabello"],
    "tipo_de_cabello": ['Todo tipo de cabello'],
    "propiedad": ['Detergente'],
    "description": ["Un champú nutritivo diseñado para hidratar y suavizar el cabello seco y rizado."],
    "ingredients_text": ["Aqua, Glycerin"],
    "ingredients_raw": [[
        "aqua",
        "glycerin"
    ]],
    "formula": [['H2O', 'C10H14N2Na2O8', 'C19H38N2O3', 'PPG-5-Ceteth-20', 'C41H80O17', 'C7H5NaO2', 'C8H10O2', 'C6H8O7', 'C16H32O6', 'C10H18O', 'Na4EDTA', 'C9H6O2', 'C10H16', 'C10H20O', 'polyquaternium-7', 'C29H50O2']]
})

    results = main_results(example2)
    print(results)


    # embeddings= embedding_ingredients(df, False)
    # metadata_dict= create_metadata_dictionairy(df)
    # collection= embedding_ingredients_populate_chromadb(df, embeddings, metadata_dict)
    # embed_ex= embedding_ingredients(example2, True)
    # metadata_ex= create_metadata_dictionairy(example2)

    # # res= query_chromadb_ingredients(collection, embeddings.iloc[50, 1:].astype(int).values, 5, where=metadata_dict[0])
    # res= query_chromadb_ingredients(collection, embed_ex, 5, where=metadata_ex[0])
    # print(res)
    # breakpoint()
