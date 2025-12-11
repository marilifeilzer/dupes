import chromadb # , EmbeddingFunction
import pandas as pd
from dupes.data.properties import encode_properties, use_encoder_load
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from dupes.data.gc_client import load_table_to_df

# Common instances
model = SentenceTransformer("all-mpnet-base-v2")
chroma_client = chromadb.PersistentClient(path="raw_data/")

def embedding_ingredients_get_data(df: pd.DataFrame):
    dropped = df[['product_id', 'formula']]
    dropped = df.dropna(subset=['formula'], axis=0)
    return dropped

def embedding_ingredients(df: pd.DataFrame, exist=False):
    # Don't use eval() - the encode_properties function now handles parsing
    if exist==True:
        encoded = use_encoder_load(df, col = 'formula')
    else:
        encoded= encode_properties(df, col= ['formula'])



    return encoded

def create_metadata_dictionairy_properties(df: pd.DataFrame, cols=["tipo_de_cabello", "color_de_cabello", "propiedad"]):
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

def create_metadata_dictionairy(df: pd.DataFrame, cols=["tipo_de_cabello", "color_de_cabello", "propiedad"]):
    # dropped= df.dropna(subset=cols, axis=0)
    metadata_dict_encoded= []
    print(f"This is df imput of create_metadata_dict {df}")

    for col in cols:
        df[col]= df[col].apply(lambda x: x.split(',')) #changed dropped to df


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

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=filter
    )

    return results

# Main functionality


def create_ingr_db() -> None:
    df= load_table_to_df()

    dropped= embedding_ingredients_get_data(df)
    embed_ingredients = embedding_ingredients(dropped)
    metadata_dict= create_metadata_dictionairy(dropped)
    embedding_ingredients_populate_chromadb(dropped, embed_ingredients, metadata_dict)

# main_functionallity
def main_results(product):
    collection = chroma_client.get_collection(name="ingredients_embed_v2")
    embed_ex= embedding_ingredients(product, True)
    metadata_ex= create_metadata_dictionairy_properties(product)
    results= query_chromadb_ingredients(collection, embed_ex, 5, where=metadata_ex[0])
    return results

def main_res_product_id(product_id, df):
    collection = chroma_client.get_collection(name="ingredients_embed_v2")
    product = df.loc[df['product_id'] == product_id]
    embed_ex= embedding_ingredients(product, True)
    metadata_ex= create_metadata_dictionairy(product)
    results= query_chromadb_ingredients(collection, embed_ex, 5, where=metadata_ex[0])
    return results


if __name__ == "__main__":
    print("Building ChromaDB collection...")
    create_ingr_db()
    print("Done.")
