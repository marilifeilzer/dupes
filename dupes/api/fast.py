from fastapi import FastAPI
import pandas as pd
from dupes.logic import predict_shampoo
from dupes.model.descriptions_chromadb import embedding_description_query_chromadb, embedding_description_get_recommendation
from dupes.model.model_chromadb import main_results
import ast
app = FastAPI()

embedding_description_get_recommendation()


@app.get("/")
def index():
    return {"working": True}

@app.get("/recomend")
def get_recomendation(description: str):


    recomendation = embedding_description_query_chromadb(description)


    return recomendation

df_cleaned= pd.read_csv('/home/marili/code/marilifeilzer/dupes/raw_data/data_0812.csv')

@app.get("/recomend_ingredients")
def get_recommendation_ingredients(
    product_id: str,
    formula: str,
    color_de_cabello: str,
    tipo_de_cabello: str,
    propiedad: str,
):

    product = pd.DataFrame({
        "product_id": [product_id],
        "formula": [formula],
        "color_de_cabello": [color_de_cabello],
        "tipo_de_cabello": [tipo_de_cabello],
        "propiedad": [propiedad]
    })

    cols = ['product_id', 'formula', 'color_de_cabello', 'tipo_de_cabello', 'propiedad']

    for col in cols:
        product[col] = product[col].apply(
            lambda x: x if isinstance(x, list) else x.split(',')
    )


    print(product)
    print(type(product))

    results = main_results(product)
    product_ids= results['ids'][0]

    product_names = [df_cleaned.
                     loc[df_cleaned["product_id"]==product, ["product_name","price_eur", "description"]]for product in product_ids]

    return product_names
