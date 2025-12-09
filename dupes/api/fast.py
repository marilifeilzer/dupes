from fastapi import FastAPI
import pandas as pd
from dupes.logic import predict_shampoo
from dupes.model.descriptions_chromadb import embedding_description_query_chromadb, embedding_description_get_recommendation
from dupes.model.model_chromadb import main_results

app = FastAPI()

# embedding_description_get_recommendation()

# df = pd.read_csv("/Users/lewagon/code/marilifeilzer/dupes/raw_data/products_data__0412.csv")

@app.get("/")
def index():
    return {"working": True}

@app.get("/recomend")
def get_recomendation(description: str):


    recomendation = embedding_description_query_chromadb(description)


    return recomendation

df_cleaned= pd.read_csv('raw_data/data_0812.csv')

@app.get("/recomend_ingredients")
def get_recommendation_ingredients(
    # product_id: str,
    formula: str = "H2O', 'C10H14N2Na2O8', 'C19H38N2O3', 'PPG-5-Ceteth-20', 'C41H80O17', 'C7H5NaO2', 'C8H10O2', 'C6H8O7', 'C16H32O6', 'C10H18O', 'Na4EDTA', 'C9H6O2', 'C10H16', 'C10H20O', 'polyquaternium-7', 'C29H50O2'",
    color_de_cabello: str = "todos_los_colores_de_cabello",
    tipo_de_cabello: str = "Todo tipo de cabello",
    propiedad: str = "Detergente" ,
):

    product = pd.DataFrame({
        # "product_id": [product_id],
        "formula": [formula],
        "color_de_cabello": [color_de_cabello],
        "tipo_de_cabello": [tipo_de_cabello],
        "propiedad": [propiedad]
    })

    cols = ['formula', 'color_de_cabello', 'tipo_de_cabello', 'propiedad']

    for col in cols:
        product[col] = product[col].apply(
            lambda x: x if isinstance(x, list) else x.split(',')
    )

    results = main_results(product)
    product_ids= results['ids'][0]

    product_names = [df_cleaned.
                     loc[df_cleaned["product_id"]==product, ["product_name","price_eur", "description"]]for product in product_ids]

    return product_names
