from fastapi import FastAPI
import pandas as pd

from dupes.logic import predict_shampoo
from dupes.model.descriptions_chromadb import embedding_description_query_chromadb, embedding_description_get_recommendation
from dupes.model.price_prediction_oof import load_model_meta
from dupes.model.optimiser import load_model_base
from dupes.model.price_prediction import preprocess_prediction_input
from dupes.model.model_chromadb import main_results, main_res_product_id
from dupes.data.gc_client import load_table_to_df

app = FastAPI()
app.state.model_meta = load_model_meta(manufacturer=True)
app.state.model_base = load_model_base(manufacturer=True)

df= load_table_to_df()

@app.get("/predict_price")
def get_price_prediction(volume_ml: int  = 450,
                         manufacturer_name: str = 'Beautyge',
                         formula: str = "['H2O', 'Cocamidopropyl hydroxysultaine', 'C12H21Na2O7S', 'C19H38N2O3', 'C15H28NNaO3', 'Acrylates copolymer', 'C14H27NaO5S', 'C16H32O6', 'Cocoamphodiacetate, disodium salt', 'C12H20O7', 'C8H10O', 'C21H40O4', 'C3H8O3', 'C6H8O7', 'C9H19NO4', 'NaOH', 'NaCl', 'C9H9NNa4O8', 'C3H8O2', 'C7H5NaO2', 'C10H21ClN2O2', 'C6H7KO2', 'C8H10O2', 'C30H62', 'C10H18O', 'C10H16']"
                         ):

    # Create dataframe with the input variables for the prediction
    input = pd.DataFrame(locals(), index=[0])

    # Preprocess it the same way as our training data
    preproc = preprocess_prediction_input(input, manufacturer=True)

    # Load the fitted base model
    base_model = app.state.model_base

    # Make the prediction
    base_pred_price_ml = base_model.predict(preproc)

    # Load the fitted meta model
    meta_model = app.state.model_meta

    # Make the prediction
    meta_pred_price_ml = meta_model.predict(base_pred_price_ml).tolist()
    meta_pred_price = meta_pred_price_ml[0] * volume_ml

    return {'prediction': round(meta_pred_price, 2)}


@app.get("/")
def index():
    return {"working": True}

@app.get("/recommend")
def get_recommendation(description: str):

    recommendation = embedding_description_query_chromadb(description)


    return recommendation


@app.get("/recommend_with_price")
def get_recommendation(description: str):
    price_model = app.state.model


    recommendation = embedding_description_query_chromadb(description)
    if len(recommendation) > 0:
        df_concat = pd.concat(recommendation)
        product_names = df_concat.product_name.values
        predict_price_df = df.loc[df.product_name.isin(product_names)][["volume_ml", "formula"]]
        predict_price_df["volume_ml"] =  predict_price_df["volume_ml"].astype(float)

        preproc = preprocess_prediction_input(predict_price_df)
        pred_price_ml = price_model.predict(preproc).tolist()
        df_concat["ml_prediction"] = pred_price_ml
        df_concat["price_prediction"] = df_concat["ml_prediction"] * df_concat["volume_ml"]
        return {"prediction": df_concat.to_dict(orient="records")}


    return recommendation

@app.get("/dupe_with_price")
def get_dupe_with_price(product_id: str):
    
    price_model = app.state.model

    df= load_table_to_df()

    dropped =  df.dropna(subset=["formula"], axis=0)
    results= main_res_product_id(product_id, dropped)

    product_ids= results['ids'][0][1:]

    product_names = [df.loc[df["product_id"]==product, ["product_name","price_eur", "description", "formula", "volume_ml"]] for product in product_ids]


    predict_price_df = df.loc[df.product_name.isin(product_names)][["volume_ml", "formula"]]
    predict_price_df["volume_ml"] =  predict_price_df["volume_ml"].astype(float)

    preproc = preprocess_prediction_input(predict_price_df)
    pred_price_ml = price_model.predict(preproc).tolist()
    predict_price_df["ml_prediction"] = pred_price_ml
    predict_price_df["price_prediction"] = predict_price_df["ml_prediction"] * predict_price_df["volume_ml"]
    return {"prediction": predict_price_df.to_dict(orient="records")}




@app.get("/recommend_ingredients")
def get_recommendation_ingredients(
    # product_id: str,
    formula: str = "H2O', 'C10H14N2Na2O8', 'C19H38N2O3', 'PPG-5-Ceteth-20', 'C41H80O17', 'C7H5NaO2', 'C8H10O2', 'C6H8O7', 'C16H32O6', 'C10H18O', 'Na4EDTA', 'C9H6O2', 'C10H16', 'C10H20O', 'polyquaternium-7', 'C29H50O2'",
    color_de_cabello: str = "todos_los_colores_de_cabello",
    tipo_de_cabello: str = "Todo tipo de cabello",
    propiedad: str = "Detergente" ,
):
    df_cleaned= pd.read_csv('/Users/panamas/code/marili/dupes/raw_data/products_clean_600_ingredients.csv')
    dropped =  df_cleaned.dropna(subset=["formula"], axis=0)

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



    product_names = [df.
                     loc[df["product_id"]==product, ["product_name","price_eur", "description"]]for product in product_ids]

    return product_names


@app.get("/recommend_dupe")
def get_recommendation_ingredients(
    product_id: str
):
    df= load_table_to_df()


    dropped =  df.dropna(subset=["formula"], axis=0)
    results= main_res_product_id(product_id, dropped)

    product_ids= results['ids'][0][1:]


    #df = df.loc[dropped["product_id"].isin(product_ids), ["product_name","price_eur", "description"]]


    #return {"prodcut_names":dropped.fillna("No data").to_dict(orient="records")}

    results_df = df.loc[df["product_id"].isin(product_ids), ["product_name","price_eur", "en_description"]].to_dict(orient="records")

    return results_df
