from fastapi import FastAPI
import pandas as pd

from dupes.logic import predict_shampoo
from dupes.model.descriptions_chromadb import embedding_description_query_chromadb, embedding_description_get_recommendation
from dupes.model.optimiser import load_model
from dupes.model.price_prediction import preprocess_prediction_input
from dupes.model.model_chromadb import main_results, main_res_product_id

app = FastAPI()
app.state.model = load_model()

@app.get("/predict_price")
def get_price_prediction(volume_ml: int  = 236.0,
                         formula: str = "['H2O', 'C14H27NaO5S', 'Cocamidopropyl hydroxysultaine', 'Cocoyl methyl taurate sodium salt', 'C16H32O6', 'C16H34O or C18H38O', 'C11H22O4', 'C10H20O2', 'C16H34O', 'C34H68O2', 'C', 'C5H11NO2', 'C17H33NO4Na', 'C9H9NNa4O8', 'C16H14N2O3', 'C8H7NaO3S', 'C21H42O6', 'C21H42O4', 'C38H74O4', 'C18H36O2', 'C21H45KO4P', 'C3H8O3', 'C58H118O21', 'C18H37COO(PEG)75', 'C16H34O2', 'C18H37(OCH2CH2)20OH', 'C10-30 Alkyl Acrylate Crosspolymer', 'NaOH', 'C3H8O2', 'C11H24O3', 'C8H8O2', 'C10H18O', 'C15H20O2', 'C10H20O', 'C10H16']"
                         ):

    # Create dataframe with the input variables for the prediction
    input = pd.DataFrame(locals(), index=[0])

    # Preprocess it the same way as our training data
    preproc = preprocess_prediction_input(input)

    # Load the fitted model
    model = app.state.model

    # Make the prediction
    pred_price_ml = model.predict(preproc).tolist()
    pred_price = pred_price_ml[0] * volume_ml

    return {'prediction': round(pred_price, 2)}


embedding_description_get_recommendation()
df = pd.read_csv("/Users/panamas/code/marili/dupes/raw_data/products_data_1012.csv")

@app.get("/")
def index():
    return {"working": True}

@app.get("/recommend")
def get_recommendation(description: str):

    recommendation = embedding_description_query_chromadb(description)

    return recommendation



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

    product_names = [df_cleaned.
                     loc[df_cleaned["product_id"]==product, ["product_name","price_eur", "description"]]for product in product_ids]

    return product_names


@app.get("/recommend_dupe")
def get_recommendation_ingredients(
    product_id: str
):
    df_cleaned= pd.read_csv('/Users/panamas/code/marili/dupes/raw_data/products_clean_600_ingredients.csv')
    dropped =  df_cleaned.dropna(subset=["formula"], axis=0)

    results= main_res_product_id(product_id, dropped)

    product_ids= results['ids'][0]

    df = dropped.loc[dropped["product_id"].isin(product_ids), ["product_name","price_eur", "description"]]

    return {"prodcut_names":df.fillna("No data").to_dict(orient="records")}
