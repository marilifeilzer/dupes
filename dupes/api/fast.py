from fastapi import FastAPI
import pandas as pd
from dupes.logic import predict_shampoo
from dupes.model.descriptions_chromadb import embedding_description_query_chromadb, embedding_description_get_recommendation
from dupes.model.model_chromadb import main_results
from dupes.model.optimiser import load_model
from dupes.model.price_prediction import preprocess_prediction_input
from dupes.model.model_chromadb import main_results, main_res_product_id

app = FastAPI()
app.state.model = load_model()

@app.get("/predict_price")
def get_price_prediction(volume_ml: int  = 350.0,
                         ingredients_raw: str = "Water, Cetearyl Alcohol, PPG-3 Benzyl Ether Myristate, Caprylic/Capric Triglyceride, Cetyl Alcohol,Octyldodecyl Ricinoleate, Quaternium-91, Cetrimonium Chloride, Divinyldimethicone/Dimethicone Copolymer, Behentrimonium Chloride, Glycerin, Cetyl Esters, Isododecane, Bis-Aminopropyl Diglycol Dimaleate, Fragrance, Panthenol, Phospholipids, Dimethicone PEG-7 Isostearate, Pseudozyma Epicola/Argania Spinosa Kernel Oil Ferment Filtrate, Pseudozyma Epicola/Camellia Sinensis Seed Oil Ferment Extract Filtrate, Tocopheryl Linoleate/Oleate, Quaternium-95, Propanediol, Punica Granatum Extract, Morinda Citrifolia Fruit Extract, PEG-8, Euterpe Oleracea Fruit Extract, Camellia Sinensis Seed Oil, Crambe Abyssinica Seed Oil, Hydroxypropyl Cyclodextrin, Persea Gratissima (Avocado) Oil, Vitis Vinifera (Grape) Seed Oil, Disodium EDTA, Polysilicone-15, C11-15 Pareth-7, Hydroxypropyl Guar, Glycine Soja (Soybean) Oil, PEG-45M, PEG-7 Amodimethicone, Amodimethicone, C12-13 Pareth-23, C12-13 Pareth-3, Laureth-9, Pentaerythrityl Tetra-Di-T-Butyl Hydroxyhydrocinnamate, PEG-4, Phenoxyethanol, Hexyl Cinnamal"
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


# embedding_description_get_recommendation()
# df = pd.read_csv("/Users/lewagon/code/marilifeilzer/dupes/raw_data/products_data__0412.csv")

# @app.get("/")
# def index():
#     return {"working": True}

# @app.get("/recommend")
# def get_recommendation(description: str):

#     recommendation = embedding_description_query_chromadb(description)

#     return recommendation

# df_cleaned= pd.read_csv('/home/panamas/code/marili/dupes/raw_data/products_cleaned.csv')
# dropped =  df_cleaned.dropna(subset=["formula"], axis=0)

# @app.get("/recommend_ingredients")
# def get_recommendation_ingredients(
#     # product_id: str,
#     formula: str = "H2O', 'C10H14N2Na2O8', 'C19H38N2O3', 'PPG-5-Ceteth-20', 'C41H80O17', 'C7H5NaO2', 'C8H10O2', 'C6H8O7', 'C16H32O6', 'C10H18O', 'Na4EDTA', 'C9H6O2', 'C10H16', 'C10H20O', 'polyquaternium-7', 'C29H50O2'",
#     color_de_cabello: str = "todos_los_colores_de_cabello",
#     tipo_de_cabello: str = "Todo tipo de cabello",
#     propiedad: str = "Detergente" ,
# ):

#     product = pd.DataFrame({
#         # "product_id": [product_id],
#         "formula": [formula],
#         "color_de_cabello": [color_de_cabello],
#         "tipo_de_cabello": [tipo_de_cabello],
#         "propiedad": [propiedad]
#     })

#     cols = ['formula', 'color_de_cabello', 'tipo_de_cabello', 'propiedad']

#     for col in cols:
#         product[col] = product[col].apply(
#             lambda x: x if isinstance(x, list) else x.split(',')
#     )

#     results = main_results(product)
#     product_ids= results['ids'][0]

#     product_names = [df_cleaned.
#                      loc[df_cleaned["product_id"]==product, ["product_name","price_eur", "description"]]for product in product_ids]

#     return product_names


# @app.get("/recommend_dupe")
# def get_recommendation_ingredients(
#     product_id: str
# ):

#     results= main_res_product_id(product_id, dropped)

#     product_ids= results['ids'][0]

#     df = dropped.loc[dropped["product_id"].isin(product_ids), ["product_name","price_eur", "description"]]

#     return {"prodcut_names":df.fillna("No data").to_dict(orient="records")}
