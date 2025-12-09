from fastapi import FastAPI
import pandas as pd
from dupes.logic import predict_shampoo
from dupes.model.descriptions_chromadb import embedding_description_query_chromadb, embedding_description_get_recommendation
from dupes.model.model_chromadb import main_results
from dupes.model.optimiser import load_model
from dupes.model.price_prediction import preprocess_prediction_input

app = FastAPI()
app.state.model = load_model()

@app.get("/predict_price")
def get_price_prediction(volume_ml: int  = 350.0,
                         propiedad: str = 'Detergente, Anti-rotura capilar, Nutritivo, Protector, Reparador, Refrescante, Espuma',
                         ingredients_raw: str = "Water, Cetearyl Alcohol, PPG-3 Benzyl Ether Myristate, Caprylic/Capric Triglyceride, Cetyl Alcohol,Octyldodecyl Ricinoleate, Quaternium-91, Cetrimonium Chloride, Divinyldimethicone/Dimethicone Copolymer, Behentrimonium Chloride, Glycerin, Cetyl Esters, Isododecane, Bis-Aminopropyl Diglycol Dimaleate, Fragrance, Panthenol, Phospholipids, Dimethicone PEG-7 Isostearate, Pseudozyma Epicola/Argania Spinosa Kernel Oil Ferment Filtrate, Pseudozyma Epicola/Camellia Sinensis Seed Oil Ferment Extract Filtrate, Tocopheryl Linoleate/Oleate, Quaternium-95, Propanediol, Punica Granatum Extract, Morinda Citrifolia Fruit Extract, PEG-8, Euterpe Oleracea Fruit Extract, Camellia Sinensis Seed Oil, Crambe Abyssinica Seed Oil, Hydroxypropyl Cyclodextrin, Persea Gratissima (Avocado) Oil, Vitis Vinifera (Grape) Seed Oil, Disodium EDTA, Polysilicone-15, C11-15 Pareth-7, Hydroxypropyl Guar, Glycine Soja (Soybean) Oil, PEG-45M, PEG-7 Amodimethicone, Amodimethicone, C12-13 Pareth-23, C12-13 Pareth-3, Laureth-9, Pentaerythrityl Tetra-Di-T-Butyl Hydroxyhydrocinnamate, PEG-4, Phenoxyethanol, Hexyl Cinnamal.",
                         manufacturer_name: str = 'Obelis S.A'
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

@app.get("/")
def index():
    return {"working": True}

@app.get("/recommend")
def get_recommendation(description: str):

    recommendation = embedding_description_query_chromadb(description)

    return recommendation

df_cleaned= pd.read_csv('raw_data/data_0812.csv')

@app.get("/recommend_ingredients")
def get_recommendation_ingredients(
    # product_id: str,
    formula: str,
    color_de_cabello: str,
    tipo_de_cabello: str,
    propiedad: str,
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

    print(product)
    print(type(product))

    results = main_results(product)
    product_ids= results['ids'][0]

    product_names = [df_cleaned.
                     loc[df_cleaned["product_id"]==product, ["product_name","price_eur", "description"]]for product in product_ids]

    return product_names
