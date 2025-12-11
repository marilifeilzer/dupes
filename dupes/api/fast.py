from fastapi import FastAPI
import pandas as pd

from dupes.model.descriptions_chromadb import (
    embedding_description_query_chromadb,
)
from dupes.model.price_prediction_oof import load_model_meta
from dupes.model.optimiser import load_model_base
from dupes.model.price_prediction import preprocess_prediction_input
from dupes.model.model_chromadb import main_res_product_id
from dupes.data.gc_client import load_table_to_df

app = FastAPI()
app.state.model_meta = load_model_meta(manufacturer=True)
app.state.model_base = load_model_base(manufacturer=True)

df = load_table_to_df()


@app.get("/")
def index():
    return {"working": True}


@app.get("/predict_price")
def get_price_prediction(
    volume_ml: int = 450,
    manufacturer_name: str = "Beautyge",
    formula: str = "['H2O', 'Cocamidopropyl hydroxysultaine', 'C12H21Na2O7S', 'C19H38N2O3', 'C15H28NNaO3', 'Acrylates copolymer', 'C14H27NaO5S', 'C16H32O6', 'Cocoamphodiacetate, disodium salt', 'C12H20O7', 'C8H10O', 'C21H40O4', 'C3H8O3', 'C6H8O7', 'C9H19NO4', 'NaOH', 'NaCl', 'C9H9NNa4O8', 'C3H8O2', 'C7H5NaO2', 'C10H21ClN2O2', 'C6H7KO2', 'C8H10O2', 'C30H62', 'C10H18O', 'C10H16']",
):

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

    return {"prediction": round(meta_pred_price, 2)}


@app.get("/recommend_with_price")
def get_recommendation(description: str):

    price_model = app.state.model_base
    breakpoint()

    recommendation = embedding_description_query_chromadb(description)
    if len(recommendation) > 0:
        df_concat = pd.concat(recommendation)
        product_names = df_concat.product_name.values

        # Get prediction data with product names for alignment
        predict_price_df = df.loc[df.product_name.isin(product_names)][
            ["product_name", "volume_ml", "formula", "manufacturer_name"]
        ].copy()
        predict_price_df["volume_ml"] = predict_price_df["volume_ml"].astype(float)

        # Store original data before preprocessing (which may filter some rows)
        original_predict_df = predict_price_df.copy()

        preproc_input = predict_price_df.drop(columns=["product_name"])
        preproc = preprocess_prediction_input(preproc_input, manufacturer=True)

        if len(preproc) != len(original_predict_df):
            valid_mask = preproc_input.notna().all(axis=1)
            predict_price_df = original_predict_df[valid_mask].copy()

        pred_price_ml = price_model.predict(preproc).tolist()

        predict_price_df["ml_prediction"] = pred_price_ml

        df_result = df_concat.merge(
            predict_price_df[["product_name", "ml_prediction"]],
            on="product_name",
            how="left",
        )
        df_result["price_prediction"] = (
            df_result["ml_prediction"] * df_result["volume_ml"]
        )

        df_result = df_result.replace([float("inf"), float("-inf")], None)
        df_result = df_result.where(pd.notna(df_result), None)
        df_result = df_result.dropna()

        return {"prediction": df_result.to_dict(orient="records")}

    return recommendation


@app.get("/dupe_with_price")
def get_dupe_with_price(product_id: str = "3001044443"):

    price_model = app.state.model_base

    df = load_table_to_df()

    dropped = df.dropna(subset=["formula"], axis=0)
    results = main_res_product_id(product_id, dropped)

    product_ids = results["ids"][0][1:]

    duplicate_products = df.loc[df["product_id"].isin(product_ids)][
        [
            "product_id",
            "product_name",
            "volume_ml",
            "formula",
            "price_eur",
            "manufacturer_name",
        ]
    ].copy()

    complete_data_products = duplicate_products.dropna(
        subset=["volume_ml", "formula", "price_eur", "manufacturer_name"]
    )

    if len(complete_data_products) == 0:
        return {
            "message": "Found duplicate products but none have complete data for price prediction",
            "duplicate_products": duplicate_products[
                ["product_id", "product_name"]
            ].to_dict(orient="records"),
            "predictions": [],
        }

    predict_price_df = complete_data_products[
        ["volume_ml", "formula", "price_eur", "manufacturer_name"]
    ].copy()
    predict_price_df["volume_ml"] = predict_price_df["volume_ml"].astype(float)

    preproc = preprocess_prediction_input(predict_price_df, manufacturer=True)

    if "price_eur" in preproc.columns:
        preproc = preproc.drop(columns=["price_eur"])
    pred_price_ml = price_model.predict(preproc).tolist()
    complete_data_products["ml_prediction"] = pred_price_ml
    complete_data_products["price_prediction"] = (
        complete_data_products["ml_prediction"] * complete_data_products["volume_ml"]
    )

    # Clean data for JSON serialization - replace NaN/inf with None
    complete_data_products = complete_data_products.replace(
        [float("inf"), float("-inf")], None
    )
    complete_data_products = complete_data_products.where(
        pd.notna(complete_data_products), None
    )

    return {
        "predictions": complete_data_products.to_dict(orient="records"),
        "total_duplicates_found": len(duplicate_products),
        "predictions_made": len(complete_data_products),
    }
