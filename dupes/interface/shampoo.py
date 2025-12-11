import streamlit as st
import pandas as pd
import requests
from dupes.data.gc_client import load_table_to_df

df = load_table_to_df()

shampoos_options = df[["product_id","product_name"]]

st.header(body="Product name", text_alignment="center")

shampoo_input = st.selectbox(label="Product name",\
    options=shampoos_options["product_name"], placeholder="Product name",\
        label_visibility="collapsed", index=None)

st.text(body="Type the product name into the search bar and choose the option that matches what you're looking for.")

if shampoo_input:

    shampoo_id = shampoos_options.loc[shampoos_options["product_name"] == shampoo_input, "product_id"].values[0]

    params = dict(product_id=shampoo_id)

    # TODO: Change the api URL to google after the test in local

    dupes_web_api = "http://127.0.0.1:8000/dupe_with_price"
    response = requests.get(dupes_web_api,params=params)

    predictions = response.json()


    for prediction in predictions['predictions']:

        with st.container(border= True):
            st.title(f"{prediction["product_name"]}")
            st.caption(f"Actual price in stores: €{prediction["price_eur"]}")
            st.caption(f"The price we think is fair: €{round(prediction["price_prediction"],2)} per {prediction["volume_ml"]} ml.")
