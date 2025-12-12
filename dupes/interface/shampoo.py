import streamlit as st
import pandas as pd
import requests
from dupes.data.gc_client import load_table_to_df

df = load_table_to_df()

shampoos_options = df[["product_id","product_name", "price_eur", "volume_ml"]]

st.header(body="Product name", text_alignment="center")

shampoo_input = st.selectbox(label="Product name",\
    options=shampoos_options["product_name"], placeholder="Product name",\
        label_visibility="collapsed", index=None)

st.text(body="Type the product name into the search bar and choose the option that matches what you're looking for.")

if shampoo_input:

    shampoo_id = shampoos_options.loc[shampoos_options["product_name"] == shampoo_input, "product_id"].values[0]
    shampoo_price= shampoos_options.loc[shampoos_options["product_name"] == shampoo_input, "price_eur"].values[0]
    shampoo_volume = shampoos_options.loc[shampoos_options["product_name"] == shampoo_input, "volume_ml"].values[0]

    params = dict(product_id=shampoo_id)

    # TODO: Change the api URL to google after the test in local

    dupes_web_api = "https://dupes-img-pub-622586200055.europe-west1.run.app/dupe_with_price"
    response = requests.get(dupes_web_api,params=params)

    predictions = response.json()


    for prediction in predictions['predictions']:

        if prediction["price_eur"]/prediction["volume_ml"]< shampoo_price/shampoo_volume:

            with st.container(border= True):
                st.image(f"img/images/{prediction["product_id"]}.jpg")
                st.title(f"{prediction["product_name"]}")
                #st.caption(f"{prediction['en_description']}")
                st.caption(f"Price of {shampoo_input} is €**{shampoo_price}** per {shampoo_volume} ml")
                st.caption(f"The retail price of your dupe: €{prediction["price_eur"]} per {prediction["volume_ml"]} ml")
                st.caption(f"The price we think is fair: €{round(prediction["price_prediction"],2)} per {prediction["volume_ml"]} ml.")
                st.caption(f"The money you will save: €{shampoo_price - prediction["price_eur"]}")
                st.caption(f"## This is a DUPE")
        else:
            with st.container(border= True):
                st.image(f"img/images/{prediction["product_id"]}.jpg")
                st.title(f"{prediction["product_name"]}")
                #st.caption(f"{prediction['en_description']}")
                st.caption(f"The retail price of {prediction["product_name"]}: €{prediction["price_eur"]} per {prediction["volume_ml"]} ml")
                st.caption(f"The price we think is fair: €{round(prediction["price_prediction"],2)} per {prediction["volume_ml"]} ml.")
                st.caption(f"## Not a DUPE")
