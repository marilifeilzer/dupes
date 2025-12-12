import streamlit as st
import pandas as pd
import requests
from google.oauth2 import service_account
from google.cloud import bigquery

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = bigquery.Client(credentials=credentials)

def load_table_to_df(
    dataset: str | None = None, table: str | None = None
) -> pd.DataFrame:

    table_id = "wagon-bootcamp-475013.dupes.Data_1012_bq"

    query = f"SELECT * FROM `{table_id}`"
    job = client.query(query)
    df = job.result().to_dataframe()

    return df

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



    dupes_web_api = "https://dupes-img-pub-622586200055.europe-west1.run.app/dupe_with_price"
    response = requests.get(dupes_web_api,params=params)

    predictions = response.json()


    for prediction in predictions['predictions']:

        with st.container(border= True):
            st.image(f"img/images/{prediction["product_id"]}.jpg")
            st.title(f"{prediction["product_name"]}")
            #st.caption(f"{prediction['en_description']}")
            st.caption(f"Actual price in stores: €{prediction["price_eur"]}")
            st.caption(f"The price we think is fair: €{round(prediction["price_prediction"],2)} per {prediction["volume_ml"]} ml.")
