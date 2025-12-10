import streamlit as st
import pandas as pd
import requests

# df = pd.read_csv("/Users/jonamoram/code/marilifeilzer/dupes/raw_data/data_0812.csv")

# shampoos_options = df[["product_id","product_name"]]


# shampoo_input = st.selectbox(label="Name of the shampoo: ",\
#     options=shampoos_options["product_name"], placeholder="type the name of the shampoo", index=None)

# shampoo_id = shampoos_options.loc[shampoos_options["product_name"] == shampoo_input, "product_id"].values[0]

# if shampoo_id:

#     params = dict(product_id=shampoo_id)

#     # TODO: Change the api URL to google after the test in local

#     dupes_web_api = "http://127.0.0.1:8000/recommend_dupe"
#     response = requests.get(dupes_web_api,params=params)

#     predictions = response.json()
