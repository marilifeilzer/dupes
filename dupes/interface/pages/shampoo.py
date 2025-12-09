import streamlit as st
import pandas as pd

df = pd.read_csv("/Users/jonamoram/code/marilifeilzer/dupes/raw_data/data_0812.csv")

shampoos_options = df[["product_id","product_name"]]


shampoo_input = st.selectbox(label="Name of the shampoo: ",\
    options=shampoos_options["product_name"], placeholder="type the name of the shampoo", index=None)

shampoo_id = shampoos_options.loc[shampoos_options["product_name"] == shampoo_input, "product_id"].values[0]


