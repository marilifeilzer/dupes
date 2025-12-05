import streamlit as st
import requests


st.markdown("""
            # Welcome to Dupes!
            We know shampoos can be expesive , so please
            let us help you find the cheapest shampoo in the market.
            """)

# User Input

#shampoo_brand_text = st.text_input('You can either tell us the name of the shampoo you need:', placeholder="L'Oreal")

nlp_text = st.text_input("Tell me what kind of shampoo you are looking for:",\
    placeholder=" I need a moisturuzing shampoo for...")


if nlp_text:
    params = dict(description=nlp_text)

    # TODO: Change the api URL to google after the test in local

    dupes_web_api = "http://127.0.0.1:8000/recomend"
    response = requests.get(dupes_web_api,params=params)

    predictions = response.json()

    for prediction in predictions:

        st.text(f"Name of the shampoo {list(prediction["product_name"].values())[0]}")
        st.text(f"Price of the shampoo {list(prediction["price_eur"].values())[0]}")
        st.text(f"Description of the shampoo {list(prediction["description"].values())[0]}")
