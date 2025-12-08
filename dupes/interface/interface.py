import streamlit as st
import requests
from streamlit_card import card



st.markdown("""
            # Welcome to Dupes!

            We are looking foward to help you find the best shampoo

            """)

# User Input

#shampoo_brand_text = st.text_input('You can either tell us the name of the shampoo you need:', placeholder="L'Oreal")

nlp_text = st.text_input("Tell us what kind of shampoo you are looking for:",\
    placeholder=" I need a moisturuzing shampoo for...")


if nlp_text:
    params = dict(description=nlp_text)

    # TODO: Change the api URL to google after the test in local

    dupes_web_api = "http://127.0.0.1:8000/recomend"
    response = requests.get(dupes_web_api,params=params)

    predictions = response.json()

    st.markdown("""
                ## We think these shampoos will be perfect for what you are looking for:
                """)

    for prediction in predictions:

        card(
            title=f"{list(prediction["product_name"].values())[0]}",
            text=[f"{list(prediction["description"].values())[0]}",\
                f"Price: €{list(prediction["price_eur"].values())[0]}",\
                    f"Predicted price: €"],
            styles={
                "card": {
                    "width": "500px",
                    "height": "500px",
                    "border-radius": "60px",
                    "box-shadow": "0 0 10px rgba(0,0,0,0.5)"
                },
                "text" : {
                    "color": "black"
                },
                "filter": {
                    "background-color": "white"
                }
                }
        )
