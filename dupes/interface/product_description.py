import streamlit as st
from streamlit_navigation_bar import st_navbar
from streamlit_card import card
import requests

import streamlit as st

with st.container():
    nlp_text = st.text_input("Tell us what hair goals you have and we’ll find affordable, high-quality shampoo matches.",\
    placeholder="frizz control, hydration, volume, repair?")


if nlp_text:
    params = dict(description=nlp_text)

    # TODO: Change the api URL to google after the test in local

    dupes_web_api = "http://127.0.0.1:8000/recomend"
    response = requests.get(dupes_web_api,params=params)

    predictions = response.json()

    st.markdown("""
                ## No ads. No sponsored brands. Just unbiased recommendations 💛
                """)

    for prediction in predictions:

        card(
            title=f"{list(prediction["product_name"].values())[0]}",
            text=[f"{list(prediction["description"].values())[0]}",\
                    f"Actual price in stores: €{list(prediction["price_eur"].values())[0]}",\
                        f"The price we think its fair: €"],
            styles={
                "card": {
                    "width": "100%",
                    "height": "500px",
                    "border-radius": "60px",
                    "box-shadow": "0 0 10px rgba(0,0,0,0.5)"
                },
                "text" : {
                    "color": "black"
                },
                "filter": {
                    "background-color": "white"
                },
                "title" : {
                    "color": "black"
                }
                }
        )
