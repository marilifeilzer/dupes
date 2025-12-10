import streamlit as st
import requests
from streamlit_card import card

pages = {
    "HOME":
        [st.Page("interface.py", title="Go to homepage")],
    "DUPE FINDER":
        [st.Page("shampoo_ui.py", title="Find your dupe")],
    "RECOMMENDATION": [
      st.Page("product_description.py", title="Find your recommendation")
     ]
}

pg = st.navigation(pages, position="top")
pg.run()

st.markdown("""
            # Welcome to Dupes! 👋🏻👋🏻

            ## Our mission is simple: help you find the right shampoo for your needs at a price you’ll love

            """)

