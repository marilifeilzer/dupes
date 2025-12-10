import streamlit as st
import requests
from streamlit_card import card

pages = {
    "HOME":
        [st.Page("home.py", title="Go to homepage")],
    "DUPE FINDER":
        [st.Page("shampoo.py", title="Find your dupe")],
    "RECOMMENDATION": [
      st.Page("product_description.py", title="Find your recommendation")
     ]
}

pg = st.navigation(pages, position="top")
pg.run()
