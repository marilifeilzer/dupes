import streamlit as st
import requests


st.markdown("""
            # Welcome to Dupes!
            We know shampoos can be expesive , so please
            let us help you find the cheapest shampoo in the market.
            """)

# User Input 
    
shampoo_brand_text = st.text_input('You can either tell us the name of the shampoo you need:', placeholder="L'Oreal")

nlp_text = st.text_input("Or you can give me a description for the kind of shampoo you are looking for",placeholder=" I need a moisturuzing shampoo")

    

params = dict(
    shampoo=shampoo_brand_text,
    description=nlp_text
)

# TODO: Change the api URL to google after the test in local

dupes_web_api = "http://127.0.0.1:8000/recomend"
response = requests.get(dupes_web_api,params=params)

prediction = response.json()

predicted_shampoo = prediction["recomendation"]

# Check for a valid input to output the results

if shampoo_brand_text:
    
    st.header(f"The most similars shampoos to {shampoo_brand_text} we could find are: {predicted_shampoo}")

else:
    
    st.header(f"We recomend you the following shampoo: {predicted_shampoo}")
