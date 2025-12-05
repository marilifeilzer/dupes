import streamlit as st
import requests


st.markdown(
    """
    # Welcome to Dupes!
    We know shampoos can be expensive, so let us help you find cheaper dupes.
    """
)

shampoo_brand_text = st.text_input(
    "You can either tell us the name of the shampoo you need:",
    placeholder="L'Oreal",
)

nlp_text = st.text_input(
    "Or give a description for the kind of shampoo you are looking for",
    placeholder="I need a moisturizing shampoo",
)

# TODO: Change the API URL after local testing
dupes_web_api = "http://127.0.0.1:8000/recomend"

if st.button("Find dupes"):
    if not shampoo_brand_text and not nlp_text:
        st.warning("Please enter a shampoo name or a description.")
    else:
        params = {
            "shampoo": shampoo_brand_text,
            "description": nlp_text,
            "top_k": 5,
        }
        try:
            with st.spinner("Searching for dupes..."):
                response = requests.get(dupes_web_api, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:  # minimal error handling for prototype
            st.error(f"Could not fetch recommendations: {exc}")
            data = {}

        recs = data.get("recomendation") or []
        if recs:
            st.subheader("We recommend these dupes:")
            for rec in recs:
                # Support both dict and plain string for safety
                if isinstance(rec, dict):
                    name = rec.get("product_name") or "Unknown product"
                    price = rec.get("price_eur")
                    suffix = f" - €{price:.2f}" if isinstance(price, (int, float)) else ""
                    st.write(f"- {name}{suffix}")
                else:
                    st.write(f"- {rec}")
        else:
            st.info("No recommendations found for that input.")
