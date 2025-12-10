import streamlit as st

st.title("""
            WELCOME TO DUPES

            """, text_alignment= 'center')

st.caption("""## This site is here to help you find cheaper alternatives to your favorite products. """, width="stretch", text_alignment="center")
st.caption("""## Don’t have a favorite yet? No worries. Just tell us what you’re looking for in a shampoo, and we’ll help you find the product that best matches your description.""", width="stretch", text_alignment="center")


st.button('DUPE FINDER', width= 'stretch')
st.button('RECOMMENDATIONS', width= 'stretch')
