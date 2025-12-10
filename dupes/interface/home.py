import streamlit as st

st.title("""
            WELCOME TO DUPES

            """, text_alignment= 'center')

st.caption("""## This site is here to help you find cheaper alternatives to your favorite products. """, width="stretch", text_alignment="center")
st.caption("""## Don’t have a favorite yet? No worries. Just tell us what you’re looking for in a shampoo, and we’ll help you find the product that best matches your description.""", width="stretch", text_alignment="center")


if st.button('DUPE FINDER', width= 'stretch'):
    st.switch_page("shampoo.py")

if st.button('RECOMMENDATIONS', width= 'stretch'):
    st.switch_page("product_description.py")
