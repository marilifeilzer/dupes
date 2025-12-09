import streamlit as st
import pandas as pd
import requests

# ------------ PAGE SETUP ------------
st.set_page_config(page_title="Dupe Finder", layout="wide")

# ------------ CUSTOM CSS ------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Montserrat', sans-serif;
}

/* Header bar */
.header {
    background-color: #d7d3ff;
    padding: 20px 60px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

/* Left icons */
.nav-left {
    display: flex;
    gap: 80px;
    align-items: center;
}

.nav-item {
    text-align: center;
    color: #726fa4;
    font-size: 16px;
}

.nav-item img {
    width: 38px;
    opacity: 0.7;
}

.nav-item:hover img {
    opacity: 1.0;
    transform: scale(1.05);
}

/* Page title */
.page-title {
    font-size: 70px;
    font-weight: 600;
    color: #a29ad8;
    letter-spacing: 6px;
}

/* Centered section */
.center {
    text-align: center;
    margin-top: 80px;
}

.search-title {
    font-size: 32px;
    font-weight: 600;
    color: #7a75a8;
}

/* Search bar container */
.search-box {
    width: 60%;
    margin: 30px auto;
    display: flex;
    border: 3px solid #6e6e6e;
    border-radius: 3px;
    height: 60px;
    align-items: center;
    padding-left: 15px;
}

.search-input input {
    border: none !important;
    outline: none !important;
    font-size: 18px;
    width: 100%;
    color: #7a75a8;
}

.search-icon {
    margin-right: 15px;
}

.search-description {
    margin-top: 20px;
    font-size: 20px;
    color: #7a7a7a;
}

</style>
""", unsafe_allow_html=True)


# ------------ HEADER BAR ------------
st.markdown("""
<div class="header">
    <div class="nav-left">
        <div class="nav-item" onclick="window.location.href='/'">
            <img src="https://img.icons8.com/ios-filled/100/766FA8/home.png"/>
            <div>home</div>
        </div>

        <div class="nav-item" onclick="window.location.href='?page=recommendation'">
            <img src="https://img.icons8.com/ios/100/766FA8/search--v1.png"/>
            <div>recommendation</div>
        </div>
    </div>

    <div class="page-title">DUPE FINDER</div>
</div>
""", unsafe_allow_html=True)


# ------------ SEARCH SECTION ------------
st.markdown("""
<div class="center">
    <div class="search-title">Product name</div>

    <div class="search-box">
        <div class="search-input">
            <input id="product_input" type="text" placeholder="Product name"/>
        </div>
        <div class="search-icon">
            <img src="https://img.icons8.com/ios-glyphs/50/6e6e6e/search--v1.png" width="30"/>
        </div>
    </div>

    <div class="search-description">
        Type the product name into the search bar and choose the option that matches what you're looking for.
    </div>
</div>
""", unsafe_allow_html=True)


df = pd.read_csv("/Users/jonamoram/code/marilifeilzer/dupes/raw_data/data_0812.csv")

shampoos_options = df[["product_id","product_name"]]


shampoo_input = st.selectbox(label="Name of the shampoo: ",\
    options=shampoos_options["product_name"], placeholder="type the name of the shampoo", index=None)

shampoo_id = shampoos_options.loc[shampoos_options["product_name"] == shampoo_input, "product_id"].values[0]

if shampoo_id:

    params = dict(product_id=shampoo_id)

    # TODO: Change the api URL to google after the test in local

    dupes_web_api = "http://127.0.0.1:8000/recommend_dupe"
    response = requests.get(dupes_web_api,params=params)

    predictions = response.json()

# ------------ STREAMLIT BACKEND HANDLING ------------
product = st.text_input("", label_visibility="collapsed")

if product:
    st.write(f"🔍 You searched for **{product}**")
    # >>> Replace with your API call <<<
