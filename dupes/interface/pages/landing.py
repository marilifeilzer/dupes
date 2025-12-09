import streamlit as st

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="Dupes | Shampoos", layout="wide")

# ---------- CUSTOM CSS FOR STYLING ----------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Montserrat', sans-serif;
}

/* Top banner */
.header {
    background-color: #d7d3ff;
    padding: 40px 0 20px 0;
    text-align: center;
}

.title-left {
    float: left;
    font-size: 90px;
    font-weight: 700;
    color: #706aa7;
    margin-left: 60px;
}

.title-right {
    float: right;
    font-size: 80px;
    font-weight: 300;
    color: #b3afd8;
    margin-right: 60px;
}

/* Welcome text */
.welcome {
    text-align: center;
    font-size: 28px;
    font-weight: 600;
    color: #726fa4;
    margin-top: 50px;
}

.description {
    text-align: center;
    font-size: 20px;
    color: #999999;
    width: 70%;
    margin: auto;
}

/* Circular buttons */
.circle-container {
    display: flex;
    justify-content: center;
    gap: 150px;
    margin-top: 60px;
}

.circle-btn {
    width: 250px;
    height: 250px;
    background-color: #7d78ab;
    border-radius: 50%;
    display: flex;
    justify-content: center;
    align-items: center;
    text-align: center;
    color: white;
    font-size: 22px;
    font-weight: 600;
    cursor: pointer;
    transition: 0.3s;
}

.circle-btn:hover {
    background-color: #6b6597;
}

</style>
""", unsafe_allow_html=True)


# ---------- UI LAYOUT ----------
# HEADER BAR
st.markdown("""
<div class="header">
    <div class="title-left">DUPES</div>
    <div class="title-right">shampoos</div>
    <div style="clear: both;"></div>
</div>
""", unsafe_allow_html=True)

# WELCOME TEXT
st.markdown('<div class="welcome">Welcome to Dupes!</div>', unsafe_allow_html=True)
st.markdown("""
<div class="description">
This site is here to help you find cheaper alternatives to your favorite products.
Don't have a favorite yet? No worries. Just tell us what you're looking for in a shampoo,
and we'll help you find the product that best matches your description.
</div>
""", unsafe_allow_html=True)

# BUTTON ROW

col1, col2 = st.columns(2)

with col1:
    if st.button("Find your Dupe here!", key="btn_dupe"):
        st.switch_page("pages/shampoo.py")

with col2:
    if st.button("Find your Recommendation", key="btn_reco"):
        st.switch_page("./interface.py")
