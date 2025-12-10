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

/* Circular buttons container */
.circle-container {
    display: flex;
    justify-content: center;
    gap: 150px;
    margin-top: 60px;
}

/* Circular buttons base styling */
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
    transition: 0.3s;
}

/* Hover effect */
.circle-btn:hover {
    background-color: #6b6597;
}

/* Invisible Streamlit button */
.circle-wrapper {
    position: relative;
    width: 250px;
    height: 250px;
}

.circle-wrapper > div > button {
    position: absolute;
    top: 0;
    left: 0;
    width: 250px;
    height: 250px;
    opacity: 0;
    cursor: pointer;
.circle-container {
    display: flex;
    justify-content: center;
    gap: 150px;
    margin-top: 60px;
}

/* Make the actual Streamlit button look like your circle button */
div.stButton > button {
    width: 250px !important;
    height: 250px !important;
    border-radius: 50% !important;
    background-color: #7d78ab !important;
    color: white !important;
    font-size: 22px !important;
    font-weight: 600 !important;
    border: none !important;
    transition: 0.3s !important;
}

/* Hover effect */
div.stButton > button:hover {
    background-color: #6b6597 !important;
}
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

# ---------- CIRCULAR BUTTONS WITH NAVIGATION ----------
st.markdown('<div class="circle-container">', unsafe_allow_html=True)

# Button 1
st.markdown("""
<div class="circle-wrapper">
    <div class="circle-btn">Find your Dupe here!</div>
</div>
""", unsafe_allow_html=True)

if st.button("go_dupe", key="btn_dupe"):
    st.switch_page("pages/shampoo.py")

# Button 2
st.markdown("""
<div class="circle-wrapper">
    <div class="circle-btn">Find your Recommendation</div>
</div>
""", unsafe_allow_html=True)

if st.button("go_reco", key="btn_reco"):
    st.switch_page("interface.py")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="circle-container">', unsafe_allow_html=True)

col1, col2 = st.columns([1,1])

with col1:
    if st.button("Find your Dupe here!", key="go_dupe"):
        st.switch_page("pages/shampoo.py")

with col2:
    if st.button("Find your Recommendation", key="go_reco"):
        st.switch_page("interface.py")

st.markdown('</div>', unsafe_allow_html=True)
