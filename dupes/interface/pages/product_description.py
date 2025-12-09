import streamlit as st

st.set_page_config(page_title="Recommendation", layout="wide")

# ---------------------- CSS ----------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Montserrat', sans-serif;
}

/* Header */
.header {
    background-color: #d7d3ff;
    padding: 30px 60px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* Left navigation icons */
.nav-left {
    display: flex;
    gap: 70px;
    align-items: center;
}

.nav-item {
    text-align: center;
    cursor: pointer;
}

.nav-item img {
    width: 45px;
    opacity: 0.75;
    transition: 0.2s;
}

.nav-item:hover img {
    opacity: 1.0;
    transform: scale(1.07);
}

.nav-label {
    color: #726fa4;
    font-size: 18px;
    margin-top: -5px;
}

/* Right-aligned page title */
.page-title {
    font-size: 70px;
    color: #b8b4e2;
    font-weight: 600;
    letter-spacing: 8px;
}

/* Centered content */
.center {
    text-align: center;
    margin-top: 100px;
}

.heading {
    font-size: 32px;
    font-weight: 600;
    color: #7b75a7;
}

/* Search bar */
.search-box {
    width: 55%;
    margin: 40px auto;
    display: flex;
    border: 3px solid #6d6d6d;
    height: 60px;
    border-radius: 3px;
    padding-left: 15px;
    align-items: center;
}

.search-input input {
    border: none !important;
    outline: none !important;
    font-size: 18px;
    width: 100%;
    color: #7a75a8;
}

.search-icon img {
    width: 30px;
    margin-right: 20px;
}

.description-text {
    font-size: 20px;
    color: #8a8a8a;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)


# ---------------------- HEADER ----------------------
st.markdown("""
<div class="header">

    <div class="nav-left">

        <div class="nav-item" onclick="window.location.href='/'">
            <img src="https://img.icons8.com/ios-filled/100/766FA8/home.png">
            <div class="nav-label">home</div>
        </div>

        <div class="nav-item" onclick="window.location.href='?page=dupe'">
            <img src="https://img.icons8.com/ios/100/766FA8/search--v1.png">
            <div class="nav-label">Dupe Finder</div>
        </div>

    </div>

    <div class="page-title">RECOMMENDATION</div>

</div>
""", unsafe_allow_html=True)


# ---------------------- MAIN SEARCH SECTION ----------------------
st.markdown("""
<div class="center">

    <div class="heading">Product description</div>

    <div class="search-box">
        <div class="search-input">
            <input id="input_text" type="text" placeholder="description" />
        </div>
        <div class="search-icon">
            <img src="https://img.icons8.com/ios-glyphs/50/6e6e6e/search--v1.png">
        </div>
    </div>

    <div class="description-text">
        Let us know what you're looking for in your new shampoo.
    </div>

</div>
""", unsafe_allow_html=True)


# ---------------------- BACKEND HANDLING (optional) ----------------------
user_description = st.text_input("", label_visibility="collapsed")

if user_description:
    st.write("Your input:", user_description)
    # → Here you will call your API later
