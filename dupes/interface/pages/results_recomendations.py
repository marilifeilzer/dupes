import streamlit as st

st.set_page_config(layout="wide")

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

/* Navigation icons */
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
    font-size: 18px;
    color: #7d75aa;
    margin-top: -5px;
}

/* Title */
.page-title {
    font-size: 65px;
    font-weight: 600;
    color: #b6b3dd;
    letter-spacing: 6px;
    line-height: 1.1;
    text-align: right;
}

/* Main container */
.result-box {
    width: 75%;
    margin: 60px auto;
    background-color: #c7beb7;
    border-radius: 40px;
    padding: 50px;
    display: flex;
    justify-content: space-between;
    gap: 50px;
}

/* Left column */
.left-col {
    width: 40%;
    text-align: center;
}

.left-col img {
    width: 90%;
    border-radius: 20px;
}

/* Purple name button */
.name-btn {
    background-color: #7c78ad;
    color: white;
    padding: 15px;
    border-radius: 30px;
    margin-top: 20px;
    font-weight: 600;
    font-size: 18px;
}

/* Tags */
.tag-row {
    margin-top: 25px;
    display: flex;
    justify-content: center;
    gap: 20px;
}

.tag {
    background-color: #cfc9ff;
    color: white;
    border-radius: 30px;
    padding: 18px 22px;
    font-size: 14px;
    font-weight: 600;
    max-width: 140px;
}

/* Right column (description panel) */
.right-panel {
    width: 55%;
    background-color: #d7d3ff;
    border-radius: 40px;
    padding: 40px;
    display: flex;
    justify-content: center;
    align-items: center;
    color: white;
    font-size: 22px;
    font-weight: 600;
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

        <div class="nav-item" onclick="window.location.href='?page=recommendation'">
            <img src="https://img.icons8.com/ios/100/766FA8/search--v1.png">
            <div class="nav-label">recommendation</div>
        </div>

    </div>

    <div class="page-title">
        matches for your<br>description
    </div>

</div>
""", unsafe_allow_html=True)



# ---------------------- MAIN RESULT BOX ----------------------
product_image = "https://m.media-amazon.com/images/I/61NfK8zl5zL._SL1500_.jpg"
product_name = "product name"
tag1 = "most similar product"
tag2 = "Price $$, undervalue etc."
product_description = "product description"

st.markdown(f"""
<div class="result-box">

    <!-- LEFT COLUMN -->
    <div class="left-col">
        <img src="{product_image}" />

        <div class="name-btn">{product_name}</div>

        <div class="tag-row">
            <div class="tag">{tag1}</div>
            <div class="tag">{tag2}</div>
        </div>
    </div>

    <!-- RIGHT COLUMN -->
    <div class="right-panel">
        {product_description}
    </div>

</div>
""", unsafe_allow_html=True)
