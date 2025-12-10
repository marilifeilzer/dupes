import streamlit as st
st.set_page_config(layout="wide")
# ---------------- CSS: ENABLE TRUE HORIZONTAL SCROLL ----------------
st.markdown("""
<style>
.scroll-container {
    display: flex;
    overflow-x: auto;
    gap: 30px;
    padding: 20px;
    scroll-behavior: smooth;
}
.scroll-container::-webkit-scrollbar {
    height: 10px;
}
.scroll-container::-webkit-scrollbar-thumb {
    background: #C8C5DF;
    border-radius: 10px;
}
/* Cards */
.card {
    min-width: 260px;   /* IMPORTANT → makes scrolling possible */
    background: #EDE9E5;
    border-radius: 20px;
    padding: 20px;
    flex-shrink: 0;     /* Prevents cards from shrinking */
    text-align: center;
}
.card img {
    width: 90%;
    border-radius: 10px;
}
.card-title {
    background: #7D78AB;
    color: white;
    padding: 10px;
    border-radius: 20px;
    margin-top: 10px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)
# ---------------- HORIZONTAL SCROLL SECTION ----------------
st.markdown('<div class="scroll-container">', unsafe_allow_html=True)
products = [
    {"image": "https://m.media-amazon.com/images/I/61NfK8zl5zL._SL1500_.jpg", "name": "Dupe 1"},
    {"image": "https://m.media-amazon.com/images/I/61lqDpiR0FL._SL1500_.jpg", "name": "Dupe 2"},
    {"image": "https://m.media-amazon.com/images/I/61AjVKCKoIL._SL1500_.jpg", "name": "Dupe 3"},
    {"image": "https://m.media-amazon.com/images/I/81yK5uVKy3L._SL1500_.jpg", "name": "Dupe 4"}
]
for p in products:
    st.markdown(f"""
        <div class="card">
            <img src="{p['image']}">
            <div class="card-title">{p['name']}</div>
        </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
